import cv2

import os
import socket
import timeit
from datetime import datetime
from tensorboardX import SummaryWriter


# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
from dataloaders import davis_2017 as db
from dataloaders import custom_transforms as tr
from layers.osvos_layers import *
from mypath import Path
from networks.ms_osvos_var import MsOSVOS
from torch import nn
from util.utility import *

saveFileName = 'OnlineMo'
pretrain = 'pathToOfflineTrainedModel'

lr = 1e-6

def onlinelearning(seq_name):
    writer_ol = SummaryWriter('')

    scales = (0.5, 1.0)
    db_root_dir = Path.db_root_dir17()  # the location of Davis2016
    save_dir = Path.save_root_dir()  # the location of where we save the result

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    nAveGrad = 5  # Average the gradient every nAveGrad iterations
    nEpochs = 2000 * nAveGrad  # Number of epochs for training

    # Parameters in p are used for the name of the model
    p = {
        'trainBatch': 1,  # Number of Images in each mini-batch
    }
    seed = 0
    gpu_id = 0
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

    # Network definition
    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),  # Mirror of the lake
                                              tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                              tr.ToTensor()])
    db_train = db.DAVIS2017(train=True, db_root_dir=db_root_dir, transform=composed_transforms, seq_name=seq_name)
    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=0)

    db_test = db.DAVIS2017(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    # See how many objs

    all_pred = []

    for ii, (sample_batched, info) in enumerate(trainloader):
        input, gts = sample_batched['image'], sample_batched['gt']

        num_obj = gts.shape[1]
        print('obj num: ', num_obj)

    for o in range(num_obj):
        aveGrad = 0
        print("Start of Online Training, sequence: " + seq_name + ' obj: ', o)

        net = nn.DataParallel(MsOSVOS(), device_ids=[0])
        print('load pretrain:', pretrain)
        net.load_state_dict(torch.load(pretrain))
        net.to(device)

        # Use the following optimizer
        optimizer = optim.Adam(net.parameters(), lr=lr,
                               betas=(0.9, 0.999), weight_decay=5e-4)

        for epoch in range(0, nEpochs):
            print("Epoch: ", epoch, "/", nEpochs)
            # One training epoch
            running_loss_tr = 0  # The loss for the whole epoch
            np.random.seed(seed + epoch)

            # ii is the index and the sample_batched is content
            for ii, (sample_batched, info) in enumerate(trainloader):
                # input images and ground truths(labels)
                inputs = {}

                input, gts = sample_batched['image'], sample_batched['gt']

                gts = gts[:, o:o+1]

                for s in scales:
                    input_size = int(input.size(2) * s), int(input.size(3) * s)
                    inputs[str(s)] = torch.nn.functional.interpolate(
                        input, size=input_size, mode='bilinear', align_corners=False)
                    # print(inputs[str(s)].shape)

                # Forward-Backward of the mini-batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                gts = gts.to(device)

                outputs, attn, var = net.forward(inputs)

                # weight = torch.FloatTensor([num_labels_pos/ total_labels, num_labels_neg / total_labels])
                # exp_var = torch.exp(var)
                loss = lovasz_softmax(outputs, gts, classes='present', per_image=False, ignore=None)
                # loss = torch.mean(exp_var * loss)
                running_loss_tr += loss.item()

                # Print stuff
                if (epoch + 1) % 500 == 0:
                    J_F, preds = evaluation(net, testloader, device, seq_name, obj=o, save=False)
                    writer_ol.add_scalars(seq_name + '_obj' + str(o),
                                          {"Overall": J_F.J_F_avg,
                                           "J": J_F.J_avg,
                                           "F": J_F.F_avg
                                           },
                                          epoch+1)

                # Backward the averaged gradient
                loss /= nAveGrad
                loss.backward()
                aveGrad += 1

                if aveGrad % nAveGrad == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    aveGrad = 0

        save_dir_res = os.path.join(save_dir, saveFileName, seq_name)
        if not os.path.exists(save_dir_res):
            os.makedirs(save_dir_res)

        J_F, preds = evaluation(net, testloader, device, seq_name, obj=o, save=True)
        all_pred.append(preds)

        save_pth = os.path.join(save_dir, saveFileName, 'model')

        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        torch.save(net.state_dict(), os.path.join(save_pth, seq_name + '_obj_' + str(o) + '.pth'))


    all_mask = torch.zeros((len(all_pred[0]), num_obj, input.shape[2], input.shape[3]))


    for o in range(len(all_pred)):
        for index in range(len(all_pred[o])):
            all_mask[index, o] = torch.from_numpy(all_pred[o][index][1])


    for t in range(all_mask.shape[0]):
        frame = Soft_aggregation(all_mask[t], num_obj+1)
        frame = torch.argmax(frame[0], dim=0)
        im = Image.fromarray(frame.numpy().astype(np.uint8)).convert('P')
        im.putpalette(info)
        im.save(os.path.join(save_dir_res, '{:05d}.png'.format(t)), format='PNG')

    writer_ol.close()


def Soft_aggregation(ps, K):
    num_objects, H, W = ps.shape
    em = torch.zeros(1, K, H, W)
    em[0, 0] = torch.prod(1 - ps, dim=0)  # bg prob
    em[0, 1:num_objects + 1] = ps  # obj prob
    em = torch.clamp(em, 1e-7, 1 - 1e-7)
    logit = torch.log((em / (1 - em)))
    return logit





def evaluation(net, testloader, device, seq_name, obj=0, save=False):
    print('Testing Network')
    scales = (0.5, 1.0)
    J_F = JFAverageMeter()
    var_list = []
    pred_list = []


    with torch.no_grad():  # PyTorch 0.4.0 style
        # Main Testing Loop
        for ii, (sample_batched, info) in enumerate(testloader):

            # fname = 'train_seqs' or 'val_seqs'
            img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

            gt = gt[:, obj:obj + 1]

            if gt.shape[1] == 0:
                gt = torch.zeros((1, 1, gt.shape[2], gt.shape[3]))
            inputs = {}

            for s in scales:
                input_size = int(img.size(2) * s), int(img.size(3) * s)
                inputs[str(s)] = torch.nn.functional.interpolate(
                    img, size=input_size, mode='bilinear', align_corners=False)

            # Forward-Backward of the mini-batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            gt = gt.to(device)

            outputs, attn, variance = net.forward(inputs, nscale=False, scales=scales)

            var_list.append(variance.cpu().data.numpy())


            for index in range(outputs.shape[0]):
                pred = outputs.cpu().data.numpy()[index, :, :, :]
                pred = np.argmax(pred, axis=0)
                ms = gt.cpu().data.numpy()[index, 0, :, :]

                J_F.update(db_eval_iou(pred, ms),
                           db_eval_boundary(pred, ms))

            if save:
                for index in range(outputs.shape[0]):
                    pred = outputs.cpu().data.numpy()[index, :, :, :]
                    pred_list.append(pred)
                #
                #     pred = np.argmax(pred, axis=0)
                #     attnMap = attn.cpu().data.numpy()[index, 0]
                #     cv2.imwrite(os.path.join(save_dir_res, os.path.basename(fname[index]) + '.png'), pred * 255)
                #     cv2.imwrite(os.path.join(save_dir_res, 'Attn' + os.path.basename(fname[index]) + '.png'), attnMap * 255)
                #
                #     var = var_list[index][0]
                #     fig = plt.figure()
                #     plt.title('Variance: {}'.format(np.sum(var)))
                #     plt.axis('off')
                #     heatmap = plt.imshow(var, cmap='viridis')
                #     fig.colorbar(heatmap)
                #     save_pth = os.path.join(save_dir_res, 'var{:05d}'.format(index) + '.png')
                #     fig.savefig(save_pth)
                #
                # if not os.path.exists(save_dir_res+'_model'):
                #     os.makedirs(save_dir_res+'_model')

                # torch.save(net.state_dict(), os.path.join(save_dir_res+'_model', seq_name + '_obj_' + str(obj) + '.pth'))

    print('***********************Eval: [Avg %.5f], [J %.5f],[F %.5f]'
          % (J_F.J_F_avg, J_F.J_avg, J_F.F_avg))
    return J_F, pred_list


if __name__ == '__main__':
    with open(os.path.join(Path.db_root_dir17(), 'ImageSets/2017', 'val_seqs1.txt'), "r") as lines:
        for line in lines:
            video = line.rstrip('\n')
            onlinelearning(video)
