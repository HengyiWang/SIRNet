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
from dataloaders import davis_2016 as db
from dataloaders import custom_transforms as tr
from layers.osvos_layers import *
from mypath import Path
from networks.ms_osvos_var import MsOSVOS
from torch import nn
from util.utility import *
import matplotlib.pyplot as plt
from networks.inter_loss import *

# Loss func settings

# writer_ol = SummaryWriter('')
lr = 1e-6
online_epochs = 5
saveFileName = 'sup_so' + str(lr) + '_' + str(online_epochs)


def onlinelearning(seq_name, pretrain=None):
    InterLoss_cuda = InterLoss().cuda()
    inter_loss_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
    inter_loss_radius = 5
    criterion = nn.CrossEntropyLoss(reduction='none')

    scales = (0.5, 1.0)
    db_root_dir = Path.db_root_dir()  # the location of Davis2016
    save_dir = Path.save_root_dir()  # the location of where we save the result

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    nEpochs = 1  # Number of epochs for training
    # Parameters in p are used for the name of the model
    p = {
        'trainBatch': 1,  # Number of Images in each mini-batch
    }
    seed = 0
    gpu_id = 0
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

    # Network definition
    net = nn.DataParallel(MsOSVOS(), device_ids=[0])
    print('load pretrain:', pretrain)
    net.load_state_dict(torch.load(pretrain))
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr,
                           betas=(0.9, 0.999), weight_decay=5e-4)
    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),  # Mirror of the lake
                                              tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                              tr.ToTensor()])
    db_train = db.DAVIS2016(train=True, db_root_dir=db_root_dir, transform=composed_transforms, seq_name=seq_name)
    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=1)
    db_test = db.DAVIS2016(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    print("Start of Online Training, sequence: " + seq_name)
    start_time = timeit.default_timer()

    # Test
    J_F = evaluation(net, testloader, device, save=False, save_dir_res='')

    # writer_ol.add_scalars(seq_name,
    #                       {"Overall": J_F.J_F_avg,
    #                        "J": J_F.J_avg,
    #                        "F": J_F.F_avg
    #                        },
    #                       0)

    # Save
    save_dir_res = os.path.join(save_dir, saveFileName, seq_name)
    if not os.path.exists(save_dir_res):
        os.makedirs(save_dir_res)


    # Main Training and Testing Loop
    for epoch in range(0, nEpochs):
        print("Epoch: ", epoch, "/", nEpochs)
        np.random.seed(seed + epoch)
        last_pred = None
        last_frame = None

        preds = []
        attns = []

        J_F = JFAverageMeter()



        for ii, sample_batched in enumerate(testloader):
            img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

            gt = torch.cat((1 - gt, gt), dim=1)

            # image [1, 3, 480, 854]
            # gt [1, 1, 480, 854]

            inputs = {}
            for s in scales:
                input_size = int(img.size(2) * s), int(img.size(3) * s)
                inputs[str(s)] = torch.nn.functional.interpolate(
                    img, size=input_size, mode='bilinear', align_corners=False)
                # print(inputs[str(s)].shape)

            # Forward-Backward of the mini-batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            gt = gt.to(device)


            # Online Adaptation
            if last_pred is None:

                with torch.no_grad():
                    last_pred = gt
                    last_frame = inputs['1.0']

                    outputs, attn, variance = net.forward(inputs, nscale=False, scales=scales)

                    pred = outputs.cpu().data.numpy()[0, :, :, :]
                    pred = np.argmax(pred, axis=0)
                    ms = gt.cpu().data.numpy()[0, 0, :, :]

                    attnMap = attn.cpu().data.numpy()[0, 0]


                    J = db_eval_iou(pred, ms)
                    F = db_eval_boundary(pred, ms)

                    preds.append(pred)
                    attns.append(attnMap)
                    # print('[Frame %d]: [Avg %.5f], [J %.5f],[F %.5f]'
                    #       % (ii, (J + F) / 2, J, F))

            else:

                with torch.no_grad():
                    pseudo, attn, variance = net.forward(inputs, nscale=False, scales=scales)
                    exp_var = torch.exp(-variance.detach())

                    pseudo = torch.argmax(pseudo, dim=1)


                for o_pch in range(0, online_epochs):

                    outputs, attn, variance = net.forward(inputs, nscale=False, scales=scales)
                    sample = {'cur': inputs['1.0'], 'prev': last_frame}
                    outputs_pk = {'cur': outputs, 'prev': last_pred}

                    loss = criterion(outputs, pseudo)
                    loss = torch.mean(exp_var * loss)
                    inter_loss = \
                        InterLoss_cuda(outputs_pk, inter_loss_kernels_desc_defaults, inter_loss_radius, sample,
                                 inputs['1.0'].shape[2],
                                 inputs['1.0'].shape[3])
                    loss = inter_loss + loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # Eval
                    with torch.no_grad():
                        outputs, attn, variance = net.forward(inputs, nscale=False, scales=scales)

                        pred = outputs.cpu().data.numpy()[0, :, :, :]
                        pred = np.argmax(pred, axis=0)
                        ms = gt.cpu().data.numpy()[0, 1, :, :]

                        attnMap = attn.cpu().data.numpy()[0, 0]

                        J = db_eval_iou(pred, ms)
                        F = db_eval_boundary(pred, ms)

                        # print('[Frame %d]: [Avg %.5f], [J %.5f],[F %.5f]'
                        #      % (ii, (J + F) / 2, J, F))


                last_pred = outputs
                last_frame = inputs['1.0']
                preds.append(pred)
                attns.append(attnMap)

            cv2.imwrite(os.path.join(save_dir_res, os.path.basename(fname[0]) + '.png'), pred * 255)
            cv2.imwrite(os.path.join(save_dir_res, 'Attn' + os.path.basename(fname[0]) + '.png'),
                        attnMap * 255)

            J_F.update(J, F)


        # Evaluation
        print('***********************Eval: [Avg %.5f], [J %.5f],[F %.5f]'
              % (J_F.J_F_avg, J_F.J_avg, J_F.F_avg))



        for index, ele in enumerate(preds):
            cv2.imwrite(os.path.join(save_dir_res, os.path.basename(fname[0]) + '.png'), ele * 255)
            cv2.imwrite(os.path.join(save_dir_res, 'Attn' + os.path.basename(fname[0]) + '.png'),
                        attns[index] * 255)





def evaluation(net, testloader, device, save=False, save_dir_res=''):
    print('Testing Network')
    scales = (0.5, 1.0)
    J_F = JFAverageMeter()
    var_list = []

    with torch.no_grad():  # PyTorch 0.4.0 style
        # Main Testing Loop
        for ii, sample_batched in enumerate(testloader):

            # fname = 'train_seqs' or 'val_seqs'
            img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

            # Forward of the mini-batch

            inputs = {}

            # print('input.shape', input.shape)
            # print('gts.shape', gts.shape)
            for s in scales:
                input_size = int(img.size(2) * s), int(img.size(3) * s)
                inputs[str(s)] = torch.nn.functional.interpolate(
                    img, size=input_size, mode='bilinear', align_corners=False)
                # print(inputs[str(s)].shape)

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
                    pred = np.argmax(pred, axis=0)
                    attnMap = attn.cpu().data.numpy()[index, 0]
                    cv2.imwrite(os.path.join(save_dir_res, os.path.basename(fname[index]) + '.png'), pred * 255)
                    cv2.imwrite(os.path.join(save_dir_res, 'Attn' + os.path.basename(fname[index]) + '.png'), attnMap * 255)

                    var = var_list[index][0]
                    fig = plt.figure()
                    plt.title('Variance: {}'.format(np.sum(var)))
                    plt.axis('off')
                    heatmap = plt.imshow(var, cmap='viridis')
                    fig.colorbar(heatmap)
                    save_pth = os.path.join(save_dir_res, 'var{:05d}'.format(index) + '.png')
                    fig.savefig(save_pth)

    print('***********************Eval: [Avg %.5f], [J %.5f],[F %.5f]'
          % (J_F.J_F_avg, J_F.J_avg, J_F.F_avg))
    return J_F


if __name__ == '__main__':
    with open(os.path.join(Path.db_root_dir(), 'val_seqs.txt'), "r") as lines:
        for line in lines:
            video = line.rstrip('\n')
            pretrain = ''

            onlinelearning(video, pretrain)
