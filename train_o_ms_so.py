# Package Includes
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



#writer_ol = SummaryWriter('')
saveFileName = 'online'
pretrain = 'pathToOfflineTrainedModel'


def onlinelearning(seq_name):
    scales = (0.5, 1.0)
    db_root_dir = Path.db_root_dir()  # the location of Davis2016
    save_dir = Path.save_root_dir()  # the location of where we save the result

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    nAveGrad = 5  # Average the gradient every nAveGrad iterations
    nEpochs = 2000 * nAveGrad  # Number of epochs for training
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


    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'runs',
                           datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname() + '-' + seq_name)
    writer = SummaryWriter(log_dir=log_dir)

    net.to(device)

    lr = 1e-6

    optimizer = optim.Adam(net.parameters(), lr=lr,
                           betas=(0.9, 0.999), weight_decay=5e-4)

    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),  # Mirror of the lake
                                              tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                              tr.ToTensor()])
    # Training dataset and its iterator
    # Training dataset would be only one frame from one sequence
    db_train = db.DAVIS2016(train=True, db_root_dir=db_root_dir, transform=composed_transforms, seq_name=seq_name)
    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=1)

    # Testing dataset and its iterator
    # Testing dataset would have the whole sequence
    db_test = db.DAVIS2016(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    loss_tr = []
    aveGrad = 0

    print("Start of Online Training, sequence: " + seq_name)
    start_time = timeit.default_timer()
    # Main Training and Testing Loop
    for epoch in range(0, nEpochs):
        print("Epoch: ", epoch, "/", nEpochs)
        # One training epoch
        running_loss_tr = 0  # The loss for the whole epoch
        np.random.seed(seed + epoch)

        # ii is the index and the sample_batched is content
        for ii, sample_batched in enumerate(trainloader):

            # input images and ground truths(labels)
            inputs = {}

            input, gts = sample_batched['image'], sample_batched['gt']

            # Forward-Backward of the mini-batch
            for s in scales:
                input_size = int(input.size(2) * s), int(input.size(3) * s)
                inputs[str(s)] = torch.nn.functional.interpolate(
                    input, size=input_size, mode='bilinear', align_corners=False)

            # Forward-Backward of the mini-batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            gts = gts.to(device)

            outputs, attn, var = net.forward(inputs)

            # Compute the losses, side outputs and fuse
            loss = lovasz_softmax(outputs, gts, classes='present', per_image=False, ignore=None)
            running_loss_tr += loss.item()

            # Print stuff
            if (epoch + 1) % 500 == 0:
                J_F = evaluation(net, testloader, device, seq_name, save=False, save_dir_res='')

                # writer_ol.add_scalars(seq_name,
                #                       {"Overall": J_F.J_F_avg,
                #                        "J": J_F.J_avg,
                #                        "F": J_F.F_avg
                #                        },
                #                       epoch+1)

            # Backward the averaged gradient
            loss /= nAveGrad
            loss.backward()
            aveGrad += 1

            # Update the weights once in nAveGrad forward passes
            if aveGrad % nAveGrad == 0:
                writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

    stop_time = timeit.default_timer()
    print('Online training time: ' + str(stop_time - start_time))

    save_dir_res = os.path.join(save_dir, saveFileName, seq_name)
    if not os.path.exists(save_dir_res):
        os.makedirs(save_dir_res)

    J_F = evaluation(net, testloader, device, seq_name, save=True, save_dir_res=save_dir_res)

    writer.close()

def evaluation(net, testloader, device, seq_name, save=False, save_dir_res=''):
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

                if not os.path.exists(save_dir_res+'_model'):
                    os.makedirs(save_dir_res+'_model')

                torch.save(net.state_dict(), os.path.join(save_dir_res+'_model', seq_name + '.pth'))

    print('***********************Eval: [Avg %.5f], [J %.5f],[F %.5f]'
          % (J_F.J_F_avg, J_F.J_avg, J_F.F_avg))
    return J_F


if __name__ == '__main__':
    with open(os.path.join(Path.db_root_dir(), 'val_seqs4.txt'), "r") as lines:
        for line in lines:
            video = line.rstrip('\n')
            onlinelearning(video)

