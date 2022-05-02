# Package Includes
from __future__ import division

import os
import socket
import timeit
import scipy.misc as sm
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter

# PyTorch includes
import torch
import cv2
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
from util.utility import *
from dataloaders import davis_2016 as db
from dataloaders import custom_transforms as tr
from networks.ms_osvos_var import MsOSVOS
from layers.osvos_layers import *
from mypath import Path
from torch import nn

# Select which GPU, -1 if CPU
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

# Setting of parameters
# Parameters in p are used for the name of the model
p = {
    'trainBatch': 8,  # Number of Images in each mini-batch
}

# # Setting other parameters
modelName = 'Ms_OSVOS_var'
resume_epoch = 0  # Default is 0, change if want to resume(restart)
nEpochs = 240  # Number of epochs for training (500.000/2079)
useTest = True  # See evolution of the test set when training?
testBatch = 4  # Testing Batch
nTestInterval = 5  # Run on test set every nTestInterval epochs
db_root_dir = Path.db_root_dir()
vis_net = 0  # Visualize the network?
snapshot = 1  # Store a model every snapshot epochs
nAveGrad = 10
load_caffe_vgg = True
save_dir = Path.save_root_dir()
if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))


net = nn.DataParallel(MsOSVOS(), device_ids=[0, 1, 2, 3])
path_to_model = '/pathToModel'
print('Loading weights:', path_to_model)
net.load_state_dict(torch.load(path_to_model), strict=False)

# Logging into Tensorboard
log_dir = os.path.join(save_dir, 'runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
writer = SummaryWriter(log_dir=log_dir, comment='-parent')
net.to(device)

# Use the following optimizer
lr = 1e-5
wd = 0.0002
optimizer = optim.Adam(net.parameters(), lr=lr,
                        betas=(0.9, 0.999), weight_decay=5e-4)

composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                          tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.5)),
                                          tr.ToTensor()])
# Training dataset and its iterator
db_train = db.DAVIS2016(train=True, inputRes=None, db_root_dir=db_root_dir, transform=composed_transforms)
trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2, drop_last=False)

# Testing dataset and its iterator
db_test = db.DAVIS2016(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor())
testloader = DataLoader(db_test, batch_size=testBatch, shuffle=False, num_workers=2)

num_img_tr = len(trainloader)
num_img_ts = len(testloader)
loss_tr = []
loss_ts = []
aveGrad = 0
best_JF = 0

print("Training Network")
# Main Training and Testing Loop
for epoch in range(resume_epoch, nEpochs):
    scales = (0.5, 1.0)
    start_time = timeit.default_timer()
    # One training epoch
    for ii, sample_batched in enumerate(trainloader):

        inputs = {}
        # shape: [Bs, C, H, W]
        input, gts = sample_batched['image'], sample_batched['gt']

        for s in scales:
            input_size = int(input.size(2) * s), int(input.size(3) * s)
            inputs[str(s)] = torch.nn.functional.interpolate(
        input, size=input_size, mode='bilinear', align_corners=False)

        # Forward-Backward of the mini-batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gts = gts.to(device)
        outputs, attn, var = net.forward(inputs)
        exp_var = torch.exp(var)
        loss = lovasz_softmax(outputs, gts, classes='present', per_image=False, ignore=None)
        loss = torch.mean(exp_var * loss)

        # Print stuff
        if ii % num_img_tr == num_img_tr - 1:
            writer.add_scalar('data/total_loss_epoch', loss, epoch)
            print('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
            print('Loss: %f' % (loss))

            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time))

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

    # One testing epoch
    J_F = JFAverageMeter()

    if useTest and epoch % nTestInterval == (nTestInterval - 1):
        inputs = {}
        with torch.no_grad():
            scales = (0.5, 1.0)

            for ii, sample_batched in enumerate(testloader):
                input, gts = sample_batched['image'], sample_batched['gt']

                # Forward pass of the mini-batch
                gts = gts.to(device)

                for s in scales:
                    input_size = int(input.size(2) * s), int(input.size(3) * s)
                    inputs[str(s)] = torch.nn.functional.interpolate(
                        input, size=input_size, mode='bilinear', align_corners=False)

                    # Forward-Backward of the mini-batch
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs, attn, var = net.forward(inputs, scales=scales)

                for index in range(outputs.shape[0]):
                    pred = outputs.cpu().data.numpy()[index, :, :, :]
                    pred = np.argmax(pred, axis=0)
                    ms = gts.cpu().data.numpy()[index, 0, :, :]

                    J_F.update(db_eval_iou(pred, ms),
                               db_eval_boundary(pred, ms))


            print('***********************Eval: [Avg %.5f], [J %.5f],[F %.5f]'
                  % (J_F.J_F_avg, J_F.J_avg, J_F.F_avg))

            if J_F.J_F_avg > best_JF:
                best_JF = J_F.J_F_avg
                print('\n Best JF: ', best_JF)
                print('Save model\n')
                torch.save(net.state_dict(),
                           os.path.join(save_dir, modelName + '_epoch-' + str(epoch) + '.pth'))

writer.close()
