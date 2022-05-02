from torch.utils.data import DataLoader

# Custom includes
import os
from dataloaders import davis_2017 as db
from dataloaders import custom_transforms as tr
from layers.osvos_layers import *
from mypath import Path
from networks.ms_osvos_var import MsOSVOS
from torch import nn
from util.utility import *
import matplotlib.pyplot as plt

model_pth = ''
saveFileName = 'Result_multi_winner_takes_all'


def evaluate(seq_name):
    save_dir = Path.save_root_dir()
    save_dir_res = os.path.join(save_dir, saveFileName, seq_name)
    if not os.path.exists(save_dir_res):
        os.makedirs(save_dir_res)

    scales = (0.5, 1.0)
    db_root_dir = Path.db_root_dir17()  # the location of Davis2016
    save_dir = Path.save_root_dir()  # the location of where we save the result

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    gpu_id = 0
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

    # Network definition
    db_train = db.DAVIS2017(train=True, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name)
    trainloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=0)
    db_test = db.DAVIS2017(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    all_pred = None

    for ii, (sample_batched, info) in enumerate(trainloader):
        input, gts = sample_batched['image'], sample_batched['gt']

        num_obj = gts.shape[1]
        print('obj num: ', num_obj)

    for o in range(num_obj):
        pretrain = os.path.join(model_pth, seq_name + '_obj_' + str(o) + '.pth')
        print("Start of Online Evaluation, sequence: " + seq_name + ' obj: ', o)
        net = nn.DataParallel(MsOSVOS(), device_ids=[0])
        print('load pretrain:', pretrain)
        net.load_state_dict(torch.load(pretrain))
        net.to(device)

        J_F, preds = evaluation(net, testloader, device, seq_name, obj=o, save=True)
        if all_pred is None:
            all_pred = preds
        else:
            all_pred = torch.cat((all_pred, preds), dim=1)

    background_mask = all_pred.max(dim=1, keepdim=True)[0].lt(0.5)
    all_pred_argmax = all_pred.argmax(dim=1, keepdim=True).float() + 1.0


    all_pred_argmax[background_mask] = 0

    for t in range(all_pred_argmax.shape[0]):
        frame = all_pred_argmax[t, 0]
        im = Image.fromarray(frame.numpy().astype(np.uint8)).convert('P')
        im.putpalette(info)
        im.save(os.path.join(save_dir_res, '{:05d}.png'.format(t)), format='PNG')





def evaluation(net, testloader, device, seq_name, obj=0, save=False):
    print('Testing Network')
    scales = (0.5, 1.0)
    J_F = JFAverageMeter()
    var_list = []
    predictions = None


    with torch.no_grad():  # PyTorch 0.4.0 style
        # Main Testing Loop
        for ii, (sample_batched, info) in enumerate(testloader):

            # fname = 'train_seqs' or 'val_seqs'
            img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

            gt = gt[:, obj:obj + 1]

            if gt.shape[1] == 0:
                gt = torch.zeros((1, 1, gt.shape[2], gt.shape[3]))

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
                    pred = outputs.cpu().data[:, 1:2]

                    if predictions is None:
                        predictions = pred
                    else:
                        predictions = torch.cat((predictions, pred), dim=0)


    print('***********************Eval: [Avg %.5f], [J %.5f],[F %.5f]'
          % (J_F.J_F_avg, J_F.J_avg, J_F.F_avg))
    return J_F, predictions


if __name__ == '__main__':
    pathToData = ''
    with open(pathToData, "r") as lines:
        for line in lines:
            video = line.rstrip('\n')
            evaluate(video)