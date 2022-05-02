import numpy as np
from PIL import Image
import os
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import cv2

device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class InterLoss(torch.nn.Module):

    def forward(
            self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input,
            mask_src=None, mask_dst=None, compatibility=None, custom_modality_downsamplers=None, out_kernels_vis=False
    ):
        assert y_hat_softmax['cur'].dim() == 4, 'Prediction must be a NCHW batch'
        N, C, height_pred, width_pred = y_hat_softmax['cur'].shape

        device = y_hat_softmax['cur'].device

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
               width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'


        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
        )
        cur_hat = self._unfold(y_hat_softmax['cur'], 0)
        prev_hat = self._unfold(y_hat_softmax['prev'], kernels_radius)

        y_hat_unfolded = torch.abs(prev_hat - cur_hat)
        loss = (kernels * y_hat_unfolded).view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred).sum(dim=2,
                                                                                                         keepdim=True)

        return torch.mean(loss,dim=1)
        # return out

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            distance = InterLoss._get_mesh(N, height_pred, width_pred, device) / desc['xy']

            cur = sample['cur'] / desc['rgb']
            prev = sample['prev'] / desc['rgb']

            prev_feat = torch.cat((prev, distance), dim=1)
            cur = torch.cat((cur, distance), dim=1)


            kernel = weight * InterLoss._create_kernels_from_features(prev_feat, cur, kernels_radius)

            kernels = kernel if kernels is None else kernel + kernels

        return kernels

    @staticmethod
    def _create_kernels_from_features(prev, cur, radius):
        N, C, H, W = prev.shape

        kernels = InterLoss._unfold(prev, radius)
        target = InterLoss._unfold(cur, 0)  # (N, C-2, diameter, diameter, H, W)

        kernels = kernels - target
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()

        return kernels

    @staticmethod
    def _unfold(img, radius):
        # Extract the patch
        # Padding = radius
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _get_mesh(N, H, W, device):
        # Encode the row & column distance via such function
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

def convert_mask(mask, max_obj):

    # convert mask to one hot encoded
    # [240, 427] --> [240, 427, 2]
    oh = []
    for k in range(max_obj+1):
        oh.append(mask==k)

    if isinstance(mask, np.ndarray):
        oh = np.stack(oh, axis=-1)
    else:
        oh = torch.cat(oh, dim=-1).float()

    return oh




pthToCurFrame = 'Results/frames/00070.jpg'
pthToCurMsk = 'Results/masks/00070.png'
pthToPrevFrame = 'Results/frames/00069.jpg'
pthToPrevMsk = 'Results/masks/00069.png'

curFrame = np.array(Image.open(pthToCurFrame)).astype(float) / 255.0
prevFrame = np.array(Image.open(pthToPrevFrame)).astype(float) / 255.0
curMsk = np.array(Image.open(pthToCurMsk))
prevMsk = np.array(Image.open(pthToPrevMsk))


curMsk = np.expand_dims(convert_mask(curMsk, 3).transpose(2, 0, 1), 0)
prevMsk = np.expand_dims(convert_mask(prevMsk, 3).transpose(2, 0, 1), 0)

curMsk = torch.from_numpy(curMsk).float()
prevMsk = torch.from_numpy(prevMsk).float()

curFrame = np.expand_dims(curFrame.transpose(2, 0, 1), 0)
prevFrame = np.expand_dims(prevFrame.transpose(2, 0, 1), 0)

curFrame = torch.from_numpy(curFrame).float()
prevFrame = torch.from_numpy(prevFrame).float()


inter_loss_cuda = InterLoss().to(device1)
# inter_loss = InterLoss()
inter_loss_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
inter_loss_radius = 5
curFrame = curFrame.to(device1)
prevFrame = prevFrame.to(device1)
curMsk = curMsk.to(device1)
prevMsk = prevMsk.to(device1)

sample = {'cur': curFrame, 'prev': prevFrame}
outputs_pk = {'cur': curMsk, 'prev': prevMsk}

inter_loss = inter_loss_cuda(outputs_pk, inter_loss_kernels_desc_defaults, inter_loss_radius, sample,
                         prevFrame.shape[2],
                         prevFrame.shape[3])

fig = plt.figure()
plt.axis('off')
heatmap = plt.imshow(inter_loss.cpu().data[0, 0], cmap='plasma')
fig.savefig('./Results/interLoss_heat.png')






