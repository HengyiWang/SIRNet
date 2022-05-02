import torch
import torch.nn.functional as F


class InterLoss(torch.nn.Module):

    # yhat 'cur', 'prev'
    # sample 'cur', 'prev'


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
        # print(kernels.shape)

        cur_hat = self._unfold(y_hat_softmax['cur'], 0)
        prev_hat = self._unfold(y_hat_softmax['prev'], kernels_radius)

        y_hat_unfolded = torch.abs(prev_hat - cur_hat)


        # print(y_hat_unfolded.shape)

        loss = torch.mean((kernels * y_hat_unfolded).view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred).sum(dim=2, keepdim=True))

        return loss

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
