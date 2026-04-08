"""Rectified Flow core logic for conditional segmentation.

Implements the linear interpolation flow:
    x_t = t * x_1 + (1 - t) * z_0
with velocity target:
    v = x_1 - z_0
and L2 velocity-matching loss.
"""

import torch


class RectifiedFlow:
    """Rectified flow for conditional segmentation.

    The flow defines a straight-line path from noise (t=0) to data (t=1):
        x_t = t * x_1 + (1 - t) * z_0
    where z_0 ~ N(0, I) and x_1 is the ground truth segmentation.

    The model learns to predict the velocity v = x_1 - z_0.
    """

    def __init__(self, eps=1e-3, time_sampling="uniform", logit_normal_m=0.0, logit_normal_s=1.0):
        """
        Args:
            eps: Small offset to avoid t=0 (numerical stability).
            time_sampling: Time sampling strategy, either "uniform" or "logit_normal".
            logit_normal_m: Location parameter for logit-normal sampling.
                Negative values bias towards data (t~0), positive towards noise (t~1).
            logit_normal_s: Scale parameter for logit-normal sampling.
                Controls the width of the distribution.
        """
        self.eps = eps
        self.T = 1.0
        self.time_sampling = time_sampling
        self.logit_normal_m = logit_normal_m
        self.logit_normal_s = logit_normal_s

    def get_z0(self, shape, device):
        """Sample initial noise z_0 ~ N(0, I).

        Args:
            shape: Shape of the noise tensor (B, C, H, W).
            device: Torch device.

        Returns:
            Gaussian noise tensor.
        """
        return torch.randn(shape, device=device)

    def sample_t(self, batch_size, device):
        """Sample timesteps according to the configured strategy.

        Supports:
        - "uniform": t ~ Uniform(eps, T)
        - "logit_normal": sample u ~ N(m, s), then t = sigmoid(u), clamped to [eps, T].
          See: https://openreview.net/pdf?id=FPnUhsQJ5B

        Args:
            batch_size: Number of samples.
            device: Torch device.

        Returns:
            Tensor of shape (batch_size,) with values in [eps, T].
        """
        if self.time_sampling == "logit_normal":
            u = torch.randn(batch_size, device=device) * self.logit_normal_s + self.logit_normal_m
            t = torch.sigmoid(u)
            t = t.clamp(self.eps, self.T)
            return t
        # Default: uniform
        return torch.rand(batch_size, device=device) * (self.T - self.eps) + self.eps

    def interpolate(self, x_1, z_0, t):
        """Compute the interpolated sample x_t = t * x_1 + (1 - t) * z_0.

        Args:
            x_1: Ground truth data (B, C, H, W).
            z_0: Noise (B, C, H, W).
            t: Time values (B,).

        Returns:
            Interpolated tensor x_t (B, C, H, W).
        """
        t_expand = t.view(-1, 1, 1, 1)
        return t_expand * x_1 + (1 - t_expand) * z_0

    def velocity_target(self, x_1, z_0):
        """Compute the velocity target v = x_1 - z_0.

        Args:
            x_1: Ground truth data (B, C, H, W).
            z_0: Noise (B, C, H, W).

        Returns:
            Velocity tensor (B, C, H, W).
        """
        return x_1 - z_0

    def loss(self, model, x_1, cond_img):
        """Compute the rectified flow velocity-matching loss.

        Args:
            model: Neural network that takes (x_t, t, cond_img=mri) and returns
                   predicted velocity.
            x_1: Ground truth segmentation (B, 3, H, W).
            cond_img: MRI conditioning image (B, 4, H, W).

        Returns:
            Scalar loss (mean L2 over batch and spatial dims).
        """
        z_0 = self.get_z0(x_1.shape, x_1.device)
        t = self.sample_t(x_1.shape[0], x_1.device)
        x_t = self.interpolate(x_1, z_0, t)
        target = self.velocity_target(x_1, z_0)
        pred = model(x_t, t, cond_img=cond_img)
        loss = torch.mean((pred - target) ** 2)
        return loss
