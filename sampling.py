"""ODE samplers for rectified flow inference.

Provides Euler (fixed-step) and RK45 (adaptive) ODE solvers to integrate
the learned velocity field from t=eps to t=1, producing segmentation masks
from random noise.
"""

import numpy as np
import torch
from scipy import integrate


@torch.no_grad()
def euler_sample(model, cond_img, shape, device, N=100, eps=1e-3):
    """Fixed-step Euler ODE solver for rectified flow inference.

    Integrates dx/dt = v_theta(x, t) from t=eps to t=1 using N steps.

    Args:
        model: Neural network that takes (x, t, cond_img=mri) -> velocity.
        cond_img: MRI conditioning image (B, 4, H, W).
        shape: Shape of the output segmentation (B, C, H, W).
        device: Torch device.
        N: Number of Euler steps.
        eps: Small offset for starting time.

    Returns:
        Predicted segmentation at t=1 (B, C, H, W).
    """
    z_0 = torch.randn(shape, device=device)
    x = z_0
    dt = (1.0 - eps) / N

    for i in range(N):
        t_val = i / N * (1.0 - eps) + eps
        t = torch.full((shape[0],), t_val, device=device)
        v = model(x, t, cond_img=cond_img)
        x = x + v * dt

    return x


@torch.no_grad()
def rk45_sample(model, cond_img, shape, device, atol=1e-5, rtol=1e-5, eps=1e-3):
    """Adaptive RK45 ODE solver using scipy.integrate.solve_ivp.

    More accurate than Euler but slower. Uses adaptive step sizing.

    Args:
        model: Neural network that takes (x, t, cond_img=mri) -> velocity.
        cond_img: MRI conditioning image (B, 4, H, W).
        shape: Shape of the output segmentation (B, C, H, W).
        device: Torch device.
        atol: Absolute tolerance for the ODE solver.
        rtol: Relative tolerance for the ODE solver.
        eps: Small offset for starting time.

    Returns:
        Tuple of (predicted segmentation at t=1, number of function evaluations).
    """
    z_0 = torch.randn(shape, device=device)

    def to_flattened_numpy(x):
        return x.detach().cpu().numpy().reshape((-1,))

    def from_flattened_numpy(x, shape):
        return torch.from_numpy(x.reshape(shape)).float().to(device)

    nfe = [0]

    def ode_func(t, x):
        nfe[0] += 1
        x_tensor = from_flattened_numpy(x, shape)
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.float32)
        with torch.no_grad():
            drift = model(x_tensor, t_tensor, cond_img=cond_img)
        return to_flattened_numpy(drift)

    solution = integrate.solve_ivp(
        ode_func,
        (eps, 1.0),
        to_flattened_numpy(z_0),
        rtol=rtol,
        atol=atol,
        method='RK45',
    )

    x = from_flattened_numpy(solution.y[:, -1], shape)
    return x, nfe[0]
