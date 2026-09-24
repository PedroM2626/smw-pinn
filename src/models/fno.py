"""
fno.py
Neural Operator Baseline: Fourier Neural Operator (Li et al., 2021) for
one-step dynamics (README 10.42).

FNO parameterizes the operator's integral kernel in the spectral domain: each
layer applies an FFT, linearly transforms the lowest Fourier modes, inverts the
FFT, and adds a pointwise bypass. Because the kernel acts on a grid-resolved
field, the formulation must declare its domains explicitly for the WRAM
telemetry setting:

- Input function u: the state-action reading u = [s_t, a_t] sampled at m = 14
  sensors, collocated on the uniform sensor lattice x_i = i/(m-1) in [0, 1].
  The per-sensor pair [u(x_i), x_i] is lifted to a width-dimensional channel
  field; this lattice doubles as the collocation grid of a latent field.
- Spectral blocks: L layers of (SpectralConv1d + pointwise bypass) with GELU,
  learning nonlocal convolution kernels over the sensor domain - the discrete
  analogue of the Green's-function kernel an FNO would learn for a PDE.
- Output function: the final latent field is projected to a scalar field f on
  [0, 1] and evaluated at the output-channel query coordinates y_q (one per
  state channel, evenly spaced) by linear interpolation - the FNO's continuous
  super-resolution evaluation mode, made possible precisely because the
  spectral representation is a smooth function of the domain, not of the grid.

Like DeepONet (10.41), no analytic kinematics are embedded: FNO is the
physically unconstrained spectral counterpart used to measure whether spectral
operator structure alone helps on a stiff 60 Hz fixed-point system.
"""

from typing import Optional

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """
    1D spectral convolution over the grid dimension: keep the lowest `modes`
    FFT coefficients of the latent field, apply a learned complex weight per
    mode and channel pair, and invert the transform.
    """

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.width = width
        self.modes = modes
        scale = 1.0 / (width * width)
        self.weight = nn.Parameter(scale * torch.randn(modes, width, width, dtype=torch.cfloat))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v: [B, N, W] latent field sampled at N grid points, W channels.

        Returns:
            [B, N, W] spectrally convolved field (real-valued).
        """
        n_grid = v.shape[1]
        vf = torch.fft.rfft(v, dim=1)  # [B, N//2 + 1, W] complex
        out_f = torch.zeros_like(vf)
        k = min(self.modes, vf.shape[1])
        # (batch, mode, in_channel) x (mode, in_channel, out_channel)
        out_f[:, :k] = torch.einsum("bmw,mwo->bmo", vf[:, :k], self.weight[:k])
        return torch.fft.irfft(out_f, n=n_grid, dim=1)


class FNODynamics(nn.Module):
    """
    FNO world model for the discrete SMW engine map:

        u = [s_t, a_t] --lift--> spectral blocks --project--> field f on [0,1]
        hat_s_{t+1}[q] = f(y_q) + b0[q]

    where y_q are the canonical output-channel coordinates. Grid-uniform
    sensor collocation and interpolation decoding keep the model a genuine
    neural operator: it consumes a sampled function and produces a function
    that may be queried anywhere in [0, 1], not a fixed-length vector map.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 6,
        width: int = 32,
        modes: int = 6,
        n_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.width = width
        self.modes = modes
        self.n_layers = n_layers

        self.num_sensors = state_dim + action_dim

        # Sensor lattice x_i in [0, 1] and canonical output-channel queries y_q.
        self.register_buffer(
            "sensor_grid",
            torch.linspace(0.0, 1.0, self.num_sensors).unsqueeze(-1),
        )
        self.register_buffer(
            "canonical_query_coords",
            torch.linspace(0.0, 1.0, state_dim),
        )

        # Pointwise lifting of [u(x_i), x_i] to the latent channel width.
        self.lift = nn.Linear(2, width)

        self.spectral_layers = nn.ModuleList(
            [SpectralConv1d(width, modes) for _ in range(n_layers)]
        )
        self.bypass_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        self.activation = nn.GELU()

        # Pointwise projection of the latent field to the scalar output field.
        self.readout = nn.Linear(width, 1)
        self.output_bias = nn.Parameter(torch.zeros(state_dim))

    def _evaluate_field(self, field: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sample the scalar grid field at query coordinates by linear interpolation.

        Args:
            field: [B, N] scalar field on the sensor lattice over [0, 1].
            y: [Q] query coordinates in [0, 1].

        Returns:
            [B, Q] interpolated evaluations f(y_q).
        """
        n_grid = field.shape[1]
        pos = y.clamp(0.0, 1.0) * (n_grid - 1)
        lo = pos.floor().long()  # [Q]
        hi = (lo + 1).clamp(max=n_grid - 1)
        w_hi = (pos - lo.float()).unsqueeze(0)  # [1, Q]
        f_lo = field[:, lo]  # [B, Q]
        f_hi = field[:, hi]  # [B, Q]
        return f_lo * (1.0 - w_hi) + f_hi * w_hi

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        query_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim] kinematic state s_t.
            action: [B, action_dim] controller reading a_t.
            query_coords: optional [Q] coordinates in [0, 1]; when omitted the
                canonical per-channel grid is used, yielding state_dim outputs.

        Returns:
            next_state_pred: [B, Q] operator evaluation at the query grid
            ([B, state_dim] on the canonical grid).
        """
        sensors = torch.cat([state, action], dim=-1)  # [B, m]
        batch = sensors.shape[0]
        grid = self.sensor_grid.unsqueeze(0).expand(batch, -1, -1)  # [B, m, 1]
        lifted = self.lift(torch.cat([sensors.unsqueeze(-1), grid], dim=-1))  # [B, m, W]

        v = lifted
        for spectral, bypass in zip(self.spectral_layers, self.bypass_layers):
            v = self.activation(spectral(v) + bypass(v))

        field = self.readout(v).squeeze(-1)  # [B, m] scalar field on [0, 1]

        coords: torch.Tensor
        if query_coords is None:
            canonical = True
            coords = self.canonical_query_coords
        else:
            canonical = False
            coords = query_coords.to(state.device).reshape(-1)

        out = self._evaluate_field(field, coords)
        # b0 is defined over the canonical output-channel grid; a custom query
        # set evaluates the raw continuous field without it.
        if canonical:
            out = out + self.output_bias.unsqueeze(0)
        return out
