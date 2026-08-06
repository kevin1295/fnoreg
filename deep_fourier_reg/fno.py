import torch
from torch import nn
from neuralop.models import FNO
from neuralop.models.spectral_convolution import FactorizedSpectralConv2d, FactorizedSpectralConv3d
from fourier_blocks import FFCResNetBlock, STResNetBlock3d

class MyFNO(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model = FNO(**model_cfg)
    
    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        return self.model(x_in)

class SEblock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEblock3d(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class FreqGatedSpectralConv2d(FactorizedSpectralConv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        h0, h1 = self.half_total_n_modes
        self.freq_gate = nn.Parameter(torch.zeros(h0, h1))
        self.last_gate = None
        self.last_fft_before = None
        self.last_fft_after = None

    def forward(self, x, indices=0):
        batchsize, _, height, width = x.shape
        x = torch.fft.rfft2(x.float(), norm=self.fft_norm)
        out_fft = torch.zeros(
            [batchsize, self.out_channels, height, width // 2 + 1],
            dtype=x.dtype,
            device=x.device,
        )

        h0, h1 = self.half_n_modes
        gate = torch.sigmoid(self.freq_gate[:h0, :h1])
        upper = self._contract(
            x[:, :, :h0, :h1],
            self._get_weight(2 * indices),
            separable=self.separable,
        )
        lower = self._contract(
            x[:, :, -h0:, :h1],
            self._get_weight(2 * indices + 1),
            separable=self.separable,
        )
        gated_upper = upper * gate
        gated_lower = lower * gate
        out_fft[:, :, :h0, :h1] = gated_upper
        out_fft[:, :, -h0:, :h1] = gated_lower

        self.last_gate = gate.detach().cpu()
        self.last_fft_before = (
            upper.abs().mean(dim=(0, 1)) + lower.abs().mean(dim=(0, 1))
        ).mul(0.5).detach().cpu()
        self.last_fft_after = (
            gated_upper.abs().mean(dim=(0, 1))
            + gated_lower.abs().mean(dim=(0, 1))
        ).mul(0.5).detach().cpu()

        x = torch.fft.irfft2(
            out_fft, s=(height, width), dim=(-2, -1), norm=self.fft_norm
        )
        if self.bias is not None:
            x = x + self.bias[indices, ...]
        return x


class AdaptiveFreqGatedSpectralConv2d(FactorizedSpectralConv2d):
    """SpectralConv2d with adaptive per-mode frequency gating.

    Unlike a static gate, the weights are computed dynamically from the FFT output:
        energy = |_contract(x_fft)|.mean(dim=channel)
        gate = sigmoid(scale * energy + bias)

    This means the gate adapts to each input -- different images get different
    frequency weightings.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        h0, h1 = self.half_total_n_modes
        self.gate_scale = nn.Parameter(torch.ones(h0, h1))
        self.gate_bias = nn.Parameter(torch.zeros(h0, h1))
        self.gate_override = None
        self.last_dynamic_gate = None
        self.last_gate = None
        self.last_fft_before = None
        self.last_fft_after = None

    def set_gate_override(self, gate):
        self.gate_override = gate

    def forward(self, x, indices=0):
        batchsize, channels, height, width = x.shape

        x = torch.fft.rfft2(x.float(), norm=self.fft_norm)

        out_fft = torch.zeros(
            [batchsize, self.out_channels, height, width // 2 + 1],
            dtype=x.dtype, device=x.device,
        )

        h0, h1 = self.half_n_modes[0], self.half_n_modes[1]

        # Upper block
        fft_upper = self._contract(
            x[:, :, :h0, :h1],
            self._get_weight(2 * indices),
            separable=self.separable,
        )  # [B, out_C, h0, h1] complex

        # Adaptive gate from FFT magnitude (pool over channels)
        energy = torch.abs(fft_upper).mean(dim=1)  # [B, h0, h1]
        dynamic_gate = torch.sigmoid(
            self.gate_scale[:h0, :h1] * energy + self.gate_bias[:h0, :h1]
        )  # [B, h0, h1]

        if self.gate_override is None:
            gate = dynamic_gate
        else:
            gate = torch.as_tensor(
                self.gate_override,
                device=dynamic_gate.device,
                dtype=dynamic_gate.dtype,
            )
            if gate.ndim == 2:
                gate = gate.unsqueeze(0)
            if gate.shape[-2:] != (h0, h1):
                raise ValueError(
                    f'Gate override has shape {tuple(gate.shape)}, expected (*, {h0}, {h1})'
                )
            if gate.shape[0] not in (1, batchsize):
                raise ValueError(
                    f'Gate override batch size is {gate.shape[0]}, expected 1 or {batchsize}'
                )
            gate = gate.expand(batchsize, -1, -1)

        self.last_dynamic_gate = dynamic_gate.detach().mean(dim=0).cpu()
        self.last_gate = gate.detach().mean(dim=0).cpu()
        self.last_fft_before = torch.abs(fft_upper).detach().mean(dim=(0, 1)).cpu()  # avg over B, C
        fft_upper_gated = fft_upper * gate[:, None, :, :]
        self.last_fft_after = torch.abs(fft_upper_gated).detach().mean(dim=(0, 1)).cpu()

        # Apply gate
        out_fft[:, :, :h0, :h1] = fft_upper_gated

        # Lower block (same gate, conjugate symmetric)
        out_fft[:, :, -h0:, :h1] = (
            self._contract(
                x[:, :, -h0:, :h1],
                self._get_weight(2 * indices + 1),
                separable=self.separable,
            )
            * gate[:, None, :, :]
        )

        if self.output_scaling_factor is not None:
            width_out = int(round(width * self.output_scaling_factor[0]))
            height_out = int(round(height * self.output_scaling_factor[1]))

        x = torch.fft.irfft2(
            out_fft, s=(height, width), dim=(-2, -1), norm=self.fft_norm
        )

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


class FourierLayer2d(nn.Module):
    def __init__(self, in_ch, out_ch, n_modes, factorization=None, rank=0.5, nonlinearity=nn.GELU):
        super().__init__()
        
        self.spectral_conv = FactorizedSpectralConv2d(in_ch, 
                                            out_ch, 
                                            n_modes, 
                                            factorization=factorization,
                                            rank=rank)
        
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.act = nonlinearity()
    
    def forward(self, x):
        x_fc = self.spectral_conv(x)
        x_skip = self.skip(x).to(x.device)
        x_act = self.act(x_skip + x_fc)
        return x_act + x
    
class GatedFourierLayer2d(FourierLayer2d):
    def __init__(self, in_ch, out_ch, n_modes, factorization=None, rank=0.5,
                 nonlinearity=nn.GELU, gate_reduction=4):
        super().__init__(in_ch, out_ch, n_modes,
                         factorization=factorization, rank=rank,
                         nonlinearity=nonlinearity)
        self.gate = SEblock(out_ch, reduction=gate_reduction)

    def forward(self, x):
        x_fc = self.spectral_conv(x)
        x_fc = self.gate(x_fc)
        x_skip = self.skip(x).to(x.device)
        x_act = self.act(x_skip + x_fc)
        return x_act + x


class FreqGatedFourierLayer2d(FourierLayer2d):
    def __init__(self, in_ch, out_ch, n_modes, factorization=None, rank=0.5,
                 nonlinearity=nn.GELU):
        super().__init__(in_ch, out_ch, n_modes,
                         factorization=factorization, rank=rank,
                         nonlinearity=nonlinearity)
        self.spectral_conv = FreqGatedSpectralConv2d(
            in_ch, out_ch, n_modes,
            factorization=factorization, rank=rank,
        )


class AdaptiveFreqGatedFourierLayer2d(FourierLayer2d):
    """FourierLayer2d variant using AdaptiveFreqGatedSpectralConv2d for adaptive frequency gating."""

    def __init__(self, in_ch, out_ch, n_modes, factorization=None, rank=0.5,
                 nonlinearity=nn.GELU):
        super().__init__(in_ch, out_ch, n_modes,
                         factorization=factorization, rank=rank,
                         nonlinearity=nonlinearity)
        self.spectral_conv = AdaptiveFreqGatedSpectralConv2d(
            in_ch, out_ch, n_modes,
            factorization=factorization, rank=rank,
        )


class FNOReg(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.lifting = nn.Conv2d(model_cfg['in_channels'], model_cfg['hidden_channels'], kernel_size=1)

        alpha = 1

        self.encoder = FFCResNetBlock(in_channels=model_cfg['hidden_channels'],
                                    out_channels=model_cfg['hidden_channels'],
                                    kernel_size=3,
                                    fu_kernel=1,
                                    alpha_in=alpha,
                                    alpha_out=alpha                                      
                                    )
        
        fact = model_cfg['factorization']
        fact = None if not fact else fact
        self.fno_blocks = nn.Sequential(
            *[
                FourierLayer2d(
                    in_ch=model_cfg['hidden_channels'],
                    out_ch=model_cfg['hidden_channels'],
                    n_modes=model_cfg['n_modes'],
                    factorization=fact,
                    rank=model_cfg['rank']
                )
                for _ in range(model_cfg['n_layers'])
            ]
        )

        self.decoder = FFCResNetBlock(in_channels=model_cfg['hidden_channels'],
                                    out_channels=model_cfg['hidden_channels'],
                                    kernel_size=3,
                                    fu_kernel=1,
                                    alpha_in=alpha,
                                    alpha_out=alpha                                      
                                    )
        
        self.projection = nn.Sequential(
            nn.Conv2d(model_cfg['hidden_channels'],
                      model_cfg['projection_channels'],
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(model_cfg['projection_channels'],
                      model_cfg['out_channels'],
                      kernel_size=1)
        )
    
    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = self.lifting(x)
        x = self.encoder(x)
        x = self.fno_blocks(x)
        x = self.decoder(x)
        return self.projection(x)


class FreqGatedFNOReg(FNOReg):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        fact = model_cfg['factorization']
        fact = None if not fact else fact
        self.fno_blocks = nn.Sequential(
            *[
                FreqGatedFourierLayer2d(
                    in_ch=model_cfg['hidden_channels'],
                    out_ch=model_cfg['hidden_channels'],
                    n_modes=model_cfg['n_modes'],
                    factorization=fact,
                    rank=model_cfg['rank'],
                )
                for _ in range(model_cfg['n_layers'])
            ]
        )


class GatedFNOReg(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.lifting = nn.Conv2d(model_cfg['in_channels'], model_cfg['hidden_channels'], kernel_size=1)

        alpha = 1

        self.encoder = FFCResNetBlock(in_channels=model_cfg['hidden_channels'],
                                    out_channels=model_cfg['hidden_channels'],
                                    kernel_size=3,
                                    fu_kernel=1,
                                    alpha_in=alpha,
                                    alpha_out=alpha
                                    )

        fact = model_cfg['factorization']
        fact = None if not fact else fact
        gate_reduction = model_cfg.get('gate_reduction', 4)
        self.fno_blocks = nn.Sequential(
            *[
                GatedFourierLayer2d(
                    in_ch=model_cfg['hidden_channels'],
                    out_ch=model_cfg['hidden_channels'],
                    n_modes=model_cfg['n_modes'],
                    factorization=fact,
                    rank=model_cfg['rank'],
                    gate_reduction=gate_reduction,
                )
                for _ in range(model_cfg['n_layers'])
            ]
        )

        self.decoder = FFCResNetBlock(in_channels=model_cfg['hidden_channels'],
                                    out_channels=model_cfg['hidden_channels'],
                                    kernel_size=3,
                                    fu_kernel=1,
                                    alpha_in=alpha,
                                    alpha_out=alpha
                                    )

        self.projection = nn.Sequential(
            nn.Conv2d(model_cfg['hidden_channels'],
                      model_cfg['projection_channels'],
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(model_cfg['projection_channels'],
                      model_cfg['out_channels'],
                      kernel_size=1)
        )

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = self.lifting(x)
        x = self.encoder(x)
        x = self.fno_blocks(x)
        x = self.decoder(x)
        return self.projection(x)


class AdaptiveFreqGatedFNOReg(nn.Module):
    """FNOReg variant with adaptive frequency-mode gating.

    Unlike the static gate in the deprecated FreqGatedFNOReg, the gate here
    is computed from the FFT output of each layer:
        gate = sigmoid(scale * |FFT_output|.mean(ch) + bias)

    This means different inputs produce different gate patterns.
    """

    def __init__(self, model_cfg):
        super().__init__()

        self.lifting = nn.Conv2d(model_cfg['in_channels'], model_cfg['hidden_channels'], kernel_size=1)

        alpha = 1

        self.encoder = FFCResNetBlock(in_channels=model_cfg['hidden_channels'],
                                      out_channels=model_cfg['hidden_channels'],
                                      kernel_size=3,
                                      fu_kernel=1,
                                      alpha_in=alpha,
                                      alpha_out=alpha)

        fact = model_cfg['factorization']
        fact = None if not fact else fact
        self.fno_blocks = nn.Sequential(
            *[
                AdaptiveFreqGatedFourierLayer2d(
                    in_ch=model_cfg['hidden_channels'],
                    out_ch=model_cfg['hidden_channels'],
                    n_modes=model_cfg['n_modes'],
                    factorization=fact,
                    rank=model_cfg['rank'],
                )
                for _ in range(model_cfg['n_layers'])
            ]
        )

        self.decoder = FFCResNetBlock(in_channels=model_cfg['hidden_channels'],
                                      out_channels=model_cfg['hidden_channels'],
                                      kernel_size=3,
                                      fu_kernel=1,
                                      alpha_in=alpha,
                                      alpha_out=alpha)

        self.projection = nn.Sequential(
            nn.Conv2d(model_cfg['hidden_channels'],
                      model_cfg['projection_channels'],
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(model_cfg['projection_channels'],
                      model_cfg['out_channels'],
                      kernel_size=1)
        )

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = self.lifting(x)
        x = self.encoder(x)
        x = self.fno_blocks(x)
        x = self.decoder(x)
        return self.projection(x)

    def get_gate_layers(self):
        return [layer.spectral_conv for layer in self.fno_blocks]

    def set_gate_overrides(self, gates=None):
        gate_layers = self.get_gate_layers()
        if gates is None:
            gates = [None] * len(gate_layers)
        if len(gates) != len(gate_layers):
            raise ValueError(
                f'Received {len(gates)} gate overrides for {len(gate_layers)} layers'
            )
        for layer, gate in zip(gate_layers, gates):
            layer.set_gate_override(gate)

    def get_last_gates(self):
        gates = [layer.last_gate for layer in self.get_gate_layers()]
        if any(gate is None for gate in gates):
            raise RuntimeError('Gate values are unavailable before a forward pass')
        return [gate.clone() for gate in gates]

    @torch.no_grad()
    def collect_gates(self, dataloader, device, num_samples):
        """Collect actual gate values produced for a set of validation samples.

        Runs num_samples through the model and collects self.last_gate from
        each Fourier layer. Returns per-layer lists of gate tensors.

        Returns:
            list[list[torch.Tensor]]: gates[layer_idx][sample_idx], each (h0, h1).
        """
        self.eval()
        n_layers = len(self.fno_blocks)
        all_gates = [[] for _ in range(n_layers)]

        samples_collected = 0
        for moving, fixed, _, _ in dataloader:
            if samples_collected >= num_samples:
                break
            batch_size = moving.size(0)
            for b in range(batch_size):
                if samples_collected >= num_samples:
                    break
                # Forward single sample
                moving_b = moving[b:b + 1].to(device)
                fixed_b = fixed[b:b + 1].to(device)
                self(moving_b, fixed_b)
                # Collect last_gate from each layer
                for layer_idx, layer in enumerate(self.fno_blocks):
                    all_gates[layer_idx].append(layer.spectral_conv.last_gate.clone())
                samples_collected += 1

        return all_gates


class FourierLayer3d(nn.Module):
    def __init__(self, in_ch, out_ch, n_modes, factorization=None, rank=0.5, nonlinearity=nn.GELU):
        super().__init__()
        
        self.spectral_conv = FactorizedSpectralConv3d(in_ch, 
                                            out_ch, 
                                            n_modes, 
                                            factorization=factorization,
                                            rank=rank)
        
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.act = nonlinearity()
    
    def forward(self, x):
        x_fc = self.spectral_conv(x)
        x_skip = self.skip(x).to(x.device)
        x_act = self.act(x_skip + x_fc)
        return x_act + x
    
class GatedFourierLayer3d(FourierLayer3d):
    def __init__(self, in_ch, out_ch, n_modes, factorization=None, rank=0.5,
                 nonlinearity=nn.GELU, gate_reduction=4):
        super().__init__(in_ch, out_ch, n_modes,
                         factorization=factorization, rank=rank,
                         nonlinearity=nonlinearity)
        self.gate = SEblock3d(out_ch, reduction=gate_reduction)

    def forward(self, x):
        x_fc = self.spectral_conv(x)
        x_fc = self.gate(x_fc)
        x_skip = self.skip(x).to(x.device)
        x_act = self.act(x_skip + x_fc)
        return x_act + x

class FNOReg3d(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.lifting = nn.Conv3d(model_cfg['in_channels'], model_cfg['hidden_channels'], kernel_size=1)
    
        self.encoder = STResNetBlock3d(in_channels=model_cfg['hidden_channels'],
                                    out_channels=model_cfg['hidden_channels'],
                                    fu_kernel=1                                     
                                    )
        
        fact = model_cfg['factorization']
        fact = None if not fact else fact
        self.fno_blocks = nn.Sequential(
            *[
                FourierLayer3d(
                    in_ch=model_cfg['hidden_channels'],
                    out_ch=model_cfg['hidden_channels'],
                    n_modes=model_cfg['n_modes'],
                    factorization=fact,
                    rank=model_cfg['rank']
                )
                for _ in range(model_cfg['n_layers'])
            ]
        )
        
        self.decoder = STResNetBlock3d(in_channels=model_cfg['hidden_channels'],
                                    out_channels=model_cfg['hidden_channels'],
                                    fu_kernel=1                                   
                                    )
        
        self.projection = nn.Sequential(
            nn.Conv3d(model_cfg['hidden_channels'],
                      model_cfg['projection_channels'],
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(model_cfg['projection_channels'],
                      model_cfg['out_channels'],
                      kernel_size=1)
        )

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = self.lifting(x)
        x = self.encoder(x)
        x = self.fno_blocks(x)
        x = self.decoder(x)
        return self.projection(x)

class GatedFNOReg3d(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.lifting = nn.Conv3d(model_cfg['in_channels'], model_cfg['hidden_channels'], kernel_size=1)

        self.encoder = STResNetBlock3d(in_channels=model_cfg['hidden_channels'],
                                    out_channels=model_cfg['hidden_channels'],
                                    fu_kernel=1
                                    )

        fact = model_cfg['factorization']
        fact = None if not fact else fact
        gate_reduction = model_cfg.get('gate_reduction', 4)
        self.fno_blocks = nn.Sequential(
            *[
                GatedFourierLayer3d(
                    in_ch=model_cfg['hidden_channels'],
                    out_ch=model_cfg['hidden_channels'],
                    n_modes=model_cfg['n_modes'],
                    factorization=fact,
                    rank=model_cfg['rank'],
                    gate_reduction=gate_reduction,
                )
                for _ in range(model_cfg['n_layers'])
            ]
        )

        self.decoder = STResNetBlock3d(in_channels=model_cfg['hidden_channels'],
                                    out_channels=model_cfg['hidden_channels'],
                                    fu_kernel=1
                                    )

        self.projection = nn.Sequential(
            nn.Conv3d(model_cfg['hidden_channels'],
                      model_cfg['projection_channels'],
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(model_cfg['projection_channels'],
                      model_cfg['out_channels'],
                      kernel_size=1)
        )

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = self.lifting(x)
        x = self.encoder(x)
        x = self.fno_blocks(x)
        x = self.decoder(x)
        return self.projection(x)
