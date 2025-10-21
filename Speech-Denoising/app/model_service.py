# app/model_service.py
import io
import os
import zipfile
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch import einsum

# -----------------------
#  Helpers replacing utils.py
# -----------------------

def calc_same_padding(kernel_size: int):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)
    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_pos_emb=512):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        from einops import rearrange
        n, device, h, max_pos_emb, has_context = (
            x.shape[-2], x.device, self.heads, self.max_pos_emb, context is not None
        )
        context = x if context is None else context

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # Shaw relative position
        seq = torch.arange(n, device=device)
        dist = (seq[:, None] - seq[None, :]).clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if mask is not None or context_mask is not None:
            mask = mask if mask is not None else torch.ones(*x.shape[:2], device=device)
            context_mask = context_mask if context_mask is not None else (
                mask if not has_context else torch.ones(*context.shape[:2], device=device)
            )
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(context_mask, "b j -> b () () j")
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()
        from einops.layers.torch import Rearrange
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class ConformerBlock(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, ff_mult=4,
                 conv_expansion_factor=2, conv_kernel_size=31,
                 attn_dropout=0.0, ff_dropout=0.0, conv_dropout=0.0):
        super().__init__()
        self.ff1 = Scale(0.5, PreNorm(dim, FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)))
        self.attn = PreNorm(dim, Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout))
        self.conv = ConformerConvModule(dim=dim, causal=False,
                                        expansion_factor=conv_expansion_factor,
                                        kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.ff2 = Scale(0.5, PreNorm(dim, FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)))
        self.post_norm = nn.LayerNorm(dim)
    def forward(self, x, mask=None):
        x = self.ff1(x) + x
        x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x

class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, f"pad{i+1}", nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0))
            setattr(self, f"conv{i+1}", nn.Conv2d(self.in_channels * (i + 1), self.in_channels,
                                                  kernel_size=self.kernel_size, dilation=(dil, 1)))
            setattr(self, f"norm{i+1}", nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, f"prelu{i+1}", nn.PReLU(self.in_channels))
    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, f"pad{i+1}")(skip)
            out = getattr(self, f"conv{i+1}")(out)
            out = getattr(self, f"norm{i+1}")(out)
            out = getattr(self, f"prelu{i+1}")(out)
            skip = torch.cat([out, skip], dim=1)
        return out

class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x

class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super().__init__()
        self.time_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel // 4,
                                             heads=4, conv_kernel_size=31,
                                             attn_dropout=0.2, ff_dropout=0.2)
        self.freq_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel // 4,
                                             heads=4, conv_kernel_size=31,
                                             attn_dropout=0.2, ff_dropout=0.2)
    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r
    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super().__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)
    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)

class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super().__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))
    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x

class TSCNet(nn.Module):
    def __init__(self, num_channel=64, num_features=201):
        super().__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)
        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        self.TSCB_3 = TSCB(num_channel=num_channel)
        self.TSCB_4 = TSCB(num_channel=num_channel)
        self.mask_decoder = MaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)
    def forward(self, x):
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)
        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)
        mask = self.mask_decoder(out_5)
        out_mag = mask * mag
        complex_out = self.complex_decoder(out_5)
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)
        return final_real, final_imag

def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** 0.3
    real_c = mag * torch.cos(phase)
    imag_c = mag * torch.sin(phase)
    return torch.stack([real_c, imag_c], 1)  # (B, 2, F, T)

def power_uncompress2(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_u = mag * torch.cos(phase)
    imag_u = mag * torch.sin(phase)
    # Return complex spectrogram (B, F, T)
    return torch.complex(real_u, imag_u).squeeze(1)

# -----------------------
#  EnhancerService
# -----------------------

class EnhancerService:
    """
    Loads TSCNet once and exposes noisy-only enhancement utilities.
    Works on CPU or CUDA, resamples to 16 kHz if needed.
    """

    def __init__(
        self,
        model_path: str,
        n_fft: int = 400,
        hop: int = 100,
        sample_rate: int = 16000,
        num_channel: int = 64,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop = hop

        self.model = TSCNet(num_channel=num_channel, num_features=n_fft // 2 + 1).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        # If your checkpoint is nested (e.g., {"state_dict": ...}), adapt here:
        # state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

        self.window = torch.hamming_window(
            self.n_fft, periodic=True, dtype=torch.float32
        ).to(self.device)

        self._frame_len = 100  # matches your original logic

    def _load_audio_16k(self, path: str) -> torch.Tensor:
        """Load audio, mono-ize, resample to 16k if needed, return (1, T) tensor on device."""
        wav, sr = torchaudio.load(path)  # (C, T)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav.to(self.device)

    @torch.no_grad()
    def enhance_file(
        self,
        noisy_wav_path: str,
        cut_len: int = 16000 * 16,
        save_enhanced_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Enhance a single noisy file. Returns enhanced float32 numpy array.
        """
        wav = self._load_audio_16k(noisy_wav_path)  # (1, T)

        # normalization (same as your code)
        c = torch.sqrt(wav.size(-1) / torch.sum((wav ** 2.0), dim=-1))
        wav = torch.transpose(wav, 0, 1)
        wav = torch.transpose(wav * c, 0, 1)

        length = wav.size(-1)
        frame_num = int(np.ceil(length / self._frame_len))
        padded_len = frame_num * self._frame_len
        padding_len = padded_len - length
        if padding_len > 0:
            wav = torch.cat([wav, wav[:, :padding_len]], dim=-1)

        if padded_len > cut_len:
            batch_size = 1  # keep it simple / consistent with your working code
            wav = torch.reshape(wav, (batch_size, -1))

        # STFT → compress → model
        wav_spec = torch.stft(
            wav, self.n_fft, self.hop, window=self.window, onesided=True, return_complex=False
        )
        wav_spec = power_compress(wav_spec).permute(0, 1, 3, 2)  # (B, 2, T, F)

        est_real, est_imag = self.model(wav_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

        # Back to complex for PyTorch 2.x istft
        complex_spec = power_uncompress2(est_real, est_imag)  # (B, F, T) complex

        est_audio = torch.istft(
            complex_spec, self.n_fft, self.hop, window=self.window, onesided=True
        )  # (B, T)

        est_audio = est_audio / c
        est_audio = torch.flatten(est_audio)[:length].detach().cpu().numpy().astype("float32")

        if save_enhanced_path:
            sf.write(save_enhanced_path, est_audio, self.sample_rate)

        return est_audio

    @torch.no_grad()
    def enhance_zip(self, noisy_zip_bytes: bytes, cut_len: int = 16000 * 16) -> bytes:
        """
        Accept a ZIP containing WAV files. Returns a ZIP of enhanced WAVs with same names.
        """
        with tempfile.TemporaryDirectory() as dtmp:
            dtmp = Path(dtmp)
            in_dir = dtmp / "noisy"
            out_dir = dtmp / "enhanced"
            in_dir.mkdir()
            out_dir.mkdir()

            with zipfile.ZipFile(io.BytesIO(noisy_zip_bytes)) as zf:
                zf.extractall(in_dir)

            for p in in_dir.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(in_dir)
                    out_path = out_dir / rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    enhanced = self.enhance_file(str(p), cut_len=cut_len)
                    sf.write(str(out_path), enhanced, self.sample_rate)

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in out_dir.rglob("*"):
                    if p.is_file():
                        zf.write(p, arcname=str(p.relative_to(out_dir)))
            return buf.getvalue()

