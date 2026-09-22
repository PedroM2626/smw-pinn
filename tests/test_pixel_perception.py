"""Tests for pixel perception: frame conversion, estimator, dataset, guard."""

import numpy as np
import pytest
import torch

from src.environment.snes_emulator import (
    RETRO_PIXEL_FORMAT_0RGB1555,
    RETRO_PIXEL_FORMAT_RGB565,
    RETRO_PIXEL_FORMAT_RGB888,
    _convert_frame_bytes,
)
from src.perception.pixel_encoder import PixelStateEstimator, StateNormalizer, preprocess_frame
from src.perception.vision_dataset import FrameStateDataset, create_vision_loaders
from tests.conftest import requires_emulator


def _u16_buffer(pixels, width, height):
    arr = np.array(pixels, dtype=np.uint16).reshape(height, width)
    return np.frombuffer(arr.tobytes(), dtype=np.uint8)


def test_convert_rgb565_primaries():
    # 0xF800=red, 0x07E0=green, 0x001F=blue, 0xFFFF=white
    raw = _u16_buffer([0xF800, 0x07E0, 0x001F, 0xFFFF], 2, 2)
    rgb = _convert_frame_bytes(raw, 2, 2, 4, RETRO_PIXEL_FORMAT_RGB565)
    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8
    assert rgb[0, 0].tolist() == [255, 0, 0]
    assert rgb[0, 1].tolist() == [0, 255, 0]
    assert rgb[1, 0].tolist() == [0, 0, 255]
    assert rgb[1, 1].tolist() == [255, 255, 255]


def test_convert_0rgb1555_primaries():
    raw = _u16_buffer([0x7C00, 0x03E0, 0x001F, 0x7FFF], 2, 2)
    rgb = _convert_frame_bytes(raw, 2, 2, 4, RETRO_PIXEL_FORMAT_0RGB1555)
    assert rgb[0, 0].tolist() == [255, 0, 0]
    assert rgb[0, 1].tolist() == [0, 255, 0]
    assert rgb[1, 0].tolist() == [0, 0, 255]
    assert rgb[1, 1].tolist() == [255, 255, 255]


def test_convert_rgb888_channel_order():
    # Little-endian 0x00RRGGBB stored as [B, G, R, 0].
    px = np.array([[[10, 20, 30, 0]]], dtype=np.uint8)
    rgb = _convert_frame_bytes(px.reshape(-1), 1, 1, 4, RETRO_PIXEL_FORMAT_RGB888)
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_convert_respects_pitch_stride():
    # Pitch wider than width: trailing bytes per row are padding.
    payload = np.array([0xF800, 0x0000, 0x001F, 0x0000], dtype=np.uint16)
    raw = np.frombuffer(payload.tobytes(), dtype=np.uint8)
    rgb = _convert_frame_bytes(raw, 1, 2, 4, RETRO_PIXEL_FORMAT_RGB565)
    assert rgb.shape == (2, 1, 3)
    assert rgb[0, 0].tolist() == [255, 0, 0]
    assert rgb[1, 0].tolist() == [0, 0, 255]


@requires_emulator
def test_frame_capture_roundtrip():
    from src.environment.snes_emulator import SnesLibretroEmulator

    emu = SnesLibretroEmulator("src/environment/bin/snes9x_libretro.dll")
    try:
        assert emu.get_frame() is None  # capture off by default
        emu.load_rom("data/raw/smw_usa.sfc")
        emu.enable_frame_capture(True)
        for _ in range(30):
            emu.step_frame()
        frame = emu.get_frame()
        assert frame is not None
        assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
        assert frame.shape[0] in (224, 448) and frame.shape[1] in (256, 512)
        emu.enable_frame_capture(False)
        assert emu.get_frame() is None
    finally:
        emu.close()


def test_estimator_forward_backward():
    model = PixelStateEstimator(base_channels=4)
    frames = torch.rand(2, 3, 24, 16)
    pred = model(frames)
    assert pred.shape == (2, 8)
    loss = pred.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any()


def test_normalizer_roundtrip_and_dict():
    rng = np.random.default_rng(0)
    states = rng.normal(100, 50, size=(64, 8)).astype(np.float32)
    norm = StateNormalizer().fit(states)
    back = norm.denormalize(norm.normalize(states))
    assert np.allclose(back, states, atol=1e-6)
    clone = StateNormalizer.from_dict(norm.to_dict())
    assert np.allclose(clone.normalize(states), norm.normalize(states))


def test_vision_dataset_and_seeded_loaders():
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 256, size=(32, 12, 10, 3), dtype=np.uint8)
    states = rng.normal(0, 5, size=(32, 8)).astype(np.float32)
    ds = FrameStateDataset(frames, states)
    f, s = ds[0]
    assert f.shape == (3, 12, 10) and s.shape == (8,)
    assert f.min() >= 0.0 and f.max() <= 1.0
    l1, _, _ = create_vision_loaders(frames, states, batch_size=8, seed=3)
    l2, _, _ = create_vision_loaders(frames, states, batch_size=8, seed=3)
    assert torch.equal(next(iter(l1))[0], next(iter(l2))[0])
    with pytest.raises(ValueError):
        FrameStateDataset(frames, states[:, :4])


def test_estimator_memorizes_tiny_set():
    torch.manual_seed(0)
    model = PixelStateEstimator(base_channels=4)
    frames = torch.rand(8, 3, 16, 12)
    states = torch.randn(8, 8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        initial = loss_fn(model(frames), states).item()
    for _ in range(40):
        opt.zero_grad()
        loss = loss_fn(model(frames), states)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = loss_fn(model(frames), states).item()
    assert final < initial


def test_preprocess_frame():
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    frame[..., 0] = 255
    t = preprocess_frame(frame)
    assert t.shape == (1, 3, 4, 5)
    assert float(t[0, 0].max()) == 1.0 and float(t[0, 1].max()) == 0.0
