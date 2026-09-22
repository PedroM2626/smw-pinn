"""Hardware-loop logic tested without the emulator (CI-safe).

A scriptable FakeEmu stands in for SnesLibretroEmulator (whose Windows DLL
cannot load on Linux CI); emulator-constructing entry points are reached via
monkeypatched module attributes. What still needs real hardware (full MPC
rollouts, boot flows) stays documented, not faked.
"""

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from src.evaluation import diagnose_obstacle_1000 as diag_mod
from src.evaluation import evaluate_hierarchical_mpc as hier_mod
from src.evaluation import evaluate_pixel_mpc as pixel_mod
from src.evaluation import evaluate_tilemap_mpc as tile_mod
from src.evaluation import mpc_reflex_ablation as reflex_mod
from src.evaluation import plot_learning_curves as curves_mod
from src.evaluation import spatial_holdout_benchmark as holdout_mod
from src.models import HardResidualPINNDynamics, StatisticalMLPDynamics
from src.models.tilemap_pinn import TilemapPINNDynamics
from src.perception.pixel_encoder import PixelStateEstimator, StateNormalizer
from src.training import train_pixel_estimator as train_pixel_mod
from src.training import train_set_multi_entity as train_set_mod
from src.training import train_terminal_value as train_value_mod
from src.training import train_unified_multimodal as train_unified_mod

BASE_12D = {
    "x": 100.0, "y": 300.0, "vx": 10.0, "vy": 0.0,
    "c_ground": 1.0, "c_ceiling": 0.0, "c_left": 0.0, "c_right": 0.0,
    "delta_x_enemy": 999.0, "delta_y_enemy": 0.0, "vx_enemy": 0.0,
    "hazard_active": 0.0, "air_state": 0.0,
}


class FakeEmu:
    """Scriptable stand-in for SnesLibretroEmulator."""

    def __init__(self, core_path="fake.dll", *, state_fn=None, sprites_fn=None,
                 patch=None, frame=None, mem_fn=None):
        self._t = 0
        self._state_fn = state_fn or (lambda t: dict(BASE_12D))
        self._sprites_fn = sprites_fn or (lambda t: [])
        self._patch = patch
        self._frame = frame
        self._mem_fn = mem_fn or (lambda addr: 0)
        self.wram_buffer = {}
        self.inputs = []

    def load_rom(self, path):
        pass

    def load_state(self, data):
        self._t = 0

    def step_frame(self):
        self._t += 1

    def set_input(self, d):
        self.inputs.append(dict(d))

    def get_smw_state(self):
        return dict(self._state_fn(self._t))

    def get_smw_extended_state(self):
        return dict(self._state_fn(self._t))

    def get_active_sprites(self):
        return [dict(s) for s in self._sprites_fn(self._t)]

    def get_local_tilemap_patch(self, x, y, radius=3):
        if callable(self._patch):
            return self._patch(self._t)
        return np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.int64)

    def read_wram_u8(self, addr):
        return self._mem_fn(addr)

    def enable_frame_capture(self, enabled=True):
        pass

    def get_frame(self):
        return None if self._frame is None else self._frame.copy()

    def close(self):
        pass


class FakeController:
    """Deterministic planner stub returning a fixed action vector."""

    def __init__(self, action=None):
        self.action = (
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            if action is None else np.array(action, dtype=np.float32)
        )
        self.calls = 0

    def plan(self, state):
        self.calls += 1
        return self.action.copy(), {"best_reward": 1.0}


def _write(path, data: bytes = b"fake-state"):
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


# ---------- mpc_reflex_ablation.run_condition ----------

def test_run_condition_pit_termination():
    def state(t):
        s = dict(BASE_12D)
        s["x"] = 100.0 + t * 2.0
        s["y"] = 300.0 if t < 5 else 501.0
        return s

    emu = FakeEmu(state_fn=state)
    res = reflex_mod.run_condition(FakeController(), emu, b"savestate", False, 600)
    assert res["termination"] == "pit_fall"
    assert res["survived_frames"] < 600
    assert res["reflex_interventions"] == {"hazard_vault": 0, "wall_vault": 0, "b_pulse": 0}


def test_run_condition_stuck_termination():
    emu = FakeEmu()  # x never moves
    res = reflex_mod.run_condition(FakeController(), emu, b"savestate", False, 160)
    assert res["termination"] == "stuck_150f"
    assert res["stuck_frames"] >= 150


def test_run_condition_reflex_counts_and_timeout():
    def state(t):
        s = dict(BASE_12D)
        s["x"] = 100.0 + t * 3.0
        s["c_right"] = 1.0
        s["hazard_active"] = 1.0
        s["delta_x_enemy"] = 40.0
        return s

    emu = FakeEmu(state_fn=state)
    ctrl = FakeController(action=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # B held for pulse
    res = reflex_mod.run_condition(ctrl, emu, b"savestate", True, 10)
    assert res["termination"] == "timeout"
    assert res["reflex_interventions"]["hazard_vault"] == 10
    assert res["reflex_interventions"]["wall_vault"] == 10
    assert res["reflex_interventions"]["b_pulse"] > 0
    assert len(res["progress_log"]) == 1  # [::10] of 10 frames


def test_run_ablation_end_to_end_with_stubs(tmp_path):
    state_file = _write(tmp_path / "s.state")
    out = tmp_path / "results"
    res = reflex_mod.run_ablation(
        max_frames=6, output_dir=str(out), controller=FakeController(),
        emulator_cls=FakeEmu, state_path=state_file,
        core_path="fake.dll", rom_path="fake.sfc",
    )
    assert set(res) == {"pure", "reflex"}
    assert res["pure"]["termination"] == "timeout"
    assert (out / "mpc_reflex_ablation.json").exists()
    assert (out / "figures" / "mpc_reflex_ablation.png").exists()


# ---------- diagnose_obstacle_1000 ----------

def test_run_diagnosis_with_stub_emu(tmp_path, monkeypatch):
    def state(t):
        s = dict(BASE_12D)
        s["x"] = 800.0 + t * 2.5
        s["y"] = 300.0
        s["vx"] = 20.0
        return s

    def sprites(t):
        s = state(t)
        if 850.0 < s["x"] < 1000.0:
            return [{"slot": 9, "id": 4, "x": s["x"] + 40.0, "y": s["y"], "vx": -1.0, "vy": 0.0}]
        return []

    def patch(t):
        p = np.zeros((7, 7), dtype=np.int64)
        if (t // 25) % 2 == 0:
            p[4:, 3] = 1
        return p

    class DiagEmu(FakeEmu):
        def __init__(self, core_path):
            super().__init__(core_path, state_fn=state, sprites_fn=sprites, patch=patch)

    monkeypatch.setattr(diag_mod, "SnesLibretroEmulator", DiagEmu)
    state_file = _write(tmp_path / "s.state")
    out = tmp_path / "results"
    payload = diag_mod.run_diagnosis(
        core_path="fake.dll", rom_path="fake.sfc", state_path=state_file,
        output_dir=str(out),
    )
    assert payload["num_samples"] > 0
    assert len(payload["ground_gaps"]) >= 1
    assert payload["num_hazard_encounters"] > 0
    assert (out / "obstacle_1000_diagnosis.json").exists()
    assert (out / "figures" / "obstacle_1000_diagnosis.png").exists()


# ---------- evaluate_tilemap_mpc ----------

def test_run_tilemap_mpc_with_stubs(tmp_path, monkeypatch):
    ckpt = tmp_path / "tilemap.pt"
    torch.save(TilemapPINNDynamics().state_dict(), ckpt)

    def state(t):
        s = dict(BASE_12D)
        s["x"] = 100.0 + t * 4.0
        return s

    class TileEmu(FakeEmu):
        def __init__(self, core_path):
            super().__init__(core_path, state_fn=state)

    monkeypatch.setattr(tile_mod, "SnesLibretroEmulator", TileEmu)
    state_file = _write(tmp_path / "s.state")
    out = tmp_path / "results"
    metrics = tile_mod.run_tilemap_mpc(
        tilemap_ckpt=str(ckpt), core_path="fake.dll", rom_path="fake.sfc",
        state_path=state_file, max_frames=2, output_dir=str(out),
    )
    assert metrics["survived_frames"] == 2
    assert metrics["termination"] == "timeout"
    assert (out / "tilemap_mpc_metrics.json").exists()


# ---------- evaluate_hierarchical_mpc ----------

def _tile_mem(addr: int) -> int:
    if addr < 0xC800:
        return 0
    row = ((addr - 0xC800) % 0x01B0) // 16
    return 0x3F if row >= 26 else 0x25


def test_run_hierarchical_with_stubs(tmp_path, monkeypatch):
    ckpt = tmp_path / "pinn.pt"
    torch.save(HardResidualPINNDynamics().state_dict(), ckpt)

    def state(t):
        s = dict(BASE_12D)
        s["x"] = 100.0 + t * 5.0
        return s

    class HierEmu(FakeEmu):
        def __init__(self, core_path):
            super().__init__(core_path, state_fn=state, mem_fn=_tile_mem)

    monkeypatch.setattr(hier_mod, "SnesLibretroEmulator", HierEmu)
    state_file = _write(tmp_path / "s.state")
    out = tmp_path / "results"
    metrics = hier_mod.run_hierarchical(
        pinn_ckpt=str(ckpt), core_path="fake.dll", rom_path="fake.sfc",
        state_path=state_file, max_frames=2, output_dir=str(out),
    )
    assert metrics["survived_frames"] == 2
    assert metrics["num_waypoints"] > 0
    assert (out / "hierarchical_mpc_metrics.json").exists()


# ---------- evaluate_pixel_mpc ----------

def test_run_pixel_mpc_with_stubs(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    est = PixelStateEstimator()
    norm = StateNormalizer().fit(rng.normal(100, 20, size=(16, 8)).astype(np.float32))
    est_ckpt = tmp_path / "est.pt"
    torch.save({"model": est.state_dict(), "normalizer": norm.to_dict()}, est_ckpt)
    pinn_ckpt = tmp_path / "pinn.pt"
    torch.save(HardResidualPINNDynamics().state_dict(), pinn_ckpt)

    def state(t):
        s = dict(BASE_12D)
        s["x"] = 100.0 + t * 4.0
        return s

    class PixelEmu(FakeEmu):
        def __init__(self, core_path):
            super().__init__(
                core_path, state_fn=state,
                frame=np.zeros((24, 32, 3), dtype=np.uint8),
            )

    monkeypatch.setattr(pixel_mod, "SnesLibretroEmulator", PixelEmu)
    state_file = _write(tmp_path / "s.state")
    out = tmp_path / "results"
    metrics = pixel_mod.run_pixel_mpc(
        estimator_ckpt=str(est_ckpt), pinn_ckpt=str(pinn_ckpt),
        core_path="fake.dll", rom_path="fake.sfc", state_path=state_file,
        max_frames=2, output_dir=str(out),
    )
    assert metrics["survived_frames"] == 2
    assert np.isfinite(metrics["mean_estimator_mae_xyv"])
    assert (out / "pixel_mpc_metrics.json").exists()


# ---------- training scripts on synthetic data ----------

def test_train_pixel_estimator_smoke(tmp_path):
    rng = np.random.default_rng(2)
    np.savez_compressed(
        tmp_path / "pix.npz",
        frames=rng.integers(0, 256, size=(24, 10, 8, 3), dtype=np.uint8),
        states=rng.normal(0, 5, size=(24, 8)).astype(np.float32),
    )
    out = tmp_path / "results"
    metrics = train_pixel_mod.run_training(
        dataset_path=str(tmp_path / "pix.npz"), epochs=2, batch_size=8,
        output_dir=str(out),
    )
    assert np.isfinite(metrics["best_val_loss"])
    assert (out / "checkpoints" / "pixel_estimator_best.pt").exists()


def test_train_terminal_value_smoke(tmp_path):
    n = 60
    log = {
        "x": list(np.linspace(0, 200, n)),
        "y": [300.0] * n,
        "vx": [10.0] * n,
        "vy": [0.0] * n,
    }
    with open(tmp_path / "traj.json", "w", encoding="utf-8") as f:
        json.dump(log, f)
    out = tmp_path / "results"
    metrics = train_value_mod.run_training(
        trajectory_log=str(tmp_path / "traj.json"), epochs=5, output_dir=str(out)
    )
    assert np.isfinite(metrics["r2_on_training_log"])
    assert (out / "checkpoints" / "terminal_value_best.pt").exists()


def test_train_unified_smoke(tmp_path):
    rng = np.random.default_rng(3)
    n = 32
    np.savez_compressed(
        tmp_path / "tile.npz",
        states=rng.normal(0, 5, size=(n, 8)).astype(np.float32),
        actions=rng.integers(0, 2, size=(n, 6)).astype(np.float32),
        tile_patches=rng.integers(0, 4, size=(n, 7, 7)),
        next_states=rng.normal(0, 5, size=(n, 8)).astype(np.float32),
    )
    s12 = rng.normal(0, 5, size=(n, 12)).astype(np.float32)
    s12[:, 11] = (rng.random(n) > 0.5).astype(np.float32)
    np.savez_compressed(
        tmp_path / "multi.npz",
        states=s12,
        actions=rng.integers(0, 2, size=(n, 6)).astype(np.float32),
        next_states=rng.normal(0, 5, size=(n, 12)).astype(np.float32),
        episodes=np.zeros(n, dtype=np.int32),
    )
    out = tmp_path / "results"
    metrics = train_unified_mod.run_training(
        tilemap_path=str(tmp_path / "tile.npz"), multi_path=str(tmp_path / "multi.npz"),
        epochs=2, batch_size=8, output_dir=str(out),
    )
    assert np.isfinite(metrics["best_val_loss"])
    assert (out / "checkpoints" / "unified_joint_best.pt").exists()


def test_train_set_multi_entity_smoke(tmp_path):
    rng = np.random.default_rng(4)
    n, k = 32, 4
    ent = np.zeros((n, k, 5), dtype=np.float32)
    ent[:, 0, :] = np.array([30.0, 0.0, -16.0, 0.0, 1.0], dtype=np.float32)
    np.savez_compressed(
        tmp_path / "set.npz",
        mario=rng.normal(0, 5, size=(n, 8)).astype(np.float32),
        entities=ent,
        actions=rng.integers(0, 2, size=(n, 6)).astype(np.float32),
        next_mario=rng.normal(0, 5, size=(n, 8)).astype(np.float32),
        next_entities=ent + rng.normal(0, 0.5, size=(n, k, 5)).astype(np.float32),
    )
    out = tmp_path / "results"
    metrics = train_set_mod.run_training(
        dataset_path=str(tmp_path / "set.npz"), epochs=2, batch_size=8,
        output_dir=str(out),
    )
    assert np.isfinite(metrics["best_val_loss"])
    assert metrics["max_entities"] == k
    assert (out / "checkpoints" / "set_multi_entity_best.pt").exists()


# ---------- spatial holdout + learning curves ----------

def test_run_spatial_holdout_synthetic(tmp_path):
    n = 200
    xs = np.linspace(0, 1000, n).astype(np.float32)
    states = np.zeros((n, 8), dtype=np.float32)
    states[:, 0] = xs
    states[:, 1] = 300.0
    np.savez_compressed(
        tmp_path / "game.npz",
        states=states,
        actions=np.zeros((n, 6), dtype=np.float32),
        next_states=states + 0.5,
        episodes=np.zeros(n, dtype=np.int32),
    )
    s12 = np.zeros((n, 12), dtype=np.float32)
    s12[:, 0] = xs
    s12[:, 11] = (xs > 700).astype(np.float32)
    np.savez_compressed(
        tmp_path / "multi.npz",
        states=s12,
        actions=np.zeros((n, 6), dtype=np.float32),
        next_states=s12,
        episodes=np.zeros(n, dtype=np.int32),
    )
    ckdir = tmp_path / "ckpt"
    ckdir.mkdir()
    torch.save(StatisticalMLPDynamics().state_dict(), ckdir / "mlp_best.pt")
    torch.save(HardResidualPINNDynamics().state_dict(), ckdir / "pinn_hard_best.pt")
    out = tmp_path / "results"
    payload = holdout_mod.run_spatial_holdout(
        dataset_path=str(tmp_path / "game.npz"), multi_path=str(tmp_path / "multi.npz"),
        checkpoints_dir=str(ckdir), output_dir=str(out),
    )
    assert payload["far_8d_transitions"] > 0
    assert payload["far_hazard_active_transitions"] > 0
    assert set(payload["single_step"]) == {"Statistical_MLP", "Hard_Residual_PINN"}
    assert (out / "spatial_holdout_metrics.json").exists()
    assert (out / "figures" / "spatial_holdout_comparison.png").exists()


def test_run_plot_learning_curves_synthetic(tmp_path):
    mf = {
        "total_real_steps": 4000, "mean_final_return": 500.0,
        "step_history": [1000, 2000, 3000, 4000],
        "return_history": [100.0, 200.0, 300.0, 500.0],
    }
    dyna = {"Dyna_PPO_Hard_PINN": {"total_progress_pixels": 115.0}}
    se = {
        "sample_sizes": [200, 5000],
        "results": {
            "Statistical_MLP": {"test_mse": [70.0, 12.0]},
            "Hard_Residual_PINN": {"test_mse": [0.7, 0.6]},
        },
    }
    for name, payload in [
        ("model_free_ppo_metrics.json", mf),
        ("dyna_ppo_metrics.json", dyna),
        ("sample_efficiency_metrics.json", se),
    ]:
        with open(tmp_path / name, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    payload = curves_mod.run_plot(output_dir=str(tmp_path))
    assert payload["model_free_frames_to_converge"] == 4000
    assert (tmp_path / "figures" / "learning_curve_comparison.png").exists()
    assert (tmp_path / "learning_curve_metrics.json").exists()
