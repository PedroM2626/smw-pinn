"""
train_dagger.py
Interactive Multi-Iteration Dataset Aggregation (DAgger) Training.
Reference: Ross & Bagnell (AISTATS 2011) - "A Reduction of Imitation Learning and
Structured Prediction to No-Regret Online Learning".

Eliminates the compounding error O(T^2) of pure Behavioral Cloning by querying
the live CEM MPC oracle on the exact state distribution induced by the learned policy.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.environment.snes_emulator import SnesLibretroEmulator
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.models.pinn_multi_entity import MultiEntityPINNDynamics
from src.planning.mpc_planner import (
    ModelPredictiveController,
    TrajectoryObjective,
)
from src.training.distill_mpc_policy import (
    DistilledActorPolicy,
    collect_mpc_expert_demonstrations,
    extract_12d_vector,
)


def run_dagger_loop(
    rom_path: str = "data/raw/smw_usa.sfc",
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    state_path: str = "data/raw/smw_yoshi_island_1.state",
    model_checkpoint: str = "results/checkpoints/pinn_multi_entity_best.pt",
    output_policy_path: str = "results/checkpoints/dagger_policy_best.pt",
    output_metrics_path: str = "results/dagger_training_metrics.json",
    dagger_iterations: int = 3,
    episodes_per_iter: int = 2,
    frames_per_episode: int = 400,
    epochs_per_iter: int = 30,
):
    print("====================================================================")
    print("  INTERACTIVE DAGGER (DATASET AGGREGATION) FOR AMORTIZED CONTROL     ")
    print("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | DAgger Iterations: {dagger_iterations}")

    # 1. Setup Expert Oracle (CEM MPC)
    base_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6)
    world_model = MultiEntityPINNDynamics(base_pinn=base_pinn).to(device)
    if os.path.exists(model_checkpoint):
        world_model.load_state_dict(torch.load(model_checkpoint, map_location=device, weights_only=True))
    world_model.eval()

    objective = TrajectoryObjective(
        weight_progress=3.0,
        weight_velocity=0.5,
        pit_penalty=1000.0,
        death_y=450.0,
        hazard_penalty=800.0,
        leap_bonus=350.0,
    )

    oracle = ModelPredictiveController(
        world_model=world_model,
        device=device,
        horizon=16,
        num_candidates=256,
        cem_iterations=3,
        objective=objective,
    )

    # 2. Iteration 0: Initial dataset from expert demonstrations
    init_s, init_a = collect_mpc_expert_demonstrations(
        rom_path=rom_path,
        core_path=core_path,
        state_path=state_path,
        model_checkpoint=model_checkpoint,
        num_episodes=2,
        max_frames=frames_per_episode,
        device=device,
    )

    aggregated_states = list(init_s)
    aggregated_actions = list(init_a)
    print(f"Iteration 0 (Expert Seed): {len(aggregated_states)} samples collected.")

    # Initialize Policy Network
    policy = DistilledActorPolicy(state_dim=12, action_dim=6).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)

    with open(state_path, "rb") as f:
        initial_savestate = f.read()

    iter_logs = []

    # 3. DAgger Iterations
    for it in range(1, dagger_iterations + 1):
        print(f"\n>>> DAGGER ITERATION {it}/{dagger_iterations} <<<")

        # Step A: Train policy on current aggregated dataset
        dataset = TensorDataset(
            torch.tensor(np.array(aggregated_states), dtype=torch.float32),
            torch.tensor(np.array(aggregated_actions), dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        policy.train()
        for ep in range(1, epochs_per_iter + 1):
            total_loss = 0.0
            batches = 0
            for s_b, a_b in loader:
                s_b, a_b = s_b.to(device), a_b.to(device)
                optimizer.zero_grad()
                logits = policy(s_b)
                loss = loss_fn(logits, a_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1

        print(f"Policy retrained on {len(aggregated_states)} samples | Final BCE Loss: {total_loss/batches:.4f}")

        # Step B: Rollout learned policy on SNES and query Oracle for corrective labels
        policy.eval()
        new_samples = 0
        policy_progresses = []

        for ep in range(episodes_per_iter):
            emu.load_state(initial_savestate)
            emu.wram_buffer[0x0100] = 0x14
            for _ in range(5):
                emu.step_frame()

            prev_b = False
            start_x = emu.get_smw_state()["x"]

            for frame in range(frames_per_episode):
                s_dict = emu.get_smw_extended_state()
                s_vec = extract_12d_vector(s_dict)

                # Learned policy executes action in the environment (on-policy distribution)
                policy_action_dict, _ = policy.predict_action(s_vec, threshold=0.5)

                # Re-jump edge trigger handling
                if policy_action_dict["B"] and prev_b and frame % 12 == 0:
                    policy_action_dict["B"] = False
                prev_b = policy_action_dict["B"]

                # Oracle computes expert label for visited state s_t
                oracle_action_vec, _ = oracle.plan(s_vec)

                # Aggregate (s_t, a_t^*)
                aggregated_states.append(s_vec)
                aggregated_actions.append(oracle_action_vec)
                new_samples += 1

                # Execute policy action on emulator
                emu.set_input(policy_action_dict)
                emu.step_frame()

                if s_dict["y"] > 450.0:
                    break

            final_x = emu.get_smw_state()["x"]
            policy_progresses.append(final_x - start_x)

        mean_prog = float(np.mean(policy_progresses))
        print(f"Iteration {it} Complete | New samples added: {new_samples} | "
              f"Total Dataset: {len(aggregated_states)} | Policy Hardware Progress: {mean_prog:.1f} px")

        iter_logs.append({
            "iteration": it,
            "total_samples": len(aggregated_states),
            "policy_mean_progress_px": mean_prog,
            "bce_loss": float(total_loss / batches),
        })

    emu.close()

    # Save final model
    os.makedirs(os.path.dirname(output_policy_path), exist_ok=True)
    torch.save(policy.cpu().state_dict(), output_policy_path)
    print(f"\nFinal DAgger policy weights saved to: {output_policy_path}")

    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    with open(output_metrics_path, "w") as f:
        json.dump({"iterations": iter_logs, "final_samples": len(aggregated_states)}, f, indent=2)

    return iter_logs


if __name__ == "__main__":
    run_dagger_loop()
