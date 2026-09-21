"""
benchmark_experiment.py
Script orquestrador do experimento comparativo acadêmico completo:
Treina os 4 modelos sob as mesmas condições, avalia acurácia de passo único,
estabilidade em horizonte longo (rollout drift), curva de eficiência amostral
e gera gráficos analíticos comparativos.
"""

import json
import os
import sys
import time
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sys.path.insert(0, os.path.abspath("."))
from src.environment.dataset_loader import (
    SMWSequenceDataset,
    create_dataloaders,
    load_and_preprocess_data,
)
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.models import (
    HardResidualPINNDynamics,
    SoftPINNDynamics,
    StatisticalLSTMDynamics,
    StatisticalMLPDynamics,
)
from src.training.trainer import DynamicsTrainer


def run_comprehensive_benchmark(
    dataset_path: str = "data/raw/smw_gameplay_dataset.npz",
    epochs: int = 35,
    batch_size: int = 128,
    seed: int = 42,
    output_dir: str = "results",
):
    print("====================================================================")
    print("  EXPERIMENTO ACADÊMICO: ML ESTATÍSTICO VS. PINN EM SUPER MARIO WORLD")
    print("====================================================================")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de processamento: {device}")
    if device.type == "cuda":
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Carregamento dos dados reais
    print("\n[1/5] Carregando e particionando dados genuínos da RAM...")
    data_dict = load_and_preprocess_data(dataset_path=dataset_path, seed=seed)
    train_loader, val_loader, test_loader = create_dataloaders(data_dict, batch_size=batch_size)

    print(
        f"Transições - Treino: {len(data_dict['train_states'])}, "
        f"Validação: {len(data_dict['val_states'])}, "
        f"Teste: {len(data_dict['test_states'])}"
    )

    state_dim = data_dict["train_states"].shape[1]
    action_dim = data_dict["train_actions"].shape[1]

    # 2. Instanciação dos Modelos
    print("\n[2/5] Inicializando arquiteturas neurais comparativas...")
    models = {
        "Statistical_MLP": StatisticalMLPDynamics(state_dim=state_dim, action_dim=action_dim),
        "Statistical_LSTM": StatisticalLSTMDynamics(state_dim=state_dim, action_dim=action_dim),
        "Soft_PINN": SoftPINNDynamics(state_dim=state_dim, action_dim=action_dim),
        "Hard_Residual_PINN": HardResidualPINNDynamics(state_dim=state_dim, action_dim=action_dim),
    }

    trainers: Dict[str, DynamicsTrainer] = {}
    histories: Dict[str, dict] = {}

    # Dataset sequencial específico para LSTM
    train_seq_ds = SMWSequenceDataset(
        data_dict["train_states"],
        data_dict["train_actions"],
        data_dict["train_next_states"],
        data_dict["train_episodes"],
        seq_len=10,
    )
    val_seq_ds = SMWSequenceDataset(
        data_dict["val_states"],
        data_dict["val_actions"],
        data_dict["val_next_states"],
        data_dict["val_episodes"],
        seq_len=10,
    )
    test_seq_ds = SMWSequenceDataset(
        data_dict["test_states"],
        data_dict["test_actions"],
        data_dict["test_next_states"],
        data_dict["test_episodes"],
        seq_len=10,
    )

    train_seq_loader = torch.utils.data.DataLoader(train_seq_ds, batch_size=batch_size, shuffle=True)
    val_seq_loader = torch.utils.data.DataLoader(val_seq_ds, batch_size=batch_size, shuffle=False)
    test_seq_loader = torch.utils.data.DataLoader(test_seq_ds, batch_size=batch_size, shuffle=False)

    # 3. Treinamento comparativo
    print("\n[3/5] Treinando modelos com protocolo unificado...")
    for name, model in models.items():
        if "lstm" in name.lower():
            m_type = "lstm"
        elif "soft" in name.lower():
            m_type = "pinn_soft"
        elif "hard" in name.lower():
            m_type = "pinn_hard"
        else:
            m_type = "mlp"

        trainer = DynamicsTrainer(
            model=model,
            model_type=m_type,
            device=device,
            learning_rate=1e-3,
            save_dir=os.path.join(output_dir, "checkpoints"),
        )
        trainers[name] = trainer

        cur_train_loader = train_seq_loader if m_type == "lstm" else train_loader
        cur_val_loader = val_seq_loader if m_type == "lstm" else val_loader

        hist = trainer.fit(
            train_loader=cur_train_loader,
            val_loader=cur_val_loader,
            epochs=epochs,
            patience=8,
            verbose=True,
        )
        histories[name] = hist

    # 4. Avaliação de Passo Único no Conjunto de Teste
    print("\n[4/5] Avaliando acurácia de passo único no Test Split...")
    single_step_results = {}
    for name, trainer in trainers.items():
        cur_test_loader = test_seq_loader if "lstm" in name.lower() else test_loader
        eval_metrics = trainer.evaluate(cur_test_loader)
        single_step_results[name] = {
            "test_loss_data": eval_metrics["val_loss_data"],
            "test_kinematic_error": eval_metrics["val_loss_kinematics"],
        }
        print(
            f"{name:20s} | Test MSE: {eval_metrics['val_loss_data']:.4f} | "
            f"Kinematic Residual: {eval_metrics['val_loss_kinematics']:.4f}"
        )

    # 5. Avaliação de Rollout Multi-passo (Trajetória Longa)
    print("\n[5/5] Executando rollouts autorregressivos multi-passo (Drift Test)...")
    evaluator = RolloutEvaluator(device=device)

    # Selecionar uma sequência contínua de 120 frames (2 segundos a 60 FPS) do conjunto de teste
    test_states = data_dict["test_states"]
    test_actions = data_dict["test_actions"]
    test_next_states = data_dict["test_next_states"]

    H = min(120, len(test_actions))
    init_state = test_states[0]
    action_seq = test_actions[:H]
    ground_truth = test_next_states[:H]

    rollout_metrics = {}
    trajectories = {}

    for name, model in models.items():
        m_type = "lstm" if "lstm" in name.lower() else "mlp"
        res = evaluator.evaluate_rollout(
            model=model,
            model_type=m_type,
            initial_state=init_state,
            action_sequence=action_seq,
            ground_truth_states=ground_truth,
        )
        rollout_metrics[name] = {
            "mean_drift_pixels": res["mean_drift"],
            "final_drift_pixels": res["final_drift"],
            "kinematic_violations": res["kinematic_violations"],
            "velocity_violations": res["velocity_violations"],
        }
        trajectories[name] = res["predicted_trajectory"]
        print(
            f"{name:20s} | Desvio Médio: {res['mean_drift']:.2f} px | "
            f"Desvio Final: {res['final_drift']:.2f} px | "
            f"Violações Cinemáticas: {res['kinematic_violations']:3d}/{H} | "
            f"Violações Vel: {res['velocity_violations']:3d}/{H}"
        )

    # 6. Geração de Gráficos e Visualizações
    print("\nGerando gráficos comparativos de alta resolução...")
    sns.set_theme(style="whitegrid")

    # Gráfico 1: Curvas de Val Loss durante o Treinamento
    plt.figure(figsize=(10, 5))
    for name, hist in histories.items():
        plt.plot(hist["val_loss"], label=f"{name} (Val MSE)", linewidth=2)
    plt.title("Convergência do Erro de Validação ao Longo das Épocas", fontsize=14, fontweight="bold")
    plt.xlabel("Época")
    plt.ylabel("Loss de Validação (Smooth L1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "training_convergence.png"), dpi=300)
    plt.close()

    # Gráfico 2: Desvio Euclidiano da Trajetória (Drift) ao Longo do Horizonte H
    plt.figure(figsize=(11, 6))
    time_steps = np.arange(1, H + 1)
    for name, model in models.items():
        m_type = "lstm" if "lstm" in name.lower() else "mlp"
        r = evaluator.evaluate_rollout(model, m_type, init_state, action_seq, ground_truth)
        plt.plot(time_steps, r["euclidean_drift"], label=f"{name}", linewidth=2.5)
    plt.title(
        "Acúmulo de Erro Autorregressivo na Trajetória do Mario (Horizonte de 120 frames / 2s)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Frame do Horizonte (t)", fontsize=11)
    plt.ylabel("Desvio Euclidiano em Relação ao Jogo Real (Pixels)", fontsize=11)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rollout_drift_comparison.png"), dpi=300)
    plt.close()

    # Gráfico 3: Trajetória 2D no Espaço do Jogo (X vs Y)
    plt.figure(figsize=(12, 6))
    plt.plot(
        ground_truth[:, 0],
        ground_truth[:, 1],
        "k--",
        label="Jogo Real (Ground Truth)",
        linewidth=3.0,
    )
    for name, traj in trajectories.items():
        plt.plot(traj[:, 0], traj[:, 1], label=f"{name}", linewidth=2.0, alpha=0.85)
    plt.gca().invert_yaxis()  # No SNES, Y=0 é o topo da tela
    plt.title("Trajetória do Mario no Plano 2D (X vs Y)", fontsize=14, fontweight="bold")
    plt.xlabel("Posição Horizontal X (Pixels)")
    plt.ylabel("Posição Vertical Y (Pixels - Invertido)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "trajectory_2d_space.png"), dpi=300)
    plt.close()

    # Salvar resultados em JSON
    all_summary = {
        "single_step_results": single_step_results,
        "rollout_metrics": rollout_metrics,
    }
    with open(os.path.join(output_dir, "benchmark_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=4)

    print("\nBenchmark principal concluído com sucesso!")
    return all_summary


if __name__ == "__main__":
    run_comprehensive_benchmark()
