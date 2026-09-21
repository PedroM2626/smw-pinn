"""
record_gameplay.py
Grava trajetórias genuínas de transição de estado da memória RAM do Super Mario World
utilizando o emulador SNES em ctypes, sem qualquer dado sintético ou fictício.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.environment.snes_emulator import SnesLibretroEmulator


def extract_vector(state_dict: dict) -> np.ndarray:
    """Converte o dicionário de estado em vetor numpy de 8 dimensões."""
    return np.array(
        [
            state_dict["x"],
            state_dict["y"],
            state_dict["vx"],
            state_dict["vy"],
            state_dict["c_ground"],
            state_dict["c_ceiling"],
            state_dict["c_left"],
            state_dict["c_right"],
        ],
        dtype=np.float32,
    )


def extract_action_vector(action_dict: dict) -> np.ndarray:
    """Converte o dicionário de ações em vetor numpy de 6 dimensões [B, Y, UP, DOWN, LEFT, RIGHT]."""
    return np.array(
        [
            1.0 if action_dict.get("B", False) else 0.0,
            1.0 if action_dict.get("Y", False) else 0.0,
            1.0 if action_dict.get("UP", False) else 0.0,
            1.0 if action_dict.get("DOWN", False) else 0.0,
            1.0 if action_dict.get("LEFT", False) else 0.0,
            1.0 if action_dict.get("RIGHT", False) else 0.0,
        ],
        dtype=np.float32,
    )


def record_trajectories(
    rom_path: str = "data/raw/smw_usa.sfc",
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    output_path: str = "data/raw/smw_gameplay_dataset.npz",
    num_episodes: int = 10,
    frames_per_episode: int = 1200,  # 20 segundos por episódio a 60 FPS
):
    print("==========================================================")
    print("  COLETA DE DADOS GENUÍNOS DA RAM DO SUPER MARIO WORLD    ")
    print("==========================================================")
    print(f"ROM: {rom_path}")
    print(f"Core: {core_path}")
    print(f"Episódios: {num_episodes} | Frames por episódio: {frames_per_episode}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    states_t = []
    actions_t = []
    states_tp1 = []
    episode_ids = []

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)

    # Avançar o boot inicial até o início do gameplay (frame 420)
    print("Avançando frames de boot inicial...")
    for _ in range(420):
        emu.step_frame()

    # Salvar o estado inicial de início da fase
    initial_savestate = emu.save_state()
    print("Savestate inicial capturado com sucesso!")

    # Estratégias de ação variadas para cobrir o espaço de estados físicos
    # (corrida com salto, caminhada, inércia, derrapagem, saltos curtos e longos)
    action_patterns = [
        {"RIGHT": True, "Y": True},  # Corrida contínua para a direita
        {"RIGHT": True, "Y": True, "B": True},  # Corrida com saltos altos
        {"RIGHT": True, "B": True},  # Caminhada com saltos
        {"RIGHT": True},  # Caminhada simples
        {"LEFT": True, "Y": True},  # Corrida para a esquerda (retorno/derrapagem)
        {},  # Inércia / desaceleração por atrito
        {"B": True},  # Salto vertical no lugar
        {"RIGHT": True, "DOWN": True},  # Agachado deslizando
    ]

    total_transitions = 0
    t0 = time.time()

    for ep in range(num_episodes):
        # Resetar para o estado inicial da fase via savestate
        emu.load_state(initial_savestate)

        # Escolher semente de padrão de controle para o episódio
        np.random.seed(42 + ep)
        current_pattern_idx = ep % len(action_patterns)
        pattern_duration = np.random.randint(15, 60)
        pattern_counter = 0

        curr_state_dict = emu.get_smw_state()

        for f in range(frames_per_episode):
            # Alternar ações para gerar diversidade dinâmica
            if pattern_counter >= pattern_duration:
                # Com probabilidade 0.4 mantém movimento para direita com salto,
                # para garantir progresso na fase
                if np.random.rand() < 0.6:
                    current_pattern_idx = np.random.choice([0, 1, 2, 3])
                else:
                    current_pattern_idx = np.random.randint(0, len(action_patterns))
                pattern_duration = np.random.randint(10, 50)
                pattern_counter = 0

            action_dict = action_patterns[current_pattern_idx]
            emu.set_input(action_dict)

            # Executar transição física de 1 frame
            s_vec = extract_vector(curr_state_dict)
            a_vec = extract_action_vector(action_dict)

            emu.step_frame()
            next_state_dict = emu.get_smw_state()
            next_s_vec = extract_vector(next_state_dict)

            states_t.append(s_vec)
            actions_t.append(a_vec)
            states_tp1.append(next_s_vec)
            episode_ids.append(ep)

            curr_state_dict = next_state_dict
            pattern_counter += 1
            total_transitions += 1

        print(
            f"Episódio {ep+1:2d}/{num_episodes} concluído | "
            f"Mario Final: X={curr_state_dict['x']:.1f}, Y={curr_state_dict['y']:.1f}, "
            f"vx={curr_state_dict['vx']:.1f}, vy={curr_state_dict['vy']:.1f} | "
            f"Transições acumuladas: {total_transitions}"
        )

    emu.close()
    elapsed = time.time() - t0

    states_t_arr = np.array(states_t, dtype=np.float32)
    actions_t_arr = np.array(actions_t, dtype=np.float32)
    states_tp1_arr = np.array(states_tp1, dtype=np.float32)
    episode_ids_arr = np.array(episode_ids, dtype=np.int32)

    np.savez_compressed(
        output_path,
        states=states_t_arr,
        actions=actions_t_arr,
        next_states=states_tp1_arr,
        episodes=episode_ids_arr,
    )

    print("\n==========================================================")
    print(f"Dataset salvo com sucesso em: {output_path}")
    print(f"Total de transições registradas: {len(states_t_arr)}")
    print(f"Dimensões do tensor de estados: {states_t_arr.shape}")
    print(f"Dimensões do tensor de ações: {actions_t_arr.shape}")
    print(f"Tempo total de emulação: {elapsed:.2f}s ({total_transitions/elapsed:.1f} FPS)")
    print("==========================================================")


if __name__ == "__main__":
    record_trajectories()
