"""
record_gameplay.py
Grava trajetórias 100% genuínas e interativas da memória RAM do Super Mario World.
O emulador inicializa o jogo, entra no Modo de Fase Interativo (Game Mode 0x14)
onde o controle comanda o Mario diretamente e em tempo real pela fase.
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


def record_interactive_trajectories(
    rom_path: str = "data/raw/smw_usa.sfc",
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    output_path: str = "data/raw/smw_gameplay_dataset.npz",
    num_episodes: int = 15,
    frames_per_episode: int = 1000,
):
    print("==========================================================")
    print("  GRAVAÇÃO DE DADOS GENUÍNOS E INTERATIVOS (MODO 0x14)    ")
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

    # 1. Avançar boot e ativar Modo de Fase Interativo (0x14)
    print("Inicializando emulador e habilitando modo de fase interativo (0x14)...")
    for _ in range(410):
        emu.step_frame()

    emu.wram_buffer[0x0100] = 0x14
    for _ in range(30):
        emu.step_frame()

    # Capturar o savestate de início da fase onde Mario é 100% controlável
    initial_savestate = emu.save_state()
    initial_state = emu.get_smw_state()
    print(f"Savestate interativo capturado com sucesso! Mario inicial: {initial_state}")

    # Padrões de ação diversificados para cobrir todo o envelope físico
    action_behaviors = [
        {"RIGHT": True, "Y": True},                 # Corrida contínua
        {"RIGHT": True, "Y": True, "B": True},       # Corrida com salto alto
        {"RIGHT": True, "B": True},                 # Caminhada com salto
        {"RIGHT": True},                            # Caminhada simples
        {"LEFT": True, "Y": True},                  # Corrida para esquerda / derrapagem
        {"LEFT": True},                             # Caminhada para esquerda
        {},                                         # Inércia / desaceleração natural por atrito
        {"B": True},                                # Salto vertical estacionário
        {"RIGHT": True, "DOWN": True},              # Deslizamento agachado
        {"RIGHT": True, "Y": True, "A": True},       # Spin Jump correndo
    ]

    total_transitions = 0
    t0 = time.time()

    for ep in range(num_episodes):
        emu.load_state(initial_savestate)
        np.random.seed(100 + ep)

        curr_pattern = 0
        pattern_duration = np.random.randint(15, 60)
        pattern_timer = 0

        curr_state_dict = emu.get_smw_state()

        for f in range(frames_per_episode):
            # Alternância de padrões com viés para exploração horizontal
            if pattern_timer >= pattern_duration:
                # 60% chance de ações para frente (direita), 40% de manobras/saltos/inércias
                if np.random.rand() < 0.65:
                    curr_pattern = np.random.choice([0, 1, 2, 3, 9])
                else:
                    curr_pattern = np.random.choice([4, 5, 6, 7, 8])
                pattern_duration = np.random.randint(10, 50)
                pattern_timer = 0

            action_dict = action_behaviors[curr_pattern]

            # Injetar comando no controle do SNES
            emu.set_input(action_dict)

            s_vec = extract_vector(curr_state_dict)
            a_vec = extract_action_vector(action_dict)

            # Executar 1 frame da física do jogo
            emu.step_frame()

            next_state_dict = emu.get_smw_state()
            next_s_vec = extract_vector(next_state_dict)

            # Só registra se Mario ainda estiver em coordenadas válidas da fase
            if 0 <= next_s_vec[1] <= 500:
                states_t.append(s_vec)
                actions_t.append(a_vec)
                states_tp1.append(next_s_vec)
                episode_ids.append(ep)
                total_transitions += 1

            curr_state_dict = next_state_dict
            pattern_timer += 1

            # Se Mario morreu ou caiu no buraco, reinicia o episódio
            if next_s_vec[1] > 500 or next_s_vec[1] < 0:
                break

        print(
            f"Episódio {ep+1:2d}/{num_episodes} concluído | "
            f"Mario Final: X={curr_state_dict['x']:.1f}, Y={curr_state_dict['y']:.1f}, "
            f"vx={curr_state_dict['vx']:.1f}, vy={curr_state_dict['vy']:.1f} | "
            f"Transições válidas acumuladas: {total_transitions}"
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
    print(f"Dataset 100% interativo salvo em: {output_path}")
    print(f"Total de transições registradas: {len(states_t_arr)}")
    print(f"Dimensões de estados: {states_t_arr.shape}")
    print(f"Dimensões de ações: {actions_t_arr.shape}")
    print(f"Tempo de emulação: {elapsed:.2f}s ({total_transitions/elapsed:.1f} FPS)")
    print("==========================================================")


if __name__ == "__main__":
    record_interactive_trajectories()
