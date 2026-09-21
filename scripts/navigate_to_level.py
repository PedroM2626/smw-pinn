"""
navigate_to_level.py
Navega pelo boot do Super Mario World via emulação direta, seleciona o arquivo 1,
entra na primeira fase (Yoshi's Island 1) e salva o estado de início da fase (savestate).
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
from src.environment.snes_emulator import SnesLibretroEmulator


def boot_and_enter_level(
    rom_path: str = "data/raw/smw_usa.sfc",
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    state_out_path: str = "data/raw/smw_yoshi_island_1.state",
):
    print(f"Iniciando emulador com ROM: {rom_path}...")
    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)

    # Passo 1: Avançar até a tela de título (~200 frames)
    print("Aguardando logo da Nintendo e tela de título...")
    for f in range(220):
        emu.step_frame()

    # Passo 2: Pressionar START para entrar no menu de arquivos
    print("Pressionando START na tela de título...")
    for _ in range(15):
        emu.set_input({"START": True})
        emu.step_frame()
    for _ in range(30):
        emu.set_input({})
        emu.step_frame()

    # Passo 3: Pressionar START para selecionar o Arquivo 1 (Mario A - 1 Player)
    print("Selecionando Arquivo 1...")
    for _ in range(15):
        emu.set_input({"START": True})
        emu.step_frame()
    for _ in range(60):
        emu.set_input({})
        emu.step_frame()

    # Passo 4: No mapa do mundo, pode aparecer a caixa de introdução
    print("Avançando introdução do mapa do mundo...")
    for _ in range(180):
        emu.set_input({"B": True})  # B fecha caixa de texto
        emu.step_frame()

    # Passo 5: Mover para Yoshi's Island 1 e entrar (pressionar B ou Start no ponto da fase)
    print("Navegando no mapa para a primeira fase...")
    for _ in range(60):
        emu.set_input({"LEFT": True, "B": True})
        emu.step_frame()

    # Aguardar transição para a fase ($7E:0100 == 0x14 significa Level Game Mode)
    print("Aguardando transição de entrada na fase ($7E:0100 == 0x14)...")
    max_frames = 1500
    level_loaded = False

    for frame_idx in range(max_frames):
        game_mode = emu.read_wram_u8(0x0100)
        # Se ainda não entrou, pressiona B para confirmar entrada na fase
        if game_mode != 0x14:
            emu.set_input({"B": True})
        else:
            emu.set_input({})

        emu.step_frame()

        if game_mode == 0x14:
            # Verificar se Mario já tem coordenadas válidas dentro da fase
            x = emu.read_wram_u16_le(0x0094)
            y = emu.read_wram_u16_le(0x0096)
            if x > 0 and y > 0:
                print(f"Fase carregada no quadro {frame_idx}! Modo: 0x{game_mode:02X}, Mario X={x}, Y={y}")
                # Aguardar alguns frames para estabilização de fade-in
                for _ in range(60):
                    emu.set_input({})
                    emu.step_frame()
                level_loaded = True
                break

    if not level_loaded:
        game_mode = emu.read_wram_u8(0x0100)
        print(f"Aviso: Modo de jogo final foi 0x{game_mode:02X}. Salvando estado atual.")

    state_bytes = emu.save_state()
    with open(state_out_path, "wb") as f:
        f.write(state_bytes)

    final_state = emu.get_smw_state()
    print(f"Savestate salvo com sucesso em {state_out_path} ({len(state_bytes)} bytes)!")
    print(f"Estado inicial do Mario: {final_state}")
    emu.close()
    return final_state


if __name__ == "__main__":
    boot_and_enter_level()
