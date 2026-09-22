"""
navigate_to_level.py
Navega pelo boot do Super Mario World via emulação direta, seleciona o arquivo 1,
entra na primeira fase (Yoshi's Island 1) e salva o estado de início da fase (savestate).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.snes_emulator import SnesLibretroEmulator


def boot_and_enter_level(
    rom_path: str = "data/raw/smw_usa.sfc",
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    state_out_path: str = "data/raw/smw_yoshi_island_1.state",
    level: int = 1,
):
    print(f"Iniciando emulador com ROM: {rom_path}...")
    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)

    # Passo 1: Avançar até a tela de título (~200 frames)
    print("Aguardando logo da Nintendo e tela de título...")
    for f in range(220):
        emu.step_frame()

    # Passos 2-3: sair da tela de titulo (0x07) e do seletor de arquivos (0x08).
    # START agora chega de verdade ao emulador; alterna START/B com polling
    # ate sair desses modos (entra no mapa/intro).
    print("Saindo do titulo / seletor de arquivos...")
    for _ in range(15):
        mode = emu.read_wram_u8(0x0100)
        if mode not in (0x07, 0x08):
            break
        for _ in range(20):
            emu.set_input({"START": True})
            emu.step_frame()
        mode = emu.read_wram_u8(0x0100)
        if mode not in (0x07, 0x08):
            break
        for _ in range(20):
            emu.set_input({"B": True})
            emu.step_frame()
    for _ in range(60):
        emu.set_input({})
        emu.step_frame()

    # Passo 4: No mapa do mundo, pode aparecer a caixa de introdução
    print("Avançando introdução do mapa do mundo...")
    for _ in range(180):
        emu.set_input({"B": True})  # B fecha caixa de texto
        emu.step_frame()

    # Passo 5: Mover para a fase alvo e entrar (pressionar B ou Start no ponto da fase)
    # level=1: Yoshi's Island 1 (ponto inicial, com LEFT de seguranca).
    # level=2: Yoshi's Island 2 (um ponto a DIREITA no mapa, acessivel sem
    # completar a fase 1). Experimental: verificado por modo/coordenadas abaixo.
    print("Navegando no mapa para a fase alvo...")
    if level == 2:
        # Yoshi's Island 2 fica a direita da Yoshi's House no mapa. Como a
        # entrada com B para no PRIMEIRO ponto (a House), primeiro atravessa
        # (~500 quadros andando para a direita passam pelo ponto da House) e
        # so depois tenta entrar com B.
        print("Atravessando o ponto da Yoshi's House em direcao a fase 2...")
        for _ in range(500):
            emu.set_input({"RIGHT": True})
            emu.step_frame()
        for _ in range(20):
            emu.set_input({})
            emu.step_frame()
        entered = False
        for round_idx in range(20):
            for _ in range(30):
                emu.set_input({"RIGHT": True})
                emu.step_frame()
            for _ in range(10):
                emu.set_input({})
                emu.step_frame()
            for _ in range(10):
                emu.set_input({"B": True})
                emu.step_frame()
            if emu.read_wram_u8(0x0100) == 0x14:
                entered = True
                print(f"Ponto da fase 2 alcançado na rodada {round_idx}.")
                break
        if not entered:
            print("Aviso: ponto da fase 2 nao confirmado; seguindo para espera de modo.")
    else:
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
                if level == 2:
                    # Yoshi's Island 2: o slide de entrada abre uma message box
                    # que so sai com pulso de Y (edge no latch $7E:0016; B/A/X
                    # segurados nao dispensam). Apos dispensar ha um fade
                    # (modo sai de 0x14 e volta); so confiar apos estabilizar.
                    print("Dispensando message box da fase 2 com pulsos de Y...")
                    for _ in range(240):
                        emu.set_input({})
                        emu.step_frame()
                    for _ in range(12):
                        for _ in range(10):
                            emu.set_input({"Y": True})
                            emu.step_frame()
                        for _ in range(40):
                            emu.set_input({})
                            emu.step_frame()
                        if emu.read_wram_u8(0x0100) == 0x14 and emu.read_wram_u8(0x0072) in (0, 1, 2):
                            break
                    st = emu.get_smw_state()
                    print(f"Estado pos-mensagem: {st}")
                    # Estabilidade: WRAM zerada durante transicoes passa em
                    # checagens ingenuas (air==0 em memoria zerada). Exige 60
                    # quadros consecutivos plausiveis antes de confiar.
                    def _plausible() -> bool:
                        s = emu.get_smw_state()
                        return (
                            emu.read_wram_u8(0x0100) == 0x14
                            and 0.0 < s["x"] < 4000.0
                            and 0.0 < s["y"] < 450.0
                            and emu.read_wram_u8(0x0072) in (0, 1, 2)
                        )

                    stable = 0
                    for _ in range(1200):
                        emu.set_input({})
                        emu.step_frame()
                        stable = stable + 1 if _plausible() else 0
                        if stable >= 60:
                            break
                    if stable < 60:
                        raise RuntimeError("Level 2: estado nunca estabilizou. Savestate NAO salvo.")
                    # Gate de honestidade: so salva se o Mario responde a inputs
                    # E termina em estado plausivel.
                    x_before = emu.get_smw_state()["x"]
                    for _ in range(90):
                        emu.set_input({"RIGHT": True, "Y": True})
                        emu.step_frame()
                    st_after = emu.get_smw_state()
                    dx = st_after["x"] - x_before
                    print(f"Verificacao de movimento: dx={dx:.1f}px, estado={st_after}")
                    if dx < 10.0 or not _plausible():
                        raise RuntimeError(
                            f"Level {level}: verificacao falhou (dx={dx:.1f}). "
                            "Savestate NAO salvo."
                        )
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
    import argparse

    parser = argparse.ArgumentParser(description="Boot SMW and capture a level savestate.")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2])
    parser.add_argument("--state-out-path", default=None)
    args = parser.parse_args()
    out = args.state_out_path or (
        "data/raw/smw_yoshi_island_2.state" if args.level == 2 else "data/raw/smw_yoshi_island_1.state"
    )
    boot_and_enter_level(state_out_path=out, level=args.level)
