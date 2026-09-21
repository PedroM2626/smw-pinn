"""
snes_emulator.py
Interface em ctypes puro para o core libretro do Snes9x.
Permite emulação de alta performance, sem interface gráfica (headless),
com acesso direto quadro a quadro à memória RAM (WRAM de 128 KB) do Super Mario World.
"""

import os
import ctypes
from typing import Dict, Optional, Tuple
import numpy as np

# Constantes do Libretro API (libretro.h)
RETRO_DEVICE_JOYPAD = 1
RETRO_DEVICE_ID_JOYPAD_B = 0
RETRO_DEVICE_ID_JOYPAD_Y = 1
RETRO_DEVICE_ID_JOYPAD_SELECT = 2
RETRO_DEVICE_ID_JOYPAD_START = 3
RETRO_DEVICE_ID_JOYPAD_UP = 4
RETRO_DEVICE_ID_JOYPAD_DOWN = 5
RETRO_DEVICE_ID_JOYPAD_LEFT = 6
RETRO_DEVICE_ID_JOYPAD_RIGHT = 7
RETRO_DEVICE_ID_JOYPAD_A = 8
RETRO_DEVICE_ID_JOYPAD_X = 9
RETRO_DEVICE_ID_JOYPAD_L = 10
RETRO_DEVICE_ID_JOYPAD_R = 11

RETRO_MEMORY_SAVE_RAM = 0    # 2KB SRAM (Save RAM)
RETRO_MEMORY_SYSTEM_RAM = 2  # 128KB WRAM do SNES ($7E:0000 - $7F:FFFF)

# Tipos de Callback em ctypes
ENV_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint, ctypes.c_void_p)
VIDEO_REFRESH_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t
)
AUDIO_SAMPLE_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int16)
AUDIO_SAMPLE_BATCH_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t
)
INPUT_POLL_CALLBACK = ctypes.CFUNCTYPE(None)
INPUT_STATE_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int16, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint
)


class RetroGameInfo(ctypes.Structure):
    _fields_ = [
        ("path", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("meta", ctypes.c_char_p),
    ]


class SnesLibretroEmulator:
    """
    Controlador do emulador SNES baseado em Snes9x Libretro via ctypes.
    """

    def __init__(self, core_path: str):
        if not os.path.exists(core_path):
            raise FileNotFoundError(f"Core libretro não encontrado em: {core_path}")

        self.core = ctypes.CDLL(core_path)
        self.rom_data = None
        self.rom_buffer = None
        self.wram_buffer = None
        self.is_loaded = False

        # Estado dos botões do controle 1 (dicionário de id -> 1 ou 0)
        self.current_input: Dict[int, int] = {i: 0 for i in range(12)}

        # Configurar assinaturas de funções C do libretro
        self._setup_signatures()

        # Registrar callbacks
        self._setup_callbacks()

        # Inicializar o core
        self.core.retro_init()

    def _setup_signatures(self):
        self.core.retro_set_environment.argtypes = [ENV_CALLBACK]
        self.core.retro_set_video_refresh.argtypes = [VIDEO_REFRESH_CALLBACK]
        self.core.retro_set_audio_sample.argtypes = [AUDIO_SAMPLE_CALLBACK]
        self.core.retro_set_audio_sample_batch.argtypes = [AUDIO_SAMPLE_BATCH_CALLBACK]
        self.core.retro_set_input_poll.argtypes = [INPUT_POLL_CALLBACK]
        self.core.retro_set_input_state.argtypes = [INPUT_STATE_CALLBACK]

        self.core.retro_init.restype = None
        self.core.retro_deinit.restype = None

        self.core.retro_load_game.argtypes = [ctypes.POINTER(RetroGameInfo)]
        self.core.retro_load_game.restype = ctypes.c_bool

        self.core.retro_unload_game.restype = None
        self.core.retro_run.restype = None

        self.core.retro_get_memory_data.argtypes = [ctypes.c_uint]
        self.core.retro_get_memory_data.restype = ctypes.c_void_p

        self.core.retro_get_memory_size.argtypes = [ctypes.c_uint]
        self.core.retro_get_memory_size.restype = ctypes.c_size_t

        self.core.retro_serialize_size.restype = ctypes.c_size_t
        self.core.retro_serialize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.core.retro_serialize.restype = ctypes.c_bool

        self.core.retro_unserialize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.core.retro_unserialize.restype = ctypes.c_bool

    def _setup_callbacks(self):
        system_dir = os.path.abspath("src/environment").encode("utf-8")
        save_dir = os.path.abspath("data").encode("utf-8")
        self._c_system_dir = ctypes.c_char_p(system_dir)
        self._c_save_dir = ctypes.c_char_p(save_dir)

        def env_callback(cmd, data):
            if cmd == 10:  # RETRO_ENVIRONMENT_SET_PIXEL_FORMAT
                return True
            elif cmd == 9:  # RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY
                if data:
                    ctypes.cast(data, ctypes.POINTER(ctypes.c_char_p))[0] = self._c_system_dir
                    return True
            elif cmd == 31:  # RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY
                if data:
                    ctypes.cast(data, ctypes.POINTER(ctypes.c_char_p))[0] = self._c_save_dir
                    return True
            elif cmd == 3:  # RETRO_ENVIRONMENT_GET_CAN_DUPE
                if data:
                    ctypes.cast(data, ctypes.POINTER(ctypes.c_bool))[0] = True
                    return True
            elif cmd == 15:  # RETRO_ENVIRONMENT_GET_VARIABLE
                return False
            return False

        def video_refresh(data, width, height, pitch):
            # Headless: ignora renderização de pixels para máxima velocidade
            pass

        def audio_sample(left, right):
            pass

        def audio_sample_batch(data, frames):
            return frames

        def input_poll():
            pass

        def input_state(port, device, index, id_):
            if port == 0 and device == RETRO_DEVICE_JOYPAD:
                return self.current_input.get(id_, 0)
            return 0

        # Manter referências para evitar garbage collection
        self._c_env = ENV_CALLBACK(env_callback)
        self._c_video = VIDEO_REFRESH_CALLBACK(video_refresh)
        self._c_audio = AUDIO_SAMPLE_CALLBACK(audio_sample)
        self._c_audio_batch = AUDIO_SAMPLE_BATCH_CALLBACK(audio_sample_batch)
        self._c_poll = INPUT_POLL_CALLBACK(input_poll)
        self._c_input = INPUT_STATE_CALLBACK(input_state)

        self.core.retro_set_environment(self._c_env)
        self.core.retro_set_video_refresh(self._c_video)
        self.core.retro_set_audio_sample(self._c_audio)
        self.core.retro_set_audio_sample_batch(self._c_audio_batch)
        self.core.retro_set_input_poll(self._c_poll)
        self.core.retro_set_input_state(self._c_input)

    def load_rom(self, rom_path: str):
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM não encontrada em: {rom_path}")

        with open(rom_path, "rb") as f:
            self.rom_data = f.read()

        self.rom_buffer = (ctypes.c_char * len(self.rom_data)).from_buffer_copy(self.rom_data)

        game_info = RetroGameInfo()
        game_info.path = rom_path.encode("utf-8")
        game_info.data = ctypes.cast(self.rom_buffer, ctypes.c_void_p)
        game_info.size = len(self.rom_data)
        game_info.meta = None

        success = self.core.retro_load_game(ctypes.byref(game_info))
        if not success:
            raise RuntimeError("Falha ao carregar a ROM no core Libretro Snes9x.")

        # Obter ponteiro direto para a memória WRAM (128 KB)
        wram_ptr = self.core.retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM)
        wram_size = self.core.retro_get_memory_size(RETRO_MEMORY_SYSTEM_RAM)

        if not wram_ptr or wram_size < 0x20000:
            raise RuntimeError(f"WRAM inacessível ou tamanho inesperado: {wram_size}")

        self.wram_buffer = (ctypes.c_uint8 * wram_size).from_address(wram_ptr)
        self.is_loaded = True

    def read_wram_u8(self, addr: int) -> int:
        """Lê um byte sem sinal da WRAM (offset 0x0000 a 0x1FFFF)."""
        offset = addr & 0x1FFFF
        return self.wram_buffer[offset]

    def read_wram_s8(self, addr: int) -> int:
        """Lê um byte com sinal (complemento de dois) da WRAM."""
        val = self.read_wram_u8(addr)
        return val - 256 if val >= 128 else val

    def read_wram_u16_le(self, addr: int) -> int:
        """Lê palavra de 16-bits little-endian."""
        b0 = self.read_wram_u8(addr)
        b1 = self.read_wram_u8(addr + 1)
        return b0 | (b1 << 8)

    def get_smw_state(self) -> Dict[str, float]:
        """
        Extrai as grandezas físicas fundamentais do Mario diretamente da RAM.
        """
        # Posições em pixels e subpixels
        x_pos = self.read_wram_u16_le(0x0094)
        y_pos = self.read_wram_u16_le(0x0096)
        x_sub = self.read_wram_u8(0x13DA)
        y_sub = self.read_wram_u8(0x13DC)

        # Posição contínua exata: 1 pixel = 256 unidades de registrador ($7E:13DA / $7E:13DC)
        # Onde 16 subpixels = 1 pixel, logo cada unidade é 1/256 de pixel.
        total_x = float(x_pos) + (float(x_sub) / 256.0)
        total_y = float(y_pos) + (float(y_sub) / 256.0)

        # Velocidades em subpixels/frame
        vx = float(self.read_wram_s8(0x007B))
        vy = float(self.read_wram_s8(0x007D))

        # Status de colisão e contato ($7E:0077)
        # SxxMUDLR: Bit 0=R, Bit 1=L, Bit 2=D(ground), Bit 3=U(ceiling)
        blocked = self.read_wram_u8(0x0077)
        c_right = float(bool(blocked & 0x01))
        c_left = float(bool(blocked & 0x02))
        c_ground = float(bool(blocked & 0x04))
        c_ceiling = float(bool(blocked & 0x08))

        air_state = float(self.read_wram_u8(0x0072))
        powerup = float(self.read_wram_u8(0x0019))

        return {
            "x": total_x,
            "y": total_y,
            "vx": vx,
            "vy": vy,
            "c_ground": c_ground,
            "c_ceiling": c_ceiling,
            "c_left": c_left,
            "c_right": c_right,
            "air_state": air_state,
            "powerup": powerup,
        }

    def set_input(self, actions: Dict[str, bool]):
        """
        Mapeia ações semânticas para os botões do controle SNES.
        """
        self.current_input[RETRO_DEVICE_ID_JOYPAD_B] = 1 if actions.get("B", False) else 0
        self.current_input[RETRO_DEVICE_ID_JOYPAD_Y] = 1 if actions.get("Y", False) else 0
        self.current_input[RETRO_DEVICE_ID_JOYPAD_A] = 1 if actions.get("A", False) else 0
        self.current_input[RETRO_DEVICE_ID_JOYPAD_X] = 1 if actions.get("X", False) else 0
        self.current_input[RETRO_DEVICE_ID_JOYPAD_UP] = 1 if actions.get("UP", False) else 0
        self.current_input[RETRO_DEVICE_ID_JOYPAD_DOWN] = 1 if actions.get("DOWN", False) else 0
        self.current_input[RETRO_DEVICE_ID_JOYPAD_LEFT] = 1 if actions.get("LEFT", False) else 0
        self.current_input[RETRO_DEVICE_ID_JOYPAD_RIGHT] = 1 if actions.get("RIGHT", False) else 0

    def step_frame(self):
        """Avança exatamente 1 frame (1/60 de segundo) no emulador."""
        self.core.retro_run()

    def save_state(self) -> bytes:
        """Serializa o estado completo da máquina (savestate) em bytes."""
        size = self.core.retro_serialize_size()
        buf = ctypes.create_string_buffer(size)
        success = self.core.retro_serialize(buf, size)
        if not success:
            raise RuntimeError("Falha ao salvar savestate do Libretro.")
        return buf.raw

    def load_state(self, state_bytes: bytes):
        """Restaura o estado completo da máquina a partir de bytes de savestate."""
        buf = ctypes.create_string_buffer(state_bytes, len(state_bytes))
        success = self.core.retro_unserialize(buf, len(state_bytes))
        if not success:
            raise RuntimeError("Falha ao restaurar savestate do Libretro.")

    def close(self):
        if self.is_loaded:
            self.core.retro_unload_game()
            self.is_loaded = False
        self.core.retro_deinit()
