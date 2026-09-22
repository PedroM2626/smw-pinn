"""
snes_emulator.py
Pure ctypes wrapper for the Snes9x Libretro core.
Provides high-performance, headless emulation with direct frame-by-frame
access to Super Mario World Working RAM (128 KB WRAM).

Register offsets come from `src.environment.wram`; asset locations (core, ROM,
savestates) come from `src.utils.paths`, which honours the SMW_CORE / SMW_ROM /
SMW_DATA_DIR environment overrides and is anchored to the repository root
instead of the current working directory.
"""

import ctypes
from typing import Dict, List

import numpy as np

from src.environment import wram
from src.utils.paths import REPO_ROOT, require_core, require_rom

# Libretro API Constants (libretro.h)
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

RETRO_MEMORY_SAVE_RAM = 0  # 2KB SRAM (Save RAM)
RETRO_MEMORY_SYSTEM_RAM = 2  # 128KB SNES WRAM ($7E:0000 - $7F:FFFF)

# Libretro pixel formats (RETRO_PIXEL_FORMAT_* in libretro.h)
RETRO_PIXEL_FORMAT_0RGB1555 = 0
RETRO_PIXEL_FORMAT_RGB565 = 1
RETRO_PIXEL_FORMAT_RGB888 = 2

# Callback types in ctypes
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


def _convert_frame_bytes(
    raw: np.ndarray, width: int, height: int, pitch: int, pixel_format: int
) -> np.ndarray:
    """Converts a raw Libretro video row buffer to uint8 RGB [H, W, 3].

    Pure function (no emulator needed) so it is unit-testable with synthetic
    buffers. Supports the three libretro pixel formats; rows are `pitch` bytes
    wide and only the first `width` pixels are kept.
    """
    if pixel_format == RETRO_PIXEL_FORMAT_RGB888:
        # 32-bit 0x00RRGGBB stored little-endian: bytes are [B, G, R, 0].
        rows = raw.reshape(height, pitch)
        pix = rows[:, : width * 4].reshape(height, width, 4)
        rgb = np.stack([pix[:, :, 2], pix[:, :, 1], pix[:, :, 0]], axis=-1)
        return rgb.astype(np.uint8)
    # 16-bit formats: pitch holds pitch//2 pixels per row.
    row_pixels = pitch // 2
    u16 = raw.view(np.uint16).reshape(height, row_pixels)[:, :width]
    if pixel_format == RETRO_PIXEL_FORMAT_RGB565:
        r = np.rint(((u16 >> 11) & 0x1F).astype(np.float32) * (255.0 / 31.0))
        g = np.rint(((u16 >> 5) & 0x3F).astype(np.float32) * (255.0 / 63.0))
        b = np.rint((u16 & 0x1F).astype(np.float32) * (255.0 / 31.0))
    else:  # RETRO_PIXEL_FORMAT_0RGB1555 (libretro default)
        r = np.rint(((u16 >> 10) & 0x1F).astype(np.float32) * (255.0 / 31.0))
        g = np.rint(((u16 >> 5) & 0x1F).astype(np.float32) * (255.0 / 31.0))
        b = np.rint((u16 & 0x1F).astype(np.float32) * (255.0 / 31.0))
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


class SnesLibretroEmulator:
    """
    SNES emulator controller based on Snes9x Libretro via ctypes.
    """

    def __init__(self, core_path: str | None = None):
        # `require_core` resolves platform suffixes and raises a message that
        # explains how to obtain the asset (git lfs pull / SMW_CORE).
        self.core = ctypes.CDLL(require_core(core_path))
        self.rom_data = None
        self.rom_buffer = None
        self.wram_buffer = None
        self.is_loaded = False

        # Pixel frame capture (opt-in; off by default to preserve headless speed).
        self._capture_frames = False
        self._last_frame = None  # np.ndarray uint8 [H, W, 3] RGB, or None
        self._pixel_format = RETRO_PIXEL_FORMAT_0RGB1555  # libretro default

        # Joypad 1 button state (id -> 1 or 0)
        self.current_input: Dict[int, int] = {i: 0 for i in range(12)}

        # Setup C function signatures
        self._setup_signatures()

        # Register callbacks
        self._setup_callbacks()

        # Initialize core
        self.core.retro_init()

    def _setup_signatures(self):
        self.core.retro_set_environment.argtypes = [ENV_CALLBACK]
        self.core.retro_set_video_refresh.argtypes = [VIDEO_REFRESH_CALLBACK]
        self.core.retro_set_audio_sample.argtypes = [AUDIO_SAMPLE_CALLBACK]
        self.core.retro_set_audio_sample_batch.argtypes = [AUDIO_SAMPLE_BATCH_CALLBACK]
        self.core.retro_set_input_poll.argtypes = [INPUT_POLL_CALLBACK]
        self.core.retro_set_input_state.argtypes = [INPUT_STATE_CALLBACK]
        self.core.retro_set_controller_port_device.argtypes = [ctypes.c_uint, ctypes.c_uint]
        self.core.retro_set_controller_port_device.restype = None

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
        # Anchored to the repository root: the core must find the same
        # directories no matter which working directory started the process.
        system_dir = str(REPO_ROOT / "src" / "environment").encode("utf-8")
        save_dir = str(REPO_ROOT / "data").encode("utf-8")
        self._c_system_dir = ctypes.c_char_p(system_dir)
        self._c_save_dir = ctypes.c_char_p(save_dir)

        def env_callback(cmd, data):
            if cmd == 10:  # RETRO_ENVIRONMENT_SET_PIXEL_FORMAT
                # The core requests a format via *data; remember it for conversion.
                if data:
                    try:
                        self._pixel_format = int(
                            ctypes.cast(data, ctypes.POINTER(ctypes.c_int)).contents.value
                        )
                    except (ValueError, OSError):
                        pass
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
            # Headless execution: discard pixel rendering unless frame capture
            # was explicitly enabled (see enable_frame_capture).
            if not self._capture_frames or not data or width == 0 or height == 0:
                return
            try:
                n_bytes = int(pitch) * int(height)
                buf = (ctypes.c_uint8 * n_bytes).from_address(int(data))
                raw = np.frombuffer(buf, dtype=np.uint8).copy()
                self._last_frame = _convert_frame_bytes(
                    raw, int(width), int(height), int(pitch), self._pixel_format
                )
            except (ValueError, OSError):
                pass

        def audio_sample(left, right):
            pass

        def audio_sample_batch(data, frames):
            return frames

        def input_poll():
            pass

        def input_state(port, device, index, id_):
            if port == 0 and ((device & 0xFF) == RETRO_DEVICE_JOYPAD):
                return self.current_input.get(id_, 0)
            return 0

        # Retain callback references to prevent garbage collection
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

        # Connect Joypad device to Ports 0 and 1
        self.core.retro_set_controller_port_device(0, RETRO_DEVICE_JOYPAD)
        self.core.retro_set_controller_port_device(1, RETRO_DEVICE_JOYPAD)

    def load_rom(self, rom_path: str | None = None):
        # Verify the dump (existence + SHA-1 warning) before touching the core.
        rom_path = require_rom(rom_path)

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
            raise RuntimeError("Failed to load ROM in Libretro Snes9x core.")

        # Obtain direct pointer to WRAM memory (128 KB)
        wram_ptr = self.core.retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM)
        wram_size = self.core.retro_get_memory_size(RETRO_MEMORY_SYSTEM_RAM)

        if not wram_ptr or wram_size < wram.WRAM_SIZE_BYTES:
            raise RuntimeError(f"WRAM inaccessible or invalid size: {wram_size}")

        self.wram_buffer = (ctypes.c_uint8 * wram_size).from_address(wram_ptr)
        self.is_loaded = True

    def read_wram_u8(self, addr: int) -> int:
        """Reads an unsigned byte from WRAM (offset 0x0000 to 0x1FFFF)."""
        offset = addr & 0x1FFFF
        return self.wram_buffer[offset]

    def read_wram_s8(self, addr: int) -> int:
        """Reads a signed byte (two's complement) from WRAM."""
        val = self.read_wram_u8(addr)
        return val - 256 if val >= 128 else val

    def read_wram_u16_le(self, addr: int) -> int:
        """Reads a 16-bit little-endian word from WRAM."""
        b0 = self.read_wram_u8(addr)
        b1 = self.read_wram_u8(addr + 1)
        return b0 | (b1 << 8)

    def get_smw_state(self) -> Dict[str, float]:
        """
        Extracts fundamental physical state variables directly from RAM.
        """
        # Integer pixel and subpixel coordinates
        x_pos = self.read_wram_u16_le(wram.ADDR_PLAYER_X)
        y_pos = self.read_wram_u16_le(wram.ADDR_PLAYER_Y)
        x_sub = self.read_wram_u8(wram.ADDR_PLAYER_X_SUB)
        y_sub = self.read_wram_u8(wram.ADDR_PLAYER_Y_SUB)

        # Exact continuous coordinates: 1 pixel = 256 register units ($7E:13DA / $7E:13DC)
        # Where 16 subpixels = 1 pixel, each unit is 1/256 pixel.
        total_x = float(x_pos) + (float(x_sub) / 256.0)
        total_y = float(y_pos) + (float(y_sub) / 256.0)

        # Velocities in subpixels/frame
        vx = float(self.read_wram_s8(wram.ADDR_VX))
        vy = float(self.read_wram_s8(wram.ADDR_VY))

        # Collision flags ($7E:0077)
        # SxxMUDLR: Bit 0=R, Bit 1=L, Bit 2=D(ground), Bit 3=U(ceiling)
        blocked = self.read_wram_u8(wram.ADDR_COLLISION)
        c_right = float(bool(blocked & wram.COLLISION_RIGHT))
        c_left = float(bool(blocked & wram.COLLISION_LEFT))
        c_ground = float(bool(blocked & wram.COLLISION_GROUND))
        c_ceiling = float(bool(blocked & wram.COLLISION_CEILING))

        air_state = float(self.read_wram_u8(wram.ADDR_AIR_STATE))
        powerup = float(self.read_wram_u8(wram.ADDR_POWERUP))

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

    def get_active_sprites(self) -> List[Dict]:
        """
        Extracts active dynamic entities/sprites from WRAM sprite tables:
        - $7E:14C8: Status table (>=8 indicates normal active execution)
        - $7E:009E: Sprite ID table (0x05=Rex, 0x0F=Goomba, 0x00-0x08=Koopas, etc.)
        - $7E:00E4 / $7E:14E0: X position registers (16-bit)
        - $7E:00D8 / $7E:14D4: Y position registers (16-bit)
        - $7E:00B6: X speed register (signed 8-bit)
        - $7E:00AA: Y speed register (signed 8-bit)
        """
        sprites = []
        for slot in range(wram.NUM_SPRITE_SLOTS):
            status = self.read_wram_u8(wram.ADDR_SPRITE_STATUS + slot)
            if status >= wram.SPRITE_STATUS_ACTIVE:
                sprite_id = self.read_wram_u8(wram.ADDR_SPRITE_ID + slot)
                x_low = self.read_wram_u8(wram.ADDR_SPRITE_X_LOW + slot)
                x_high = self.read_wram_u8(wram.ADDR_SPRITE_X_HIGH + slot)
                y_low = self.read_wram_u8(wram.ADDR_SPRITE_Y_LOW + slot)
                y_high = self.read_wram_u8(wram.ADDR_SPRITE_Y_HIGH + slot)
                x_pos = float((x_high << 8) | x_low)
                y_pos = float((y_high << 8) | y_low)
                vx = float(self.read_wram_s8(wram.ADDR_SPRITE_VX + slot))
                vy = float(self.read_wram_s8(wram.ADDR_SPRITE_VY + slot))

                sprites.append(
                    {
                        "slot": slot,
                        "id": sprite_id,
                        "status": status,
                        "x": x_pos,
                        "y": y_pos,
                        "vx": vx,
                        "vy": vy,
                    }
                )
        return sprites

    def get_nearest_hazard(
        self, mario_x: float, mario_y: float, horizon_px: float = 300.0
    ) -> Dict[str, float]:
        """
        Computes spatial displacement vector (delta_x, delta_y, vx, is_active)
        to the nearest active enemy/hazard ahead of Mario.
        """
        sprites = self.get_active_sprites()
        nearest = None
        min_dist = float("inf")

        for s in sprites:
            dx = s["x"] - mario_x
            dy = s["y"] - mario_y
            dist = np.sqrt(dx * dx + dy * dy)
            # Prioritize hazards in the active forward/horizontal window
            if -32.0 <= dx <= horizon_px and dist < min_dist:
                min_dist = dist
                nearest = (dx, dy, s["vx"], s["id"])

        if nearest is not None:
            return {
                "delta_x_enemy": float(nearest[0]),
                "delta_y_enemy": float(nearest[1]),
                "vx_enemy": float(nearest[2]),
                "hazard_active": 1.0,
                "hazard_id": float(nearest[3]),
            }
        else:
            return {
                "delta_x_enemy": horizon_px,
                "delta_y_enemy": 0.0,
                "vx_enemy": 0.0,
                "hazard_active": 0.0,
                "hazard_id": 0.0,
            }

    def get_smw_extended_state(self) -> Dict[str, float]:
        """
        Extracts 12-dimensional state representation:
        8-dimensional kinematics + 4-dimensional relative hazard perception.
        """
        base = self.get_smw_state()
        hazard = self.get_nearest_hazard(base["x"], base["y"])
        base.update(hazard)
        return base

    def get_local_tilemap_patch(
        self, mario_x: float, mario_y: float, radius: int = 3
    ) -> np.ndarray:
        """
        Extracts a (2*radius + 1) x (2*radius + 1) tile grid centered at Mario's position
        directly from the SNES WRAM level block buffer ($7E:C800).
        Each tile in Super Mario World is 16x16 pixels.

        Returns an integer array where:
            0: Air / Passable space (tile 0x25 or boundary void)
            1: Solid terrain (Grass top 0x00, earth fill 0x3F, pipes 0x73-0x76, question/turn blocks)
            2: Dynamic hazards / Spikes
            3: Slopes / Inclines
        """
        grid_dim = 2 * radius + 1
        patch = np.zeros((grid_dim, grid_dim), dtype=np.int64)

        center_tile_x = int(mario_x) // wram.TILE_SIZE_PX
        center_tile_y = int(mario_y) // wram.TILE_SIZE_PX

        for r_idx, dy in enumerate(range(-radius, radius + 1)):
            target_tile_y = center_tile_y + dy
            if target_tile_y < 0 or target_tile_y >= wram.TILEMAP_ROWS_PER_SUBSCREEN:
                continue

            for c_idx, dx in enumerate(range(-radius, radius + 1)):
                target_tile_x = center_tile_x + dx
                if target_tile_x < 0:
                    continue

                subscreen = target_tile_x // wram.TILEMAP_COLUMNS_PER_SUBSCREEN
                col_in_sub = target_tile_x % wram.TILEMAP_COLUMNS_PER_SUBSCREEN
                row_in_sub = target_tile_y

                # SMW horizontal level buffer formula: $7E:C800 + subscreen * 0x01B0 + row * 16 + col
                addr = (
                    wram.ADDR_TILEMAP_BASE
                    + subscreen * wram.TILEMAP_SUBSCREEN_STRIDE
                    + row_in_sub * wram.TILEMAP_COLUMNS_PER_SUBSCREEN
                    + col_in_sub
                )
                if addr > wram.WRAM_SIZE_BYTES - 1:
                    continue

                tile_id = self.read_wram_u8(addr)

                if tile_id == wram.TILE_AIR:
                    patch[r_idx, c_idx] = 0
                elif tile_id in wram.TILE_SOLID or (
                    tile_id != wram.TILE_AIR
                    and target_tile_y >= wram.TILEMAP_ROWS_PER_SUBSCREEN - 5
                ):
                    patch[r_idx, c_idx] = 1
                elif tile_id in wram.TILE_SLOPE:
                    patch[r_idx, c_idx] = 3
                else:
                    patch[r_idx, c_idx] = 1 if tile_id != wram.TILE_AIR else 0

        return patch

    # --- Engine mode and episode hygiene -------------------------------------

    def get_game_mode(self) -> int:
        """Current value of the engine mode byte at $7E:0100."""
        return self.read_wram_u8(wram.ADDR_GAME_MODE)

    def in_level(self) -> bool:
        """True when the engine is in interactive level/gameplay mode (0x14)."""
        return self.get_game_mode() == wram.GAME_MODE_INTERACTIVE

    def enable_gameplay_mode(self, mode: int = wram.GAME_MODE_INTERACTIVE) -> None:
        """Force the engine mode byte (physics live, WRAM telemetry valid).

        Every closed-loop benchmark used to poke `emu.enable_gameplay_mode()`
        inline; this is the single place that writes it.
        """
        if self.wram_buffer is None:
            raise RuntimeError("ROM not loaded: cannot set the engine mode byte.")
        self.wram_buffer[wram.ADDR_GAME_MODE] = mode

    def start_episode(
        self,
        savestate: bytes,
        gameplay_mode: bool = True,
        warmup_frames: int = 5,
    ) -> Dict[str, float]:
        """Restore a savestate, enter gameplay mode and settle the engine.

        The load-state -> set-mode -> warm-up -> read-state preamble was
        duplicated in ~15 evaluation scripts; stepping a few frames lets the
        engine refresh derived WRAM so the first observation is not stale.
        """
        self.load_state(savestate)
        if gameplay_mode:
            self.enable_gameplay_mode()
        for _ in range(warmup_frames):
            self.step_frame()
        return self.get_smw_state()

    def set_input(self, actions: Dict[str, bool]):
        """
        Maps semantic actions to SNES controller buttons.

        Supported keys: B, Y, A, X, UP, DOWN, LEFT, RIGHT, START, SELECT,
        L, R. Unknown keys raise KeyError instead of being silently dropped
        (a dropped START once cost a full Yoshi's Island 2 capture attempt).
        """
        mapping = {
            "B": RETRO_DEVICE_ID_JOYPAD_B,
            "Y": RETRO_DEVICE_ID_JOYPAD_Y,
            "SELECT": RETRO_DEVICE_ID_JOYPAD_SELECT,
            "START": RETRO_DEVICE_ID_JOYPAD_START,
            "UP": RETRO_DEVICE_ID_JOYPAD_UP,
            "DOWN": RETRO_DEVICE_ID_JOYPAD_DOWN,
            "LEFT": RETRO_DEVICE_ID_JOYPAD_LEFT,
            "RIGHT": RETRO_DEVICE_ID_JOYPAD_RIGHT,
            "A": RETRO_DEVICE_ID_JOYPAD_A,
            "X": RETRO_DEVICE_ID_JOYPAD_X,
            "L": RETRO_DEVICE_ID_JOYPAD_L,
            "R": RETRO_DEVICE_ID_JOYPAD_R,
        }
        unknown = sorted(k for k in actions if k not in mapping)
        if unknown:
            raise KeyError(f"Unknown joypad buttons: {unknown}. Valid: {sorted(mapping)}.")
        for key, btn_id in mapping.items():
            self.current_input[btn_id] = 1 if actions.get(key, False) else 0

    def step_frame(self):
        """Advances emulator execution by exactly 1 frame (1/60 second)."""
        self.core.retro_run()

    def save_state(self) -> bytes:
        """Serializes complete machine state into raw bytes."""
        size = self.core.retro_serialize_size()
        buf = ctypes.create_string_buffer(size)
        success = self.core.retro_serialize(buf, size)
        if not success:
            raise RuntimeError("Failed to serialize Libretro savestate.")
        return buf.raw

    def load_state(self, state_bytes: bytes):
        """Restores complete machine state from raw savestate bytes."""
        buf = ctypes.create_string_buffer(state_bytes, len(state_bytes))
        success = self.core.retro_unserialize(buf, len(state_bytes))
        if not success:
            raise RuntimeError("Failed to unserialize Libretro savestate.")
        # Reset controller input latches to prevent input state contamination across episodes
        self.current_input = {i: 0 for i in range(12)}

    def close(self):
        if self.is_loaded:
            self.core.retro_unload_game()
            self.is_loaded = False
        self.core.retro_deinit()

    def enable_frame_capture(self, enabled: bool = True) -> None:
        """Opt-in RGB frame capture (default off: zero headless overhead).

        When enabled, every `step_frame()` stores the latest rendered frame,
        retrievable via `get_frame()`. SNES frames are natively 256x224
        (up to 512x448 in hires modes).
        """
        self._capture_frames = bool(enabled)
        if not enabled:
            self._last_frame = None

    def get_frame(self) -> np.ndarray | None:
        """Returns a copy of the last captured RGB frame [H, W, 3] uint8.

        Returns None if capture is disabled or no frame arrived yet (e.g. the
        core duplicated the previous frame with data=NULL).
        """
        if self._last_frame is None:
            return None
        return self._last_frame.copy()
