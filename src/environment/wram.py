"""
wram.py
Single source of truth for the Super Mario World WRAM register map.

Every address here is a `$7E:` offset inside the 128 KB SNES Working RAM that
the reverse-engineered engine uses for player physics, sprite state and level
geometry. These values used to be copy-pasted as bare hex literals across the
evaluation scripts; changing one (for example the pit-death threshold) now
means editing exactly one line.

See README section 3.3 for the reverse-engineering evidence behind each entry.
"""

from __future__ import annotations

# --- Execution mode -----------------------------------------------------------
ADDR_GAME_MODE = 0x0100  # $7E:0100 - current engine mode byte
GAME_MODE_INTERACTIVE = 0x14  # Level/gameplay mode (physics are live, WRAM valid)
GAME_MODE_TITLE = 0x07  # Title screen
GAME_MODE_FILE_SELECT = 0x08  # File selector
GAME_MODE_WORLD_MAP = 0x0E  # World map / node selection
ADDR_BUTTON_LATCH = 0x0016  # $7E:0016 - debounced joypad latch byte

# --- Player physics -----------------------------------------------------------
ADDR_PLAYER_X = 0x0094  # $7E:0094 - Mario X, integer pixels (u16 little-endian)
ADDR_PLAYER_Y = 0x0096  # $7E:0096 - Mario Y, integer pixels (u16 little-endian)
ADDR_PLAYER_X_SUB = 0x13DA  # $7E:13DA - X subpixel accumulator (256 units = 1 px)
ADDR_PLAYER_Y_SUB = 0x13DC  # $7E:13DC - Y subpixel accumulator (256 units = 1 px)
ADDR_VX = 0x007B  # $7E:007B - horizontal speed, signed subpixels/frame
ADDR_VY = 0x007D  # $7E:007D - vertical speed, signed subpixels/frame
ADDR_COLLISION = 0x0077  # $7E:0077 - collision flags byte (SxxMUDLR)
ADDR_AIR_STATE = 0x0072  # $7E:0072 - Mario animation/air state
ADDR_POWERUP = 0x0019  # $7E:0019 - power-up state (0 small, 1 caped, 2 fire)

# Collision flag bitmasks inside ADDR_COLLISION.
COLLISION_RIGHT = 0x01
COLLISION_LEFT = 0x02
COLLISION_GROUND = 0x04
COLLISION_CEILING = 0x08

# --- Dynamic sprite tables (12 engine slots) ----------------------------------
NUM_SPRITE_SLOTS = 12
ADDR_SPRITE_STATUS = 0x14C8  # $7E:14C8 + slot - spawn/status byte (>= 8 is live)
ADDR_SPRITE_ID = 0x009E  # $7E:009E + slot - enemy ROM ID (0x05 = Rex)
ADDR_SPRITE_X_LOW = 0x00E4  # $7E:00E4 + slot - X position, low byte
ADDR_SPRITE_X_HIGH = 0x14E0  # $7E:14E0 + slot - X position, high byte
ADDR_SPRITE_Y_LOW = 0x00D8  # $7E:00D8 + slot - Y position, low byte
ADDR_SPRITE_Y_HIGH = 0x14D4  # $7E:14D4 + slot - Y position, high byte
ADDR_SPRITE_VX = 0x00B6  # $7E:00B6 + slot - horizontal speed (signed)
ADDR_SPRITE_VY = 0x00AA  # $7E:00AA + slot - vertical speed (signed)
SPRITE_STATUS_ACTIVE = 8  # Values >= 8 mean the sprite is running normally
SPRITE_ID_REX = 0x05

# --- Level geometry (tilemap) -------------------------------------------------
ADDR_TILEMAP_BASE = 0xC800  # $7E:C800 - first subscreen's horizontal level buffer
TILEMAP_SUBSCREEN_STRIDE = 0x01B0  # Bytes between consecutive subscreen buffers
TILEMAP_COLUMNS_PER_SUBSCREEN = 16  # A subscreen is 16 tiles wide
TILEMAP_ROWS_PER_SUBSCREEN = 27  # ... and 27 tiles tall (432 px)
TILE_SIZE_PX = 16  # SMW tiles are 16x16 pixels
WRAM_SIZE_BYTES = 0x20000  # 128 KB WRAM window exposed by the core

# Tile IDs used by the terrain classifier in `get_local_tilemap_patch`.
TILE_AIR = 0x25
TILE_SOLID = (0x00, 0x3F, 0x73, 0x74, 0x75, 0x76)  # ground, fill, pipes
TILE_SLOPE = (0x80, 0x81, 0x82)
TILE_HAZARD_SPIKES = (0x8C,)

# --- Engine-fixed scalar constants -------------------------------------------
SUBPIXELS_PER_PIXEL = 16.0  # 1/16 px velocity units, from the fixed-point engine
FRAMES_PER_SECOND = 60.0  # SNES refresh rate
DEATH_Y = 450.0  # Below this Y the player is lost in a pit
FLOOR_Y = 336.0  # Yoshi's Island 1 ground contact altitude
