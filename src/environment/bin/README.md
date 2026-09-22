# Bundled emulator core (third-party notice)

This directory holds the native Libretro core the suite drives through Python `ctypes`:

- `snes9x_libretro.dll` - Snes9x Libretro core (64-bit Windows build), the exact binary
  the published results were recorded with. `snes9x_libretro.dll.zip` is its Git-LFS
  archive form.

## What it is and is not

- It is **third-party software** with its own license terms (the Snes9x non-commercial
  license); it is not covered by this repository's MIT license, which applies to the
  benchmark code in `src/`, `tests/` and `scripts/` only.
- It is **native machine code** executed inside the Python process. Replace it only with
  a core build you trust (see `SECURITY.md`).

## The ROM is not here

The commercial *Super Mario World (USA)* image is intentionally **not** distributed.
Supply your own dump and point the suite at it; the expected SHA-1 and the `SMW_ROM`
override are documented in README Section 11.2.

## Locating / overriding the core

`src/utils/paths.py` resolves the core (in order): the `SMW_CORE` environment variable,
else the platform-correct file next to this README. On Linux/macOS, drop a
`snes9x_libretro.so` / `.dylib` here (or set `SMW_CORE`) and no code change is needed -
`resolve_core_path()` appends the right suffix automatically.

## Large files

These binaries are versioned through Git-LFS. Fetch them with:

```bash
git lfs install
git lfs pull
```

Without an LFS pull the file is a small pointer and the emulator-dependent tests skip.
