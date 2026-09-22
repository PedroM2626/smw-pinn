# Security Policy

## Supported versions

This is a research artifact, not a hosted service. Security work is scoped to keeping
the *development and reproduction* toolchain safe for the people running it.

| Component | Supported |
| :--- | :--- |
| `main` branch (current benchmark code) | yes |
| Tagged paper-reproduction release (`v0.1.0`) | yes |

## Reporting a vulnerability

Please do **not** open a public issue for a security problem. Email the maintainer listed
in [`CITATION.cff`](CITATION.cff) and give time to respond before any disclosure.

## What this repository deliberately does (threat model)

Reviewers of a reproducible research repo should know where the sharp edges are, so they
are stated rather than discovered:

- **Third-party native code.** `src/environment/bin/snes9x_libretro.dll` is a Snes9x
  Libretro core loaded through Python `ctypes`. It executes native machine code in the
  process. Only replace it with a core you trust; see
  [`src/environment/bin/README.md`](src/environment/bin/README.md).
- **You supply the ROM.** The commercial *Super Mario World* image is not distributed.
  The suite verifies the SHA-1 published in the README as a *warning*, not a hard gate,
  so exploratory runs with a different dump are possible; treat unverified dumps as
  untrusted inputs to the emulator core.
- **No secrets are handled or required.** Network features are optional: `wandb` is an
  opt-in extra and only activates with `--wandb` plus your own API key. Nothing in the
  default CPU/CI path talks to a network service.

## Never commit

ROM dumps, savestates containing personal data, API keys, or any `.env` file. The
`.gitignore` blocks the ROM by pattern; keep it that way.

## Dependency envelope

Dependencies are capped to the validated PyTorch/CUDA envelope (see
[`CONTRIBUTING.md`](CONTRIBUTING.md)). Dependabot (`.github/dependabot.yml`) keeps the
caps; widening them requires full re-validation. `pip check` runs in CI to catch
resolution conflicts.
