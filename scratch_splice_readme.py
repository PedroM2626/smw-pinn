"""Scratch: (re)splice the generated hero block and section 10.60 into README.md, idempotently."""

import json

from src.evaluation.dashboard_study import render_caption, render_section

payload = json.load(open("results/dashboard_study_metrics.json", encoding="utf-8"))
lines = open("README.md", encoding="utf-8").read().splitlines()

HERO = "## What the Actor Sees, and What It Believed"
SECTION = "### 10.60 "


def _drop_block(src, start_pred, end_pred):
    out, i = [], 0
    while i < len(src):
        if start_pred(src[i]):
            while i < len(src) and not end_pred(src[i]):
                i += 1
            continue
        out.append(src[i])
        i += 1
    return out


# remove any previously spliced hero (ends at its trailing '---') and section (ends at '## 11.')
work = _drop_block(lines, lambda l: l == HERO, lambda l: l.strip() == "---")
work = _drop_block(work, lambda l: l.startswith(SECTION), lambda l: l.startswith("## 11. "))

caption = render_caption(payload)
section = render_section(payload)

anchor = next(i for i, line in enumerate(work[:30]) if line.strip() == "---")
out = work[: anchor + 1] + [""] + caption + ["---", ""] + work[anchor + 1 :]

head = next(i for i, line in enumerate(out) if line.startswith("## 11. "))
out = out[:head] + section + out[head:]

open("README.md", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("hero lines:", len(caption), "section lines:", len(section), "total:", len(out))
