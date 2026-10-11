"""
episode_hygiene.py - what to do when the engine stops processing the player.

10.59 found that most of the exceptions to §4.1 in the committed recordings are whole player records
repeating while WRAM keeps changing, and 10.60 identified the mechanism from the recorded picture:
the engine opened one of its own message boxes, which suspends the player object until the A button
arrives. A is not in this repository's six-button action set, so a recorder that ignores the hold
does not measure a stopped body - it measures an interface event and records it as physics.

`PlayerHold` is the shared decision: watch the state vector, say when it has stopped changing, and
say what to press when it has. It belongs here rather than in each recorder because the rule has to
be one rule - the difference between "the world held still" and "the game asked for a button" is the
thing being measured, and two recorders disagreeing about it would put that difference back into the
data.

The dashboard recorder of 10.60 uses the same signature with a longer trigger - it waits for the
box's text to finish printing before pressing anything, so the recorded picture is readable - and
keeps its own loop for that reason; the rule it shares with this one is `trigger`, not the wait.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

#: consecutive identical state records before the engine is assumed to be holding the player
DEFAULT_TRIGGER = 10
#: how long to keep pulsing A at a held player before giving up on the episode
DEFAULT_LIMIT = 240


@dataclass
class PlayerHold:
    """Detect 10.59's held-player signature and produce the operator's A pulse.

    Args:
        trigger: identical consecutive records before a hold is declared.
        limit: maximum frames of A pulsing per hold.
    """

    trigger: int = DEFAULT_TRIGGER
    limit: int = DEFAULT_LIMIT
    _previous: Optional[np.ndarray] = field(default=None, repr=False)
    _run: int = field(default=0, repr=False)
    _operator: int = field(default=0, repr=False)
    _operator_total: int = field(default=0, repr=False)
    events: List[Dict[str, float]] = field(default_factory=list, repr=False)

    @property
    def held(self) -> bool:
        """Whether the record just observed equals the one before it."""
        return self._run > 0

    @property
    def run(self) -> int:
        """Length of the current run of identical records, in frames."""
        return self._run

    @property
    def holds(self) -> int:
        """Holds seen so far that were reported to the caller."""
        return len(self.events)

    @property
    def operator_frames(self) -> int:
        """Frames on which this object asked for the A button, over its whole life.

        The per-hold counter resets when motion resumes, so a caller that read it would report the
        residual of the last hold rather than the total.
        """
        return int(self._operator_total)

    def observe(self, vector: np.ndarray) -> bool:
        """Update the run length with the state record just read. Returns `held`."""
        if self._previous is not None and np.all(np.abs(vector - self._previous) < 1e-9):
            self._run += 1
        else:
            if self._run >= self.trigger:
                self.events.append(
                    {"held_frames": self._run, "operator_frames": self._operator, "resumed": 1.0}
                )
            self._run = 0
            self._operator = 0
        self._previous = np.asarray(vector, dtype=np.float64).copy()
        return self.held

    def buttons(self, buttons: Dict[str, bool]) -> Dict[str, bool]:
        """Add the A pulse to a button dict while a hold is long enough to need one.

        Returns:
            The buttons to send. `give_up` is reported by raising nothing: once the limit is spent
            the pulse stops, and the caller decides whether to end the episode.
        """
        if self._run < self.trigger or self._operator >= self.limit:
            return buttons
        pulse = dict(buttons)
        pulse["A"] = (self._run % 8) < 4
        self._operator += 1
        self._operator_total += 1
        return pulse

    def exhausted(self) -> bool:
        """True when a hold has outlasted the A pulses, so the episode cannot continue."""
        return self._run >= self.trigger and self._operator >= self.limit
