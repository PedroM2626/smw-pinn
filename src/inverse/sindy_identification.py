r"""
sindy_identification.py
Sparse identification of nonlinear dynamics, the canonical algorithm (README 10.56).

Sections 10.43 and 10.44 identify the engine's law with two engines this project built: a
*posited* template, whose dictionary encodes what Section 4 says the physics is, and a free
genetic search over compositions. Both are named after their implementation, and the
published negatives about the gravity gate are recorded against them. Neither is the standard
estimator for this inverse problem.

That estimator is SINDy (Brunton, Proctor and Kutz, PNAS 2016): build a generic polynomial
dictionary, regress the time derivative against it with ridge, then sequentially threshold
the smallest coefficients and re-fit on the surviving support until the support stops
changing. It needs no structure from the documentation, no evolutionary budget and no
external runtime - only numpy - which is exactly what makes it the right control for a
negative that has been reported three times by two non-standard searches.

Two decisions are recorded rather than hidden, because they decide what the output means.

* **The derivative is a finite difference over one frame.** The engine is a discrete
  fixed-point accumulator, so `xdot` is the observed increment $s_{t+1} - s_t$ at one frame.
  Every coefficient is therefore in the units the WRAM already uses - sub-pixels per frame for
  a velocity increment, pixels per frame for a position increment - which lets the recovered
  numbers be compared with Section 4's constants directly, without a rescaling step.
* **Positions do not enter the dictionary.** $x$ and $y$ are absolute coordinates, the law is
  translation-invariant to first order, and including them would add a spurious linear term to
  every equation while inflating the library. They are still *predicted*, so the position
  equation is the one place where $\Delta x = v_x / 16$ can be discovered rather than assumed.
"""

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# The variables the library is built over: the two velocities, the four contact flags and the
# six action channels the engine reads as commands. Positions are excluded - see the module
# docstring. Names follow the 8D state and the 6-ch action layout of README section 3.
LIBRARY_INPUTS: Tuple[str, ...] = (
    "vx",
    "vy",
    "c_ground",
    "c_ceiling",
    "c_left",
    "c_right",
    "a_jump",
    "a_run",
    "a_up",
    "a_down",
    "a_left",
    "a_right",
)

# Channels that are already switches: their squares equal themselves, so those monomials are
# dropped from the dictionary instead of being handed to the thresholding step to untangle.
BINARY_INPUTS: Tuple[str, ...] = LIBRARY_INPUTS[2:]

# The recorded channels SINDy is asked to explain, in order: the two positions, the two
# velocities, then the four discrete flags (which are classification outputs, not dynamics).
TARGET_NAMES: Tuple[str, ...] = (
    "dx",
    "dy",
    "dvx",
    "dvy",
    "c_ground",
    "c_ceiling",
    "c_left",
    "c_right",
)
DYNAMIC_TARGETS: Tuple[str, ...] = ("dx", "dy", "dvx", "dvy")


def library_terms(max_degree: int = 2) -> List[Tuple[int, ...]]:
    """The exponent vectors of the candidate dictionary: a constant, then each degree.

    Binary channels are idempotent, so any monomial that squares one of them is identical to a
    lower-degree term and is left out - a dictionary that contains both is a dictionary whose
    columns are linearly dependent, and the ridge step would split one coefficient between them.
    """
    width = len(LIBRARY_INPUTS)
    terms: List[Tuple[int, ...]] = [tuple(0 for _ in range(width))]
    for degree in range(1, max_degree + 1):
        for combo in itertools.combinations_with_replacement(range(width), degree):
            power = tuple(combo.count(index) for index in range(width))
            if any(
                power[index] > 1
                for index, name in enumerate(LIBRARY_INPUTS)
                if name in BINARY_INPUTS
            ):
                continue
            terms.append(power)
    return terms


def term_names(terms: Sequence[Tuple[int, ...]]) -> List[str]:
    """Readable names for the dictionary columns."""
    out: List[str] = []
    for power in terms:
        factors = [
            name if count == 1 else f"{name}^{count}"
            for name, count in zip(LIBRARY_INPUTS, power)
            for _ in [0]
            if count
        ]
        out.append("1" if not factors else " ".join(factors))
    return out


def evaluate_library(inputs: np.ndarray, terms: Sequence[Tuple[int, ...]]) -> np.ndarray:
    """Build the design matrix $\\Theta(\\Xi)$ from $[N, \\text{len(LIBRARY_INPUTS]}]$ channels."""
    inputs = np.asarray(inputs, dtype=np.float64)
    columns = []
    for power in terms:
        term = np.ones(inputs.shape[0], dtype=np.float64)
        for index, exponent in enumerate(power):
            if exponent:
                term = term * np.power(inputs[:, index], exponent)
        columns.append(term)
    return np.column_stack(columns)


def prune_columns(
    theta: np.ndarray, names: Sequence[str]
) -> Tuple[np.ndarray, List[str], List[str], List[int]]:
    """Drop constant columns and exact duplicates, and report what was dropped.

    Returns the pruned matrix, its names, the dropped descriptions and the surviving column
    indices - the indices because the *same* selection has to be applied to the held-out
    design, whose own duplicates need not coincide with the training set's.

    A polynomial dictionary over discrete switches is rank-deficient by construction: on this
    data `c_right` implies `c_ground`, so the product `c_ground c_right` is a copy of
    `c_right`, and ridge splits one coefficient between the two - which is not a wrong fit but
    is an unreadable law. The pruning is reported rather than silent, because *which* products
    collapse is a statement about the level geometry: a right-contact flag that only ever
    occurs on the floor is a fact about Mario's frame, not about the regression.
    """
    theta = np.asarray(theta, dtype=np.float64)
    kept: List[int] = []
    dropped: List[str] = []
    seen: Dict[bytes, str] = {}
    for index in range(theta.shape[1]):
        column = theta[:, index]
        fingerprint = column.tobytes()
        if index != 0 and column.std() == 0.0:
            dropped.append(f"{names[index]} (constant)")
            continue
        if fingerprint in seen:
            dropped.append(f"{names[index]} (duplicate of {seen[fingerprint]})")
            continue
        seen[fingerprint] = names[index]
        kept.append(index)
    return theta[:, kept], [names[i] for i in kept], dropped, kept


def library_inputs_of(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Select the channels SINDy may see, in `LIBRARY_INPUTS` order."""
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    velocities = states[:, 2:4]
    contacts = states[:, 4:8]
    commanded = actions[:, :6]
    return np.column_stack([velocities, contacts, commanded])


def increments(states: np.ndarray, next_states: np.ndarray) -> np.ndarray:
    """The discrete-time derivative: the observed increment of every recorded channel."""
    return np.asarray(next_states, dtype=np.float64) - np.asarray(states, dtype=np.float64)


@dataclass
class SindyFit:
    """One sparse identification: the coefficients, the surviving support, and its error."""

    names: List[str]
    coefficients: np.ndarray
    support: List[List[int]]
    alpha: float
    ridge_lambda: float
    iterations: int
    targets: List[str] = field(default_factory=list)

    def equations(self) -> Dict[str, List[Tuple[str, float]]]:
        """Each discovered law as the terms that survived thresholding, with its coefficient."""
        return {
            name: [(self.names[i], float(self.coefficients[i, j])) for i in self.support[j]]
            for j, name in enumerate(self.targets)
        }

    def sparsity(self) -> Dict[str, int]:
        return {name: len(self.support[j]) for j, name in enumerate(self.targets)}

    def coefficient_of(self, target: str, term: str) -> float:
        """The fitted coefficient of one named term in one equation, or 0.0 if it was cut."""
        j = self.targets.index(target)
        for i in self.support[j]:
            if self.names[i] == term:
                return float(self.coefficients[i, j])
        return 0.0

    def relative_error(
        self, theta: np.ndarray, xdot: np.ndarray, references: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Per-target relative $\\ell_2$ error of the discovered law, on rows it never saw."""
        prediction = theta @ self.coefficients
        out: Dict[str, float] = {}
        for j, name in enumerate(self.targets):
            truth = xdot[:, j]
            scale = float(np.linalg.norm(truth))
            out[name] = (
                float(np.linalg.norm(prediction[:, j] - truth) / scale) if scale else float("nan")
            )
        pooled = float(np.linalg.norm(xdot))
        out["pooled"] = (
            float(np.linalg.norm(prediction - xdot) / pooled) if pooled else float("nan")
        )
        return out

    def variance_explained(
        self, theta: np.ndarray, xdot: np.ndarray, mean: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """$R^2$ per target, the statistic the template engine of 10.43 reports.

        Reported next to the $\\ell_2$ relative error because the two answer different
        questions when the target has near-zero mean: a velocity increment sits at zero by
        construction, so a constant-fit explains nothing and the two numbers nearly coincide,
        while a position increment has a large mean and the relative error alone hides how
        little variance the law actually captures.
        """
        prediction = theta @ self.coefficients
        centre = xdot.mean(axis=0) if mean is None else mean
        out: Dict[str, float] = {}
        for j, name in enumerate(self.targets):
            truth = xdot[:, j]
            total = float(np.sum((truth - centre[j]) ** 2))
            out[name] = (
                float(1.0 - np.sum((prediction[:, j] - truth) ** 2) / total)
                if total
                else float("nan")
            )
        total_pooled = float(np.sum((xdot - centre) ** 2))
        out["pooled"] = (
            float(1.0 - np.sum((prediction - xdot) ** 2) / total_pooled)
            if total_pooled
            else float("nan")
        )
        return out


def _huber_weights(residual: np.ndarray, delta: float = 1.345) -> np.ndarray:
    """Huber weights from a median-absolute-deviation scale, so the constant and the
    discontinuous frames stop dragging the coefficients."""
    scale = float(np.median(np.abs(residual - np.median(residual)))) or float(residual.std())
    if scale <= 0.0:
        return np.ones_like(residual)
    cutoff = delta * scale
    return np.clip(cutoff / np.maximum(np.abs(residual), cutoff), 0.0, 1.0)


def sindy(
    xdot: np.ndarray,
    theta: np.ndarray,
    names: Sequence[str],
    alpha: float = 0.05,
    ridge_lambda: float = 1e-3,
    max_iterations: int = 20,
    targets: Optional[Sequence[str]] = None,
    robust: bool = False,
) -> SindyFit:
    r"""Sequential thresholding with ridge re-fit (Brunton et al., algorithm 1).

    Args:
        xdot: $[N, k]$ the increments, i.e. the discrete-time derivatives.
        theta: $[N, p]$ the library evaluated on the inputs.
        names: the readable name of every library column.
        alpha: the coefficient magnitude below which a term is dropped, in the WRAM's own
            units, so it is comparable with the constants the identification is trying to
            recover - which is why it is reported rather than hidden behind an information
            criterion.
        ridge_lambda: the ridge penalty on every term except the constant, because a constant
            offset *is* the gravity law and shrinking it biases the one thing being measured.
        max_iterations: how many threshold / re-fit passes before giving up.
        targets: names for the columns of `xdot`.
        robust: re-weight each re-fit by Huber weights. This is not the canonical algorithm -
            it is the control that says whether a recovered constant is missing because the
            law is wrong or because a handful of frames where the engine stops, wraps or
            respawns dominates a least-squares residual.
    """
    xdot = np.asarray(xdot, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    n_targets = xdot.shape[1]
    target_names = list(targets) if targets else [f"channel_{j}" for j in range(n_targets)]

    standardising = theta.std(axis=0)
    scale = np.where(
        standardising > 0.0, 1.0 / np.where(standardising > 0.0, standardising, 1.0), 1.0
    )
    design = theta * scale

    penalty = np.eye(design.shape[1]) * ridge_lambda
    penalty[0, 0] = 0.0

    support: List[List[int]] = [list(range(design.shape[1])) for _ in range(n_targets)]
    coefficients = np.zeros((design.shape[1], n_targets), dtype=np.float64)
    used = 0
    for iteration in range(1, max_iterations + 1):
        used = iteration
        previous = [list(columns) for columns in support]
        for j in range(n_targets):
            columns = np.array(support[j], dtype=int)
            sub = design[:, columns]
            truth = xdot[:, j]
            base_gram = penalty[np.ix_(columns, columns)]
            rhs = sub.T @ truth
            weights = None
            solution = np.zeros(columns.size, dtype=np.float64)
            for _ in range(6 if robust else 1):
                gram = sub.T @ sub if weights is None else sub.T @ (sub * weights[:, None])
                gram = gram + base_gram
                right = rhs if weights is None else sub.T @ (truth * weights)
                solution = np.linalg.lstsq(gram, right, rcond=None)[0]
                if not robust:
                    break
                weights = _huber_weights(truth - sub @ solution)
            # The threshold is applied in the WRAM's own units, not in the standardised ones
            # the conditioning needs: alpha has to be comparable with the constants being
            # recovered, so 0.05 here means "below five hundredths of a sub-pixel per frame".
            original = np.zeros(design.shape[1], dtype=np.float64)
            original[columns] = solution * scale[columns]
            coefficients[:, j] = original
            keep = [int(i) for i in columns if abs(original[i]) > alpha]
            support[j] = keep if keep else [int(columns[int(np.argmax(np.abs(original[columns])))])]
        if all(previous[j] == support[j] for j in range(n_targets)):
            break

    return SindyFit(
        names=list(names),
        coefficients=coefficients,
        support=support,
        alpha=alpha,
        ridge_lambda=ridge_lambda,
        iterations=used,
        targets=target_names,
    )
