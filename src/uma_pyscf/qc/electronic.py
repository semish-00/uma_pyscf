"""Electronic-structure QC: is this label a converged, uncontaminated result?

Four questions are asked of every label record, each of them a pure function of
the record and the ``electronic`` section of a QC config:

* Did the SCF converge? The schema layer already refuses to *import* an
  unconverged cross-code result, but a record can also be written by hand or by
  a future calculator adapter, so the verdict is re-established here rather than
  assumed.
* Is the open-shell wavefunction close enough to the spin state it claims?
  ``|<S^2> - S(S+1)|`` is the standard spin-contamination measure, and a doublet
  that came out at 0.9 is not the doublet the record says it is.
* Is any gradient component absurdly large? Is the whole gradient matrix?
  These two are sanity ceilings, not science: they catch a collision geometry or
  a broken calculation, and they are deliberately far above any force a model is
  expected to learn.

Each function returns a check dict -- ``name``, ``passed``, ``observed``,
``threshold`` -- rather than raising, because a failing check is a recorded
verdict about a record and not an error in the run. What *is* an error is a
check that cannot be evaluated: a section without the threshold it needs raises
through the helpers in :mod:`uma_pyscf.qc.config`, so a missing condition can
never be mistaken for a satisfied one. An unknown key in the section is refused
earlier, when the config is loaded.

**The singlet case is deliberately skipped, not silently passed.** A restricted
singlet has ``<S^2> = 0`` by construction and the check would be vacuous, so it
is reported with ``passed`` true and ``observed`` null -- present in the output,
visibly not evaluated. That is not the same as saying a singlet is beyond
suspicion: a broken-symmetry UKS singlet can be heavily contaminated, and
detecting it needs the reference ``<S^2>`` of the guess rather than of the
multiplicity. That check belongs to a later, Gate-informed revision of this
module, and inventing a tolerance for it now would put an unreviewed number in
front of a real physical question.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import fsum, sqrt
from typing import Any

from ..schemas.label_record import LabelRecord
from .config import flag, positive_threshold

__all__ = [
    "CHECK_CONVERGED",
    "CHECK_GRADIENT_MAX_COMPONENT",
    "CHECK_GRADIENT_NORM",
    "CHECK_S2_DEVIATION",
    "ELECTRONIC_CHECK_NAMES",
    "check_converged",
    "check_gradient_max_component",
    "check_gradient_norm",
    "check_s2_deviation",
    "electronic_checks",
    "gradient_max_abs",
    "gradient_norm",
]

CHECK_CONVERGED = "converged"
CHECK_S2_DEVIATION = "s2_deviation"
CHECK_GRADIENT_MAX_COMPONENT = "gradient_max_component"
CHECK_GRADIENT_NORM = "gradient_norm"

#: The electronic checks, in the order :func:`electronic_checks` runs them.
ELECTRONIC_CHECK_NAMES: tuple[str, ...] = (
    CHECK_CONVERGED,
    CHECK_S2_DEVIATION,
    CHECK_GRADIENT_MAX_COMPONENT,
    CHECK_GRADIENT_NORM,
)

_SECTION = "electronic"

#: What an open-shell record's ``observed`` says when it reports no ``<S^2>``.
MISSING = "missing"


def _check(name: str, passed: bool, observed: Any, threshold: Any) -> dict[str, Any]:
    """Return one check result in the shape the QC report stores."""
    return {"name": name, "passed": passed, "observed": observed, "threshold": threshold}


def gradient_max_abs(record: LabelRecord) -> float:
    """Return the largest absolute gradient component of a record, hartree/bohr."""
    return max(
        abs(component) for row in record.results.gradient_hartree_per_bohr for component in row
    )


def gradient_norm(record: LabelRecord) -> float:
    """Return the Frobenius norm of a record's gradient matrix, hartree/bohr.

    The sum is accumulated with :func:`math.fsum`, which is exactly rounded, so
    the norm does not depend on the order the atoms happen to be listed in.
    """
    return sqrt(
        fsum(
            component * component
            for row in record.results.gradient_hartree_per_bohr
            for component in row
        )
    )


def check_converged(record: LabelRecord, section: Mapping[str, Any]) -> dict[str, Any]:
    """Check that the SCF reported convergence.

    ``require_converged`` is required to be true by a v1 config, so in practice
    this is "did it converge". The flag is still honoured rather than ignored:
    the check reports what it was asked to require, and the report says so.
    """
    required = flag(section, "require_converged", _SECTION)
    converged = record.results.converged
    return _check(CHECK_CONVERGED, converged or not required, converged, required)


def check_s2_deviation(record: LabelRecord, section: Mapping[str, Any]) -> dict[str, Any]:
    """Check open-shell spin contamination against the configured tolerance.

    A record with multiplicity 1 is skipped (``passed`` true, ``observed``
    null); see the module docstring for why the broken-symmetry singlet case is
    a later addition rather than an omission. An open-shell record that reports
    no deviation is reported with ``observed`` ``"missing"`` and fails unless
    the config explicitly allows it: a spin state nobody measured is not a spin
    state anybody verified.
    """
    threshold = positive_threshold(section, "s2_max_abs_deviation", _SECTION)
    required = flag(section, "require_s2_for_open_shell", _SECTION)
    if record.state.multiplicity <= 1:
        return _check(CHECK_S2_DEVIATION, True, None, threshold)
    deviation = record.results.s2_deviation
    if deviation is None:
        return _check(CHECK_S2_DEVIATION, not required, MISSING, threshold)
    observed = abs(deviation)
    return _check(CHECK_S2_DEVIATION, observed <= threshold, observed, threshold)


def check_gradient_max_component(
    record: LabelRecord, section: Mapping[str, Any]
) -> dict[str, Any]:
    """Check the largest absolute gradient component against its ceiling."""
    threshold = positive_threshold(section, "gradient_max_abs_hartree_per_bohr", _SECTION)
    observed = gradient_max_abs(record)
    return _check(CHECK_GRADIENT_MAX_COMPONENT, observed <= threshold, observed, threshold)


def check_gradient_norm(record: LabelRecord, section: Mapping[str, Any]) -> dict[str, Any]:
    """Check the Frobenius norm of the gradient matrix against its ceiling."""
    threshold = positive_threshold(section, "gradient_norm_max_hartree_per_bohr", _SECTION)
    observed = gradient_norm(record)
    return _check(CHECK_GRADIENT_NORM, observed <= threshold, observed, threshold)


def electronic_checks(
    record: LabelRecord, section: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Run every electronic check on one record, in :data:`ELECTRONIC_CHECK_NAMES` order.

    All four run even after one fails. A QC report is read to decide what to fix
    about a batch of labels, and "it did not converge" is far less useful than
    "it did not converge *and* its gradient is enormous"; stopping at the first
    failure would hide the second fact.
    """
    return (
        check_converged(record, section),
        check_s2_deviation(record, section),
        check_gradient_max_component(record, section),
        check_gradient_norm(record, section),
    )
