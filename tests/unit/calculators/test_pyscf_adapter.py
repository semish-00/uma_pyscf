"""Resource-tier behavior that can be tested without PySCF or CUDA."""

from __future__ import annotations

import unittest

from uma_pyscf.calculators.model import CalculationFailure
from uma_pyscf.calculators.pyscf_adapter import _configure_pyscf_resources


class FakeLib:
    def __init__(self, *, mismatch: bool = False) -> None:
        self.requested: int | None = None
        self.mismatch = mismatch

    def num_threads(self, value: int) -> int:
        self.requested = value
        return value - 1 if self.mismatch else value


class FakePyscf:
    def __init__(self, *, mismatch: bool = False) -> None:
        self.lib = FakeLib(mismatch=mismatch)


class ResourceConfigurationTests(unittest.TestCase):
    def test_applies_threads_and_returns_memory_limit(self) -> None:
        pyscf = FakePyscf()
        actual = _configure_pyscf_resources(
            pyscf,
            {"ncpus": 16, "max_memory_mb": 48000},
        )
        self.assertEqual(actual, (16, 48000))
        self.assertEqual(pyscf.lib.requested, 16)

    def test_refuses_a_runtime_that_does_not_honor_the_thread_tier(self) -> None:
        with self.assertRaises(CalculationFailure) as caught:
            _configure_pyscf_resources(
                FakePyscf(mismatch=True),
                {"ncpus": 8, "max_memory_mb": 24000},
            )
        self.assertEqual(caught.exception.category, "runtime_environment")
        self.assertIn("resource tier requires 8", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
