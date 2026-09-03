# C0 sampling and multi-GPU label dispatch — 2026-09-03

## Outcome

- C0 baseline before this run: 117/180 accepted GPU4PySCF labels.
- Base-UMA moderate-temperature candidate generation is running as Slurm job
  `1825031`; its target is a blind 27-record portfolio.
- A controlled strong-distortion pool generated 18/18 geometry-QC-passing
  candidates across six neutral-singlet parents.
- The strong-distortion preparation job `1825041` completed and froze the
  candidate and portfolio manifests with checksums.
- GPU4PySCF label array `1825042_[0-7]` completed all eight deterministic,
  cost-balanced execution shards. All 18 DFT calculations converged; provisional
  QC accepted 17 and retained one boundary rejection. This raises the
  conservative C0 accepted count from 117 to 134/180.

## Scientific handling

The distortion factors are 0.70, 0.78, and 1.22 on one terminal M-H or M-Cl
bond per parent. Candidate generation rejects overlap/fragment errors but does
not reject large forces or high energies. Teacher convergence and electronic
QC decide whether a geometry becomes `valid_high_energy`, is quarantined as
`electronic_ambiguous`, or is rejected as `hard_invalid`.

The retained boundary case is
`c0_strong_distortion_pool_v1_si2h5cl_bond02_x0p7`. Its maximum gradient is
1.0066986 hartree/bohr against the provisional 1.0 threshold, while the SCF,
protocol, geometry, and gradient-norm checks pass. It is not silently promoted
or discarded: the raw converged label remains available for the high-energy
boundary audit, and the current accepted count excludes it.

The independent-reaction comparison shows that the 10,000-step engineering
model is overfit and is not a valid acquisition model. Holdout-guided follow-up
will use base-UMA residuals only on diagnostic C0/P0 groups, preserve complete
reaction/parent grouping, and exclude the known ambiguous SiHCl3 detached-H
point until a separate state/stability audit resolves it.

## Next actions

1. If at least 27 moderate-MD trajectories complete, freeze the 27-record
   portfolio and dispatch it through the same multi-GPU label array.
2. Fill or adjudicate the one-record strong-distortion boundary gap without
   weakening the global QC threshold merely to pass one record.
3. Build the curated graph-edit/unknown-reaction 18-record C0 tranche.
4. After C0 reaches 180/180, freeze a non-overlapping T0 before model-aware
   acquisition or retraining.
