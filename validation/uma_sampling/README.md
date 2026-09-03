# Base-UMA candidate generation

## C0 moderate-temperature MD

The C0 runner uses six neutral-singlet parent compositions that are absent from
engineering-50 and the first 117 C0 records. It relaxes each approximate parent
with base UMA, then runs 48 short NVT Langevin trajectories: 300/600/900/1200 K,
two seeds, and a 0.5 fs timestep. The first 200 steps are excluded as
thermalization; 800 production steps are saved every 20 steps.

These trajectories generate finite-temperature candidates. They are not
claimed to be equilibrated trajectories or physical-property simulations. Five
mass-weighted-arc-length frames per trajectory form the blind pool. The
portfolio caps retain at most five records per parent and one per trajectory,
freezing 27 records without using UMA scores or teacher labels.

Run on the SoftBank GPU host with:

```bash
sbatch scripts/slurm/run_c0_moderate_md_softbank_slurm.sh
```
