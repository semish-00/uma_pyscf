# Base-UMA candidate generation

## C0 moderate-temperature MD

The C0 runner uses six neutral-singlet parent compositions that are absent from
engineering-50 and the first 117 C0 records. It relaxes each approximate parent
with base UMA, then runs 48 short NVT Langevin trajectories: 180/300/450/650 K,
two seeds, and a 0.25 fs timestep. The lower thermostat targets compensate for
effective heating measured in the 100/300/650 K preflight and aim at observed
means of roughly 300–1100 K. The first 400 steps are excluded as thermalization;
1600 production steps are saved every 20 steps. A trajectory is rejected if its
saved mean temperature falls outside 0.5–2.5 times its target.

These trajectories generate finite-temperature candidates. They are not
claimed to be equilibrated trajectories or physical-property simulations. Five
mass-weighted-arc-length frames per trajectory form the blind pool. The
portfolio caps retain at most five records per parent and one per trajectory,
freezing 27 records without using UMA scores or teacher labels.

Run on the SoftBank GPU host with:

```bash
sbatch scripts/slurm/run_c0_moderate_md_softbank_slurm.sh
```
