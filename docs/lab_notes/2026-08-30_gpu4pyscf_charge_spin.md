# GPU4PySCF charge/spin parity実行記録

- 日付: 2026-08-30
- 対象: SiH3 / GeH3、neutral/cation/anion、singlet/doublet/triplet/quartetの12状態
- protocol: density-fitting条件付き候補、grid 5、VV10 grid 5、SCF `1e-10`
- state selection: `pending_scientific_review`

このmatrixは数値parity probeであり、各charge/spin状態をteacher dataへ採用する科学的承認ではない。

## 初回probe

- CPU PBS jobs: `1458`–`1469`、12/12 exit 0
- GPU Slurm job: `1796288`、`COMPLETED`、exit `0:0`、00:02:22
- GPU session: 12/12初回成功

暫定CPU–GPU gateは11/12だった。`geh3_neutral_quartet`だけ、両計算が収束し
`<S^2>`もquartet targetに近いにもかかわらず、次の大差を示した。

| Metric | Implicit-guess result |
|---|---:|
| absolute energy difference | 1.006059e-3 Eh |
| gradient component RMSE | 6.251602e-2 Eh/bohr |
| gradient component max | 1.740047e-1 Eh/bohr |

CPU energyは`-2078.326516262 Eh`、GPU energyは`-2078.327522320 Eh`で、異なる
open-shell SCF rootへの収束が疑われた。

## SCF-root診断

CPU PySCF object上でMINAO initial densityを一度だけ生成し、`to_gpu()`前後のCPU/GPU
両engineへ同じdensity matrixを明示的に渡した。

- CPU diagnostic PBS job: `1473`、exit 0、00:02:14
- GPU diagnostic Slurm job: `1796331`、exit 0、00:00:26

| Metric | Shared-MINAO result |
|---|---:|
| absolute energy difference | 4.349340e-8 Eh |
| gradient component RMSE | 2.392339e-6 Eh/bohr |
| gradient component max | 6.688361e-6 Eh/bohr |
| absolute S2 difference | 1.256955e-9 |
| MO occupation agreement | exact |

これにより、外れ値はGPU演算誤差ではなく、deviceごとに暗黙initial guessを生成したため
別SCF rootへ入ったことが原因と確認した。

## Runner修正と全12状態再検証

`run_pyscf.py`は、CPU PySCF objectで指定initial guess（default `minao`）からdensityを
生成し、それをCPU/GPU双方の`kernel(dm0=...)`へ明示的に渡すよう変更した。resultには
initial guessとdevice変換前生成のprovenanceを記録する。

- CPU PBS jobs: `1474`–`1485`、12/12 exit 0
- GPU Slurm job: `1796353`、`COMPLETED`、exit `0:0`、00:02:27
- suite: `charge_spin_density_fit_minao_probe_v1`
- CPU/GPU: 12/12初回成功
- 暫定CPU–GPU gate: 12/12 PASS

| Metric | Worst value | Case |
|---|---:|---|
| absolute energy difference | 5.941774e-8 Eh | SiH3 anion triplet |
| gradient component RMSE | 2.340439e-6 Eh/bohr | GeH3 neutral quartet |
| gradient component max | 6.543129e-6 Eh/bohr | GeH3 neutral quartet |

CPU wall time合計は1,463.84秒、GPUは97.97秒、aggregate speedupは14.942xだった。
最大spin contaminationはGeH3 cation tripletの`S2 deviation = 0.01562`で、CPU/GPUは
同じ値を再現した。状態自体の採否は別途科学reviewする。

## 成果物

```text
validation/orca_gpu4pyscf/suites/charge_spin_density_fit_probe_v1.json
validation/orca_gpu4pyscf/suites/charge_spin_density_fit_minao_probe_v1.json
validation/orca_gpu4pyscf/analysis/charge_spin/
/lustre/user140002/artifacts/gpu4pyscf-spin/1796288/session.json
/lustre/user140002/artifacts/gpu4pyscf-spin-minao/1796353/session.json
```
