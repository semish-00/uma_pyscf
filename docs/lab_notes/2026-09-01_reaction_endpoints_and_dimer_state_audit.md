# Reaction endpoint preparation and Si/Ge dimer state audit

- 実施日: 2026-09-01
- 実装commit: `c991fc8`
- 実行環境: SoftBank GPU cluster (`sb-gpu`)
- 結論: 8 reactionのendpointをpath計算へ進められる形で準備できた。Si2H3、
  Si2H5、Ge2H3、Ge2H5は、今回比較した全geometryでdoubletがquartetより低く、
  `<S^2>`も良好だった。ただしwavefunction stability/occupationを含むsentinel監査前なので、
  state registryは`pending_scientific_review`のまま維持する。

## 1. Endpoint preparation

| item | result |
|---|---|
| Slurm job | `1802498` (`COMPLETED`, exit `0:0`) |
| elapsed | `00:02:57` |
| reaction families | 8 |
| endpoint fragments | 24 |
| converged fragments | 24 / 24 |
| final criterion | `fmax <= 0.01 eV/angstrom` |
| worst final max force | `0.0085921809 eV/angstrom` (Ge2H3 product fragment) |
| artifact root | `/lustre/user140002/runs/calibration/reaction_endpoints_v1/1802498` |

生成物はfragmentごとにbase UMA (`uma-s-1p2`, OMol, CUDA)で最適化し、固定した
atom mappingとfragment partitionを保って再配置した。反応物・生成物の元素順は8 family
すべてで一致し、意図しない初期衝突はなかった。

- endpoint summary SHA-256:
  `df579769d5cbe748937bfc2b9346e197bf17d5021e9f2ecb2d53cdb2b3a9b23a`
- generated state-audit sampling config SHA-256:
  `4ed614473cee712f2946cba9a06d583114f37b413181b63ccfa81e47a709fd9d`

## 2. Multi-geometry state audit

各dimer組成についてM-M距離を0.95、1.00、1.05倍した3 geometryを作り、neutral
doublet/quartetを同じGPU4PySCF protocolで比較した。これはC0-S engineering evidenceであり、
C0 180件のteacher-label数には加算しない。

| item | result |
|---|---|
| Slurm job | `1802504` (`COMPLETED`, exit `0:0`) |
| elapsed | `00:07:34` |
| candidate / geometry QC | 24 accepted / 0 rejected |
| GPU4PySCF labels | 24 completed / 0 failed / 0 blocked |
| label QC | 24 accepted / 0 rejected |
| SCF route | 24 / 24 `primary_density_fit`; direct fallbackなし |
| artifact root | `/lustre/user140002/runs/calibration/si_ge_dimer_state_audit_v1/1802504` |

| source | doublet lower | minimum E(quartet)-E(doublet) | max doublet S2 deviation | max quartet S2 deviation |
|---|---:|---:|---:|---:|
| Si2H3 | 3 / 3 | 2.469977 eV | 0.002500 | 0.005404 |
| Si2H5 | 3 / 3 | 6.448457 eV | 0.002058 | 0.003369 |
| Ge2H3 | 3 / 3 | 2.575527 eV | 0.002602 | 0.008487 |
| Ge2H5 | 3 / 3 | 5.967304 eV | 0.002070 | 0.004235 |

全12 geometry pairが、事前に定めたreview条件（doubletが低い、両stateが収束、
両stateの`|S2 - S2_ideal| <= 0.05`）を満たした。したがって4組成ともneutral doubletを
reaction-path候補として強く支持する。

- candidate manifest SHA-256:
  `5d534f6974f298d2562eca1de02f33ba8c1617f8ef1351feb0dd5c4e4e867ea2`
- label summary SHA-256:
  `a243c0e9a748f08263540a8844d720e20f89f8c1d323de2e325a55aaaed469d4`
- QC report SHA-256:
  `4b884477dae63f60dcb1339d330903cb9577bf4b85078e8edbf11c122029396f`
- state-audit summary SHA-256:
  `5c3b72888622b43e07eb30fbb0c3dbb70422209a024d8d35d04755404d3c0f14`

## 3. Decision and next gate

今回の密度フィッティング計算だけではwavefunction stabilityとoccupationの監査を完了して
いないため、stateを自動承認しない。次の順序とする。

1. C0 independentのsinglet 4 pathsをbase UMAのCI-NEB/stringで生成する。
2. endpoint、climbing image近傍、中間領域からmass-weighted arc lengthで9件/pathを固定する。
3. 固定した36件をGPU4PySCFでlabel/QCし、C0を81/180から117/180へ進める。
4. Si2H3、Si2H5、Ge2H3、Ge2H5は代表geometryでCPU/direct sentinelと
   wavefunction stability/occupationを監査し、doublet承認可否を記録する。
5. 承認後にcoverage extension 4 pathsを同様に生成・labelする。

この順序なら、state承認待ちでC0 independent pathを止めず、open-shell labelを
production datasetへ早まって混ぜることもない。
