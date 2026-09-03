# C0 independent reaction paths and GPU4PySCF labels

- 実施日: 2026-09-01–2026-09-03
- 結果: 4 reaction / 36構造の経路サンプリング、GPU4PySCF label、QC完了
- C0進捗: 81/180 -> 117/180
- 実行commit: `e9f53a49832e4bb6cc0134ae59f435f47e7612c4`

## 1. Scope

engineering-50とparent/reactionを共有しないC0 independent 4 familyを対象にした。

| reaction_id | reaction | state | selected |
|---|---|---|---:|
| `sihcl3_to_sicl2_hcl` | SiHCl3 -> SiCl2 + HCl | neutral singlet | 9 |
| `gehcl3_to_gecl2_hcl` | GeHCl3 -> GeCl2 + HCl | neutral singlet | 9 |
| `sih3cl_to_sihcl_h2` | SiH3Cl -> SiHCl + H2 | neutral singlet | 9 |
| `geh3cl_to_gehcl_h2` | GeH3Cl -> GeHCl + H2 | neutral singlet | 9 |

## 2. Sampling method and interpretation

`si_ge_reaction_endpoints_v1`で`fmax <= 0.01 eV/angstrom`へ収束済みの端点と
固定済みのatom mappingを使った。17 imagesをIDPP補間し、base
UMA-S-1.2 / OMolでASE NEBを以下の2段階で実行した。

1. `aseneb`、spring `0.10 eV/angstrom^2`、FIRE `maxstep=0.08 angstrom`で
   `fmax <= 0.30 eV/angstrom`までcoarse pathを作る。
2. climbing imageを有効にし、`maxstep=0.03 angstrom`で100 stepの固定
   refinementを行う。
3. endpoint、最高energy imageとその前後、残りをmass-weighted arc lengthで
   選んで9構造/pathに固定する。

これは教師候補を作るための**reaction-path sampling**であり、精密なTSや
活性化障壁の計算ではない。解離経路にはclimbing imageが収束すべき内部
停留点を持たない場合があるため、`all_paths_converged=false`は意図した記録で
ある。代わりにcoarse収束、非有限値なし、隣接image重複なし、climbing近傍
保持、最終NEB `fmax <= 5.0 eV/angstrom`、geometry QCをfail-closedで評価した。

`improvedtangent`は反応物側の平坦なenergy列でzero tangentを作り、NaNを生じた。
そのため`aseneb`に変更し、有限projected force guardを各stepに置いた。
調整中の中断runは診断用であり、candidate manifestには入れていない。

## 3. Path result

Slurm job `1809301`は`COMPLETED` / exit `0:0`、elapsed `01:07:38`だった。

| reaction_id | coarse steps | fixed climb steps | final NEB fmax (eV/angstrom) | accepted |
|---|---:|---:|---:|---|
| `sihcl3_to_sicl2_hcl` | 75 | 100 | 3.094720 | yes |
| `gehcl3_to_gecl2_hcl` | 94 | 100 | 0.627459 | yes |
| `sih3cl_to_sihcl_h2` | 72 | 100 | 0.441216 | yes |
| `geh3cl_to_gehcl_h2` | 74 | 100 | 1.204493 | yes |

- sampling acceptance: 4/4
- selected candidate: 36件、record ID 36 unique
- trajectory import geometry QC: 36 accepted / 0 rejected
- 隣接imageの最小Cartesian displacement: `0.282540 angstrom`

## 4. GPU4PySCF label and QC

Slurm job `1824441`で固定manifestを
`omol_wb97mv_tzvpd_v1` (`wB97M-V/def2-TZVPD`)でlabelし、意図的な解離を許す
`omol_wb97mv_tzvpd_dissociation_qc_v1`でQCした。jobは`COMPLETED` /
exit `0:0`、elapsed `00:08:20`だった。

| item | result |
|---|---|
| label | 36 completed / 0 failed / 0 blocked / 0 skipped |
| SCF route | 36/36 `primary_density_fit`; direct fallback 0 |
| label QC | 36 accepted / 0 rejected |
| composition | 4 composition x 9 accepted |
| failed QC checks | 0 |
| max gradient component | `0.0849266 Hartree/Bohr` |

全36件はneutral singletであり、non-default state registryを必要としない。
QC完了は学習poolへの収録可能性を示すが、C0のrelease approvalではない。

## 5. Artifact identity

| artifact | SHA-256 |
|---|---|
| path summary | `dc95426d14a2bbeb41bd48d9e71c8251e6831d94bbcf5ac4e9c96dff23ffd0cd` |
| candidate manifest | `c4f3b61922ed94ae4a7fd695e893e1074feca10de2c7795eefdc94d0da2b1538` |
| trajectory geometry QC | `3c733807a548a9594f84de3571e94ca11626dcdc3a8103d656925577657544cd` |
| label summary | `06f11c40d1ffbad48aa4cf7228d56f014dd2566f39bb6ed693f723863ac6c5cc` |
| attempt ledger | `6bcdb2c28f358ee004d8d8075829b01a6b2fd5a03267438bf7f4bc6766371a64` |
| label QC report | `aaa3e933bcef5ebafa2bf6f57c4a7669af58111591fc15daa8fd495843ac4462` |

Remote artifact roots:

- `/lustre/user140002/runs/calibration/c0_independent_ci_neb_v1/1809301`
- `/lustre/user140002/runs/label/c0_independent_reaction_paths_36_v1/1824441`

## 6. Next actions

1. C0本体はindependent neutral-singlet parentを固定し、moderate-temperature MD
   27件を次のquotaとする。
2. strong-distortion/high-temperature MD 18件とcurated graph edit 18件は別manifestにする。
3. C0-SはSi2H3/Si2H5/Ge2H3/Ge2H5のCPU/direct stability sentinelを行い、
   neutral doubletの承認後にcoverage-extension 4 pathをlabelする。
4. C0 180/180とCPU/GPU sentinelが揃うまでreleaseは閉じたままとする。
