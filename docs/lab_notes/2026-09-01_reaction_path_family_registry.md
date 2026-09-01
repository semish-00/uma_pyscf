# C0 reaction-path family registry

- 実施日: 2026-09-01
- 状態: endpoint生成とC0-S GPU audit完了。path計算は未実施
- 詳細計画:
  [Teacher-data sampling, calibration, and oracle-pool plan](../plans/05_teacher_data_sampling_and_calibration.md#42-reaction-paths)

## 1. 目的

C0の次sourceであるreaction pathを、既存engineering-50と独立なcalibration用と、
指定反応を確実に収録するcoverage extensionに分ける。いずれも各1反応9件を
開始値とし、計8 family / 72 GPU4PySCF labelsを予定する。

## 2. C0 independent path tranche: 36 labels

| reaction_id | reaction | state | labels |
|---|---|---|---:|
| `sihcl3_to_sicl2_hcl` | SiHCl3 -> SiCl2 + HCl | neutral singlet | 9 |
| `gehcl3_to_gecl2_hcl` | GeHCl3 -> GeCl2 + HCl | neutral singlet | 9 |
| `sih3cl_to_sihcl_h2` | SiH3Cl -> SiHCl + H2 | neutral singlet | 9 |
| `geh3cl_to_gehcl_h2` | GeH3Cl -> GeHCl + H2 | neutral singlet | 9 |

この36件だけをC0 180件の`interpolation_neb_irc`に数える。

## 3. Requested reaction-coverage extension: 36 labels

| reaction_id | reaction | state | labels | gate |
|---|---|---|---:|---|
| `sih4_to_sih2_h2` | SiH4 -> SiH2 + H2 | neutral singlet | 9 | engineering overlapを明記 |
| `geh4_to_geh2_h2` | GeH4 -> GeH2 + H2 | neutral singlet | 9 | engineering overlapを明記 |
| `si2h5_to_si2h3_h2` | Si2H5 -> Si2H3 + H2 | neutral doublet | 9 | C0-S state approval |
| `ge2h5_to_ge2h3_h2` | Ge2H5 -> Ge2H3 + H2 | neutral doublet | 9 | C0-S state approval |

SiH4とGeH4はengineering-50のparentである。指定反応としてteacher labelは作るが、
C0の取得法calibrationやT0 fixed testの独立な根拠には使わない。既存PFP
`SiH4 -> SiH2 + H2` trajectoryはendpoint/path generatorのengineering fixtureに限り、
最終の9件は新規manifestとprovenanceで固定する。

## 4. Si2H3 / Ge2H3 state policy

Si2H3とGe2H3は奇数電子系である。第一候補は、文献で報告されたdisilenyl/digermenyl
radicalに対応する中性doubletとする。各endpointの複数geometryでdoublet/quartetを
C0-S比較し、energy ordering、`<S^2>`、SCF occupation、wavefunction stabilityを監査する。

最初のreaction coordinateは次とする。

- Si2H5(doublet) -> Si2H3(doublet) + H2(singlet)
- Ge2H5(doublet) -> Ge2H3(doublet) + H2(singlet)

これは反応全体をdoublet PESで追える。一方、Si2H4 -> Si2H3 + Hのような
radical-fragment解離は生成物側のspin couplingを別途扱う必要があるため、初回の
72件からは除外する。

## 5. 実行順

1. 8 reactionのendpoint構造と原子mappingを固定する。
2. singlet 6 reactionのendpointを最適化する。
3. Si2H3/Ge2H3系はC0-Sを先行し、doubletを承認する。
4. CI-NEB/string、必要な反応だけIRCを実行する。
5. endpoint、TS近傍、中間領域を含む9件/reactionをarc-lengthで固定する。
6. geometry QC後、C0 36件とcoverage extension 36件を別manifestでlabelする。

## 6. 実行状況

2026-09-01に8 reactionのendpointをbase UMAで準備し、全24 fragmentが
`fmax <= 0.01 eV/angstrom`へ収束した。続いてSi2H3、Si2H5、Ge2H3、Ge2H5の
doublet/quartetを3 geometryずつGPU4PySCFで比較し、24/24 labelとQCが完了した。
全12 pairでdoubletが2.47 eV以上低く、S2 deviationも0.009未満だった。

結果、artifact hash、承認を保留する理由は
[Reaction endpoint preparation and Si/Ge dimer state audit](2026-09-01_reaction_endpoints_and_dimer_state_audit.md)
に固定した。次はC0 independent 4 familyのCI-NEB/stringである。

## 7. 参考にした反応・状態情報

- SiH4の単分子分解では、singlet SiH2 + H2への初期反応が報告されている:
  https://doi.org/10.1002/kin.550111104
- Si2H3のdisilenyl radical H2SiSiHはground-state doubletとして報告されている:
  https://doi.org/10.1016/j.cplett.2004.06.008
- Ge2H3のdigermenyl radicalはground-state doubletとして報告されている:
  https://doi.org/10.1016/j.chemphys.2006.08.022

これらはreaction familyとstate-auditの初期値を決める根拠であり、GPU4PySCFでの
state承認とpath収束の代替ではない。
