# C0 local-displacement 45件 GPU4PySCF実行記録

- 実行日: 2026-09-01
- 実行先: SoftBank AIデータセンター GPU cluster
- 対象: C0 `equilibrium/local/NMS` quota 45/45
- DFT protocol: `omol_wb97mv_tzvpd_v1`
- 利用判定: calibration用labelとしてQC合格。dataset releaseは未解禁

## 候補生成

base UMA (`uma-s-1p2`, OMol)で6つの独立parentを緩和し、各parentから
Cartesian displacement 8件を生成した。Slurm job `1802368`は`COMPLETED
(0:0)`、経過時間は`00:02:33`だった。

- seed geometry QC: 6 accepted / 0 rejected
- generated candidates: 48 accepted / 0 rejected
- blind portfolio: 45 selected / 48
- artifact root: `/lustre/user140002/runs/calibration/local_candidates_v1/1802368`
- portfolio SHA-256:
  `8de101a5a14723fad4c5ded4aca793df57e0219c79d46e77f684bc8928b6d9f1`

## GPU4PySCF labelとQC

Slurm job `1802379`は`COMPLETED (0:0)`、経過時間は`00:09:47`だった。
45件すべてで一次のdensity-fitting attemptが収束し、direct fallbackは発生
しなかった。

| 項目 | 結果 |
|---|---:|
| label completed | 45 |
| label failed / blocked | 0 / 0 |
| production QC accepted | 45 |
| production QC rejected | 0 |
| 5-atom / 8-atom | 30 / 15 |
| gradient max component, min / mean / max | 0.00524 / 0.03694 / 0.10650 Eh/bohr |

組成別は`SiH3Cl=8`、`GeH3Cl=8`、`SiH2Cl2=7`、`GeH2Cl2=7`、
`Si2H6=7`、`Ge2H6=8`である。

- artifact root: `/lustre/user140002/runs/label/calibration_local_45_v1/1802379`
- label summary SHA-256:
  `2db844f70c25914dd3debb42c42f74f58be7cfc9f45392291f2f2bbceea98445`
- QC report SHA-256:
  `ed8691780db76795d9573540045194e6442f2121b788ad513411f6b73bd4e976`

## 解釈と次の手順

これはengineering-50とは別parentで作った、最初の実C0教師データである。
ただしQC config自体のrelease statusは
`engineering_only_pending_scientific_freeze`であり、composition baseline、科学的
threshold、非default state registryが揃うまで、公式training datasetとしては
releaseしない。

次は計画順に`internal scan/dissociation` 36件を同じ固定DFT protocolで
labelし、C0を81/180まで進める。
