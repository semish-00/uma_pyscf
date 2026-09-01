# C0 internal scan / dissociation 36件 GPU4PySCF実行記録

- 実行日: 2026-09-01
- 実行先: SoftBank AIデータセンター GPU cluster
- 対象: C0 `internal scan/dissociation/association` quota 36/36
- DFT protocol: `omol_wb97mv_tzvpd_v1`
- 結果: 36 label成功、source-aware QC 36/36 accepted

## 候補生成と固定

C0-localでbase UMA緩和した6 parentに対し、4つの塩素化parentでは
M-Cl結合を、`Si2H6`/`Ge2H6`では端M-H結合を0.75–2.00倍に変化させた。
48件をgeometry QC 48/48で作成し、model/DFT scoreを見ずpparentごと6件、
計36件を固定した。

- candidate artifact root:
  `/lustre/user140002/runs/calibration/internal_scan_v1/c595f88`
- selected manifest SHA-256:
  `444f5fa87cda713ae044dd247232a8182d1261755d21c5f3f9c926d2b4734090`

## GPU4PySCF label

Slurm job `1802458`は`COMPLETED (0:0)`、経過時間は`00:07:56`だった。
36件すべてで一次density-fitting attemptがSCF収束し、failed、blocked、
direct fallbackは0件だった。

- label artifact root:
  `/lustre/user140002/runs/label/calibration_internal_scan_36_v1/1802458`
- label summary SHA-256:
  `b851764e4efc9f048ba14d868d36f1d61848e10b153718400c05bf6e499d01e2`

## Source-aware QC

既存の単一分子用QCは21 accepted / 15 rejectedとした。全15件の理由は
`fragments`で、計画した解離側構造を正しく検出したものであり、計算失敗ではない。

`omol_wb97mv_tzvpd_dissociation_qc_v1`ではfragmentの存在のみを許可し、
protocol/runtime、SCF、gradient、原子間距離、duplicate、checksum、state検査は
維持した。再QCは36 accepted / 0 rejectedだった。

| gradient max component | Eh/bohr |
|---|---:|
| minimum | 0.01376 |
| mean | 0.08811 |
| maximum | 0.39161 |

- dissociation QC report SHA-256:
  `c0be62af2234c60aefa844a23650c9df83af1df10e91e77340e4e7ad6cb0f566`

C0は累計81/180件となった。次は`interpolation/NEB/IRC` 36件で、
endpoint、barrier近傍、中間領域を同一reaction groupで固定する。
