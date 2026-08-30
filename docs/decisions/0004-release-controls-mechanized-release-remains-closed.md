# 0004: release条件を機械化し、科学承認まではreleaseを閉じる

- 状態: **採択（機構）／保留（科学閾値・state承認）**
- 日付: 2026-08-31
- 対象commit: `ed2445f262fcd1f39fcf134c4908d8f76bb2f73e`
- 関連: [decision 0003](0003-gate1-gpu4pyscf-conditional-go-proposal.md)、
  [release条件review実行記録](../lab_notes/2026-08-31_release_conditions_review.md)

## 判断

次の二つをproduction libraryのrelease controlとして採用する。

1. energy baselineは、splitの`train` partitionだけからfitする
   `per_element_least_squares_v1`（元素数による線形atomic reference）とする。fitに使った
   record ID、各record file checksum、split ID/checksum、design rank、partition別metricを
   versioned artifactへ保存する。rank不足、非accepted record、splitとのrecord不一致、trainに
   存在しない元素への外挿はfail closedとする。
2. non-default charge/multiplicityはversioned state registryの実体を必須とする。
   `state_registry:`という文字列prefixだけでは承認とみなさない。composition、charge、
   multiplicity、exact provenance、entry status、registry checksumが一致し、entryが`approved`
   の場合だけlabel/QCを通す。

一方、dataset releaseはまだ許可しない。production QCのenergy/gradient/spin閾値は
engineering sentinelのままfreezeせず、Part Iの12 charge/spin stateも全件
`pending_scientific_review`とする。`release_allowed: false`と
`engineering_only_pending_scientific_freeze`を維持する。

## 50構造baseline evidence

SoftBank job 1797134のQC accepted 50 recordに、parentを分割しない40 train / 10 holdoutを適用した。
seed 1はSiCl4をholdoutに置き、trainのH/Si/Ge/Cl design rankは4である。

| Metric | train | holdout (SiCl4) |
|---|---:|---:|
| records / compositions | 40 / 4 | 10 / 1 |
| residual RMSE (Ha) | 0.0370791 | 0.0414686 |
| residual max abs (Ha) | 0.124721 | 0.0479154 |
| residual mean (Ha) | 約0 | -0.0374416 |

holdout mean offsetは約-23.5 kcal/molであり、四元素のatomic additivityだけでは未知組成の
total-energy中心を十分に予測できないことを示す。ただし、この50件は各組成のCartesian
displacementを含み、residual RMSEには意図した構造energy変動も入る。したがって、この値を
label accuracy thresholdとして使わない。atomic baselineは学習時の中心化と組成offset分析の
provenance mechanismとして採用し、未知組成の評価はpartition別mean errorと
same-composition relative metricを併記する。

artifactは次に保存した（Git非追跡）。

```text
/lustre/user140002/runs/label/engineering_50_v1/1797134/baseline/
  splits/engineering_50_baseline_split_v1.json
  artifacts/engineering_50_atomic_baseline_v1.json
```

SHA256はsplitが`9a5c4c384835618756380c606ba48d6401ed3ae35f8444b912238fdc75cf0496`、
baseline artifactが`e617e9ce907d02275f9e0cc80338521bcfc15f15859aee53afa0a5b9786f9859`である。

## QC threshold review

50構造ではgradient最大componentは0.298421 Ha/bohr、全Cartesian normは0.539927 Ha/bohr
だった。現行engineering sentinelの1.0 / 2.0 Ha/bohrは全件を通したが、50件は五組成の
限定的displacementで、反応経路や強いnonequilibrium構造を含まない。この比だけで
production thresholdをfreezeしない。

open-shellについてはPart Iの12-state probeで`S2 deviation`最大0.01562、CPU/GPU数値比較は
12/12通過した。しかし各stateは単一geometryの数値parity確認であり、state ordering、
geometry変化に対するroot安定性、教師stateとしての妥当性を証明しない。暫定0.05も
release保証値には昇格させない。

## releaseを開くために残る証拠

- equilibrium近傍から強いdistortionまでを含む独立calibration setで、gradientとenergy
  residual分布を組成・severity別に評価する。
- SiH3/GeH3の候補stateを複数geometryで比較し、state ordering、SCF root、`<S^2>`、
  independent referenceをreviewする。
- state entryごとにevidence、判断者、decisionを記録し、承認対象だけを`approved`へ変更する。
- 上記を根拠にQC config v2を新規作成する。既存v1の閾値を上書きしない。

これらが揃う前の計算はengineering evidenceとして保存できるが、release datasetや
fine-tuning入力にはしない。
