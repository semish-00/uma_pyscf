# 教師データcalibration再設計・第一実装increment

- 実施日: 2026-09-01
- 入力: Deep Research sampling survey、engineering-50、10,000-step overfit、MF0 UMA/PFP結果
- 状態: 詳細計画採択、score非依存portfolio assemblyとpath thinning実装完了
- 科学的効力: infrastructure evidenceのみ。C0 label、Gate 2、model昇格は未実施
- 詳細計画:
  [Teacher-data sampling, calibration, and oracle-pool plan](../plans/05_teacher_data_sampling_and_calibration.md)

## 1. 方針変更

GPU4PySCF予算を強い制約とせず、初期active learningで選ばれたunionだけをlabelする方針を
改めた。まず生成法の違う候補をblind quotaで固定し、acquisition非依存の正解集合を作る。

- C0 calibration: 180 GPU4PySCF labels
- C0-S state review: 24–36 engineering labels
- T0 independent fixed test: 200 labels
- P0 acquisition-independent oracle pool: 1,000 labels
- P1 expansion: evidence確認後に最大5,000 labels

PFPはv9.0.0/R2SCAN_PLUS_D3をneutral分子の候補生成・criticに限って使う。PFP値は教師labelへ
混ぜず、帯電系にも使わない。CPU PySCFはC0/P0の5–10% sentinel、strong distortion、
dissociation、SCF retryの監査に使う。

## 2. 実装

`uma-pyscf assemble-portfolio`を追加した。

- 複数candidate manifestとfile SHA-256をreceiptへ固定
- source quotaを不足時fail closedで充足
- parent round-robinをseed付きで決定論化
- global `max_per_parent` / `max_per_trajectory`
- composition、element-pair distance、charge、multiplicityによるcross-source duplicate除去
- selectedと全skip reasonが入力全件を説明するversioned report
- output manifest/reportのatomic write

C0 180件の開始配分を`configs/sampling/calibration_portfolio_180_v1.yaml`へ固定した。実source
manifestは未生成なので、このconfigはまだ実label投入を許可しない。

trajectory importerには`mass_weighted_arc_length`を追加した。endpointを保持し、H/Si/Ge/Clの
固定標準原子量でCartesian path lengthを計算する。対象外元素、atom identity/order変更、
非有限変位はfail closedする。既存configはdefaultの`uniform_index`として互換動作する。

Matlantis側には`run_langevin_md.py`を追加した。PFP v9.0.0/R2SCAN_PLUS_D3、neutral singlet、
NVT Langevin、0.5 fs、200 steps、300/600/900/1200 K、2 seedをpreflight configへ固定した。
velocityとthermostatは別RNGとし、並進・回転運動量を除いた後に初期温度を再scaleする。
Estimator/Calculatorはtrajectory間で共有せず、trajectory、観測系列、runtime、diagnostics、
resume identityを保存する。

ローカルdry-runではengineering seed 5/5を生成し、40 trajectories × 200 stepsのrun gridを
依存関係なしで展開できた。これはrunner検証であり、同じ5 parentはengineering-50と重複するため
C0/T0には入れない。

## 3. 次の実行順

1. equilibrium/local/NMSとinternal scan/dissociationのsource manifestを生成する。
2. 既存NEB/IRCをarc-lengthで再importし、独立reaction/parentを追加する。
3. Matlantisで40 trajectoryのPFP NVT Langevin engineering preflightを実行し、
   energy range、temperature range、max force、centered radiusをreviewする。
4. reviewed bond-edit whitelistから未知反応候補を作る。
5. geometry QC後の候補総数を500–1,000件にし、blind assemblyでC0 180件をfreezeする。
6. GPU4PySCF label jobとCPU PySCF sentinel jobを投入する。

source quotaを満たせない場合は同じsource generatorのseed/parentを追加する。UMA/PFP scoreを見て
不足分を埋めたり、他source quotaへ黙って振り替えたりしない。
