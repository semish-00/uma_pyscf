# Part I: GPU4PySCF検証計画

- 文書状態: 実行計画
- 基準日: 2026-08-22
- 開始条件: CPU PySCF–ORCA chemical ladder完了
- 終了条件: Gate 1 decision record作成
- 後続計画: [Part II: UMAファインチューニング実装計画](02_uma_finetuning_implementation_plan.md)

## 1. 目的

GPU4PySCFが、本プロジェクトのωB97M-V/def2-TZVPD energy/force教師データ生成engineとして
採用可能かを判断し、採用する場合は再現可能なproduction DFT protocolを固定する。

このPartではUMAの学習pipelineを実装しない。検証結果に依存しない最小限のschema検討を除き、
dataset本体の大量生成も開始しない。

## 2. 検証する比較

```text
CPU PySCF ── GPU4PySCF    GPU移植・精度・近似の差
     │             │
     └──── ORCA ───┘      事前学習labelとのcross-code差
```

優先順位:

1. CPU PySCF–GPU4PySCF
2. GPU4PySCF内部の数値設定収束
3. GPU4PySCF–ORCA

CPU–GPUが一致しない状態で、GPU–ORCA差を事前学習labelとの違いとして解釈しない。

## 3. Scope

### 必須scope

- 既存`si_ge_h_cl_ladder_v1`の29構造
- H2、SiH4、SiCl4のsmoke subset
- RKSとUKS
- H/Si/Ge/Cl
- energyとanalytic gradient
- `<S^2>`が取得可能な開殻系
- wall time、GPU memory、CPU memory、収束状態

### 追加scope

GPU実行が安定した後、Gate判定前に小さなcharge/spin matrixを追加する。

- SiH3、GeH3のneutral/cation/anion候補
- 同一geometryの複数multiplicity
- electron count–spin parityが有効な組合せのみ

荷電系の科学的状態選択は別途reviewし、機械的に全組合せを正解labelとして採用しない。

### Scope外

- 大規模dataset生成
- UMA fine-tuning
- 周期境界、slab、band
- 100原子超や重元素の性能保証
- Hessianを全構造で計算すること

## 4. Workstream A: GPU環境の固定

### A1. Hardware/software inventory

記録するもの:

- GPU modelと搭載数
- GPU memory
- NVIDIA driver
- CUDA runtime/toolkit
- Python
- PySCF
- GPU4PySCF
- CuPy
- cuTENSOR
- LibXC
- OS、CPU、RAM

成果物:

- `configs/environments/gpu4pyscf-<host>.yaml`
- version確認logの小さなsummary
- lock fileまたは再構築可能なenvironment定義

### A2. Installation smoke test

- GPUがCuPyから見える。
- PySCF moleculeをGPUへ変換できる。
- ωB97M-V、VV10、def2-TZVPD、RKS/UKS、analytic gradientが動く。
- device番号を明示できる。
- login shellに依存せず、batch/non-interactive executionで動く。

失敗時はversion組合せを一度に複数変更せず、driver/CUDA/CuPy/GPU4PySCFのどの境界かを分離する。

## 5. Workstream B: runnerのGPU対応確認

既存`validation/orca_gpu4pyscf/run_pyscf.py --device gpu`を起点とし、次を検査する。

- geometry、atom order、charge、multiplicityがCPU runと同一
- `multiplicity - 1`をPySCF `spin`に使う
- ordinary gridとVV10 gridがGPU変換後も意図した値
- `grid_response=True`
- RKS/UKS選択
- energy/gradientの単位
- gradientとforceの符号
- GPU runtime provenance
- outputのatomic writeまたは中断時の不完全結果識別

検証コードは`validation/`内に留める。一般実装への移植はGate 1後に行う。

## 6. Workstream C: 段階的計算

### C0. Dry run

- 29 manifestをGPU host上で解決できる。
- basis availabilityとelectron/spin parity検査が全件通る。
- output先がGit非追跡である。

### C1. Minimal smoke

順序:

1. H2 singlet
2. SiH4 closed-shell
3. SiCl4 closed-shell
4. SiH3 doublet
5. H3Si–GeCl3 mixed molecule

各caseを1件ずつ実行し、SCF、gradient、memory、runtimeを確認する。5件が成功するまで29件を一括投入しない。

### C2. Diagnostic lane

既存CPU PySCFと可能な限り同じ条件で比較する。

- grid level 5
- VV10 grid level 5
- tight SCF
- density fittingなしを第一候補

GPU4PySCFが特定のdirect条件をサポートしない、または非実用的に遅い場合は、その事実を
decision recordへ残し、CPU上でdensity fitting誤差を先に測ってからproduction laneへ進む。

### C3. Production-candidate lane

大量label生成を想定した候補設定を比較する。

- density fitting on/off
- GPU4PySCFが推奨・対応するauxiliary basis
- ordinary grid候補
- VV10 grid候補
- SCF convergence
- memory上限とbatch size

一度に一軸だけ変更し、数値差と速度差を分離する。最速設定ではなく、設定変更誤差が
cross-code差および想定学習誤差より十分小さい設定を採用する。

### C4. Full ladder

- 29構造を固定manifestのまま実行
- failed caseは上書きせず、attemptと変更理由を残す
- 初回成功率、最終成功率、再試行回数を分けて集計
- 開殻系は`<S^2>`とtarget deviationを確認

## 7. Metricと暫定判定基準

### 必須metric

- signed/absolute total-energy difference
- 同一組成内relative-energy difference
- gradient component RMSE/MAE/max
- force normと最大force
- `<S^2>`、target、deviation
- SCF収束率と再試行率
- wall time
- peak GPU/CPU memory（取得可能な範囲）

### CPU–GPUの暫定数値gate

CPU–GPUは同一PySCF familyであるため、ORCA–CPUより厳しい基準を期待する。

| Metric | Provisional target |
|---|---:|
| absolute energy difference | `≤ 5e-6 Eh` |
| gradient component RMSE | `≤ 2e-5 Eh/bohr` |
| gradient component max | `≤ 1e-4 Eh/bohr` |
| final calculation success | 29/29 |

この閾値は科学的にfreeze済みではない。smoke subsetで、precision、density fitting、grid由来の
差を分離してからGate 1 reportで確定する。単一の外れ値を隠すため全体RMSEだけを使わない。

### 相対判定

絶対閾値に加え、次を満たすことをGOの条件とする。

- CPU–GPU差が、対応するORCA–CPU差より原則小さい。
- charge、spin、Cl含有、random displacementの特定categoryでのみ差が増大しない。
- production候補設定の近似誤差が、採用するlabel品質目標より小さい。
- energyとgradientで相反する設定選択にならない。

## 8. 異常時の診断順序

1. geometry、atom order、charge、spin、単位
2. RKS/UKSとSCF収束解
3. ordinary grid / VV10 grid / `grid_response`
4. density fittingとauxiliary basis
5. precision、GPU kernel、package version
6. open-shellの`<S^2>`と初期guess
7. ECP/basisの取扱い

設定を緩めて成功させたcaseは、標準protocolと同じdatasetへ無条件に混ぜない。

## 9. Gate 1判定

### GO

- 必須29構造が固定protocolで成功
- CPU–GPUが確定gateを通過
- ORCA–GPU差が既知のORCA–CPU差と整合
- production protocol、environment、制限事項をversion管理できる

### Conditional GO

例:

- UKSだけ特定設定が必要
- Cl含有系のgridを上げる必要がある
- 対象原子数またはGPU memoryに上限がある
- density fittingを使う代わりに明示的な補正・QCが必要

制限を機械判定可能なconfig/QC ruleにできる場合のみConditional GOとする。

### NO-GO

- 構造依存のenergy/gradient差が再現性なく発生
- spin stateがCPUとGPUで安定して一致しない
- 本番規模で成功率または性能が実用水準に達しない
- 問題を検出する自動QCを定義できない

## 10. 成果物

- GPU environment manifest/lock
- GPU対応済みvalidation runnerとtest
- 29構造のnormalized GPU result（rawはGit非追跡）
- CPU–GPU–ORCA comparison CSV/plot
- performance table
- production DFT config候補
- `docs/decisions/`のGate 1 decision record
- Part IIへ反映する制約一覧

## 11. 実行チェックリスト

- [ ] GPU hostと接続・作業directoryを決定
- [ ] hardware/software inventoryを保存
- [ ] environmentを構築・version固定
- [ ] 5-case smokeを完了
- [ ] CPU–GPUの差分parser/comparatorを確認
- [ ] diagnostic laneを完了
- [ ] production-candidate convergence ladderを完了
- [ ] 29-case full ladderを完了
- [ ] charge/spin mini-matrixを完了
- [ ] parity plot、RMSE、性能表を作成
- [ ] Gate 1 decision recordを作成
- [ ] GOの場合、Part IIの前提と設定を更新
