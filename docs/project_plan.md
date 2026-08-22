# UMA × GPU4PySCF プロジェクト計画書

- 文書状態: 初版
- 基準日: 2026-08-22
- 対象リポジトリ: `uma_pyscf`
- 現在地: CPU PySCF–ORCA検証完了、GPU4PySCF検証着手前
- 現在の判断: **GPU4PySCF検証へGO。大量ラベル生成とUMA学習はGate 1通過まで保留**

## 1. 目的

GPU4PySCFを用いて、電荷・スピン多重度を明示したωB97M-V/def2-TZVPDの
energy/force教師データを生成し、UMAの`omol`系checkpointを限定化学領域へ
ファインチューニングする再現可能な方法を確立する。

最初の科学対象はH/Si/Ge/Clを含む気相分子とし、以下を重視する。

- 中性・荷電分子
- 閉殻・開殻状態
- 同一geometryにおける複数charge/multiplicity
- Si–H、Ge–H、Si–Cl、Ge–Cl、Si–Ge結合
- 平衡近傍、歪み、反応・解離方向

表面＋気相分子は重要な拡張対象だが、まず非周期分子・有限clusterで方法を確立する。
周期slab、band、半導体中のキャリア・電場効果は本計画の初期scope外とし、別の周期電子構造手法で扱う。

## 2. 計画を二段階に分離する理由

本計画の最大の技術的不確実性は、実際の教師データ生成engineとなるGPU4PySCFが、
CPU PySCFおよびORCA 6.0.0と十分に整合するかである。この確認前に大規模なdataset生成・
学習基盤を作ると、数値条件の変更による手戻りが大きい。

したがって、次の二つを独立したwork packageとして管理する。

| Work package | 目的 | 終了条件 |
|---|---|---|
| Part I: GPU4PySCF validation | 本番label engineとproduction条件を決める | Gate 1のGO/NO-GO記録 |
| Part II: UMA fine-tuning implementation | dataset生成から学習・評価までを実装する | pilot modelの評価と次段階判断 |

詳細計画:

- [Part I: GPU4PySCF検証計画](plans/01_gpu4pyscf_validation_plan.md)
- [Part II: UMAファインチューニング実装計画](plans/02_uma_finetuning_implementation_plan.md)

Part IIは設計計画として先に固定するが、実装開始条件はGate 1のGOである。Gate 1が
条件付きGOの場合は、採用したGPU4PySCF設定と制限事項をPart IIへ反映してから着手する。

例外として、Gate 1の結果に依存しない基盤・機構は前倒しで実装する
（[decisions/0001](decisions/0001-start-part2-foundation-before-gate1.md)、
[decisions/0002](decisions/0002-extend-early-start-to-gate-independent-machinery.md)。
判定原則: 機構は前倒し可、実行と科学条件の確定はGateに従う）。
大量label生成、calculatorsの本番protocol、学習実行はGate 1通過まで開始しない。

## 3. 現在までに完了したこと

### 環境・計算基盤

- UjilabのPBS/OpenMPI運用を確認
- `/home/seki/uma_pyscf`に作業領域を構築
- ORCA 6.0.0と対応するOpenMPI 4.1.6を導入
- ORCA/PySCF用PBS templateと比較runnerを実装
- raw計算結果をGit追跡外に保つ運用を確立

### CPU PySCF–ORCAクロスコード検証

- H/Si/Ge/Clの36構造
- 閉殻、中性二重項、結合scan、ランダム変位、Si–Ge混合分子
- 36/36でSCF・gradient計算成功、暫定gate通過
- 29構造suiteの同一組成内relative-energy RMSE: `0.002332 kcal/mol`
- 全468 gradient成分のRMSE: `0.000653 eV/Å`
- 二重項4系でORCA/PySCFの`<S^2>`が一致

この結果から、ORCA由来のUMA事前学習表現をPySCF系labelで更新するpilotは
科学的に妥当と考えられる。ただし、この結論はCPU PySCFまでであり、GPU4PySCFは未検証である。

## 4. 全体マイルストーン

| ID | Milestone | 状態 | 判定・成果物 |
|---|---|---|---|
| M0 | 実現性調査とscope定義 | 完了 | feasibility note |
| M1 | ORCA環境とCPUクロスコード基盤 | 完了 | ORCA/PBS/runner |
| M2 | CPU PySCF–ORCA chemical ladder | 完了 | 36構造、全件PASS |
| M3 | GPU環境固定とsmoke test | 未着手 | environment manifest |
| M4 | CPU–GPU–ORCA三者比較 | 未着手 | Gate 1 report |
| M5 | production label protocol固定 | Gate 1内 | versioned DFT config |
| M6 | dataset生成・QC基盤 | Gate 1後 | dataset MVP |
| M7 | 1,000–5,000構造pilot dataset | Gate 1後 | versioned manifest |
| M8 | UMA-S fine-tuning pilot | Gate 1後 | checkpoint + training record |
| M9 | 科学的評価・forgetting評価 | Gate 1後 | evaluation report |
| M10 | active learning / scope拡張 | M9後 | iteration decision |

日付による固定deadlineは、GPU機の利用条件とUMA checkpoint利用環境を確認した後に設定する。
当面は各Gateの品質基準を優先し、計算件数だけを完了条件にしない。

## 5. Gate構造

### Gate 1: GPU4PySCFを教師engineとして採用できるか

判定候補:

- **GO**: CPU–GPU差がORCA–CPU差より十分小さく、production設定を固定できる。
- **Conditional GO**: 対象元素、系サイズ、reference、または数値設定を限定すれば採用可能。
- **NO-GO**: 構造・charge・spin依存の差が解消せず、信頼できるforce labelを生成できない。

NO-GOの場合は、CPU PySCFで小規模datasetを作る、ORCAをlabel engineとして継続する、
または対象scopeを縮小する案を比較し、Part IIをそのまま開始しない。

### Gate 2: pilot datasetの品質

- SCF、spin、force、重複、衝突、単位、provenanceのQCを満たす。
- split leakageがなく、charge/spin/構造分布を説明できる。
- holdoutでDFT label分布とcoverageを確認する。

### Gate 3: fine-tuningの科学的価値

- 対象domainでbase UMAより改善する。
- 元domainの劣化を定量化し、許容範囲内に収める。
- relative energy、force、spin/charge状態分離で改善が再現する。

## 6. 計算・データの基本原則

### 計算条件

- chargeとspin multiplicityを各構造の必須入力とする。
- PySCFの`spin=2S`とUMA/OMolのmultiplicity `2S+1`の変換を一か所に集約する。
- electron countとspin parityを計算前に検査する。
- energyだけでなくanalytic gradientを原則必須とする。
- engine/version/grid/density fitting/basis/ECP/SCF設定をlabelと一緒に記録する。
- SCF不収束を自動的に成功扱いせず、再試行履歴もprovenanceに残す。

### データ分割

- 同じ親分子のscan、MD近接frame、charge/spin siblingを無造作に別splitへ置かない。
- random frame splitを主評価に使わない。
- scaffold、反応系列、組成、charge、spinを用いた複数のholdout軸を用意する。

### Gitと保存

Gitで追跡する:

- source、test、config schema、small fixture
- dataset manifest、checksum、件数、split定義
- 集計CSV、図、判断記録、環境定義

Gitで追跡しない:

- raw SCF output、wavefunction、GBW、density matrix
- trajectory、ASE DB/LMDB本体
- training checkpoint、全実行log、cache

GPUサーバーをraw labelとdataset本体の主保存先とし、ローカルには必要なsubsetと集計結果のみ同期する。

## 7. 役割別ディレクトリ方針

```text
uma_pyscf/
├── docs/
│   ├── project_plan.md
│   ├── plans/
│   ├── lab_notes/
│   └── decisions/              # Gate判断・設計判断
├── validation/
│   └── orca_gpu4pyscf/         # 計画用の独立検証。srcからimportしない
├── configs/
│   ├── dft/
│   ├── sampling/
│   ├── datasets/
│   ├── finetune/
│   └── evaluation/
├── src/uma_pyscf/
│   ├── schemas/
│   ├── calculators/
│   ├── sampling/
│   ├── datasets/
│   ├── qc/
│   ├── training/
│   └── evaluation/
├── scripts/                    # thin CLI entry points
├── tests/
├── data/                       # Git非追跡
├── runs/                       # Git非追跡
└── artifacts/                  # Git非追跡
```

`validation/orca_gpu4pyscf/`は採用可否を判断するための実験コードとして独立を維持する。
そこで確立した一般化可能な処理だけを、Gate 1後にtest付きで`src/uma_pyscf/`へ移植する。

このレイアウトをmodule責務・作成タイミング・移植方針まで具体化した設計は
[本番リポジトリ構成設計](plans/03_production_repository_structure.md)にあり、
Gate 1判定時に採用を確定する。

## 8. 計画の更新方法

- 日々の観察・試行錯誤: `docs/lab_notes/`
- 再利用可能な手順・注意点: repository skill
- 変更理由を残す設計判断: `docs/decisions/`
- milestone、scope、Gateの変更: 本計画書と該当Part計画
- raw結果: Git非追跡領域

Gate判定時には、日付、対象commit、計算環境、入力suite、metric、既知の制限、判定者を
一つのdecision recordへ保存する。

## 9. 直近の次アクション

1. GPU機のOS、GPU、driver、CUDA、Python環境をinventory化する。
2. PySCF/GPU4PySCF/CuPy/cuTENSORの互換versionを固定する。
3. H2、SiH4、SiCl4、中性二重項、混合分子でsmoke testを行う。
4. 既存29構造suiteをGPU4PySCFで計算する。
5. CPU–GPU–ORCAのparity/RMSEと性能をまとめ、Gate 1を判定する。

詳細は[Part I](plans/01_gpu4pyscf_validation_plan.md)に従う。

## 10. 関連文書

- [初期実現性調査](lab_notes/2026-08-12_uma_gpu4pyscf_feasibility.md)
- [クロスコード検証protocol](../validation/orca_gpu4pyscf/protocol.md)
- [CPU PySCF–ORCA総括](lab_notes/2026-08-13_orca_pyscf_validation_summary_and_finetuning_assessment.md)
- [Si/Ge/H/Cl suite投入記録](lab_notes/2026-08-13_si_ge_h_cl_ladder_submission.md)
