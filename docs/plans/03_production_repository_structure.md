# Part II準備: 本番リポジトリ構成設計

- 文書状態: 設計提案（Gate 1のGO/Conditional GOで採用を確定する）
- 基準日: 2026-08-22
- 位置づけ: [プロジェクト計画書](../project_plan.md)第7節と
  [Part II計画](02_uma_finetuning_implementation_plan.md)第5節の粗いレイアウトを、
  実装に着手できる粒度まで具体化する
- 採用手続き: Gate 1判定時に本書を確定し、変更点があれば`docs/decisions/`へ
  decision recordを残す。**Gate 1前に`src/`以下を作成しない**

## 1. 目的と適用条件

Gate 1がGOまたはConditional GOになった時点で、Part IIのMilestone P2.0以降が
参照する唯一のディレクトリ・パッケージ設計を与える。本書は「何をどこに置くか」
「どのMilestoneでどこを作るか」「validation/から何を移植するか」を先に固定し、
実装時の場当たり的な配置判断をなくすことを目的とする。

Gate 1がNO-GOの場合、本書は失効し、代替案（CPU PySCF縮小dataset、ORCA継続等）の
検討とあわせて再設計する。

## 2. 設計原則

project_plan第6節の原則を、構成規則として言い換える。

1. **Configuration first** — 科学条件（DFT設定、sampling、split、学習条件）は
   すべて`configs/`のversioned fileに置く。scriptやCLI引数に科学条件を書かない。
   全configに`schema_version`と明示単位を必須とする。
2. **Library / CLI分離** — 再利用ロジックは`src/uma_pyscf/`、実行入口は
   `scripts/`とconsole entry point。PBS/GPU batch scriptはconfigをlibraryへ
   渡すだけの薄い層に保つ。
3. **Fail closed** — charge/spin parity、単位、符号、必須fieldが不明なら停止。
   この検査はlibraryの入口（schema層）で行い、各所に散らさない。
4. **空ディレクトリの先行作成禁止** — 各moduleはそれを必要とするMilestoneで、
   testと同時に作る（第7節の対応表に従う）。
5. **validation/の隔離** — `validation/orca_gpu4pyscf/`はPart Iの実験として
   凍結し、`src/`から一切importしない。逆も同様。一般化できる処理は
   コピーでなく**test付きの再実装として移植**する（第8節）。
6. **一方向依存** — パッケージ内の依存は
   `core → schemas → (sampling | calculators | qc | datasets | training | inference | evaluation) → cli`
   の向きのみ許す。横方向（例: sampling→calculators）の依存を作らない。
   横断的に必要な定数・変換はすべて`core`に置く。module間の受け渡しは
   moduleのimportではなく**recordの受け渡し**で行う（例: inferenceは
   prediction recordを書き、evaluationはそのfileを読む。evaluation→inference
   のimportは作らない）。
7. **Git追跡境界** — source、config、schema、manifest、checksum、集計CSV、
   決定記録は追跡する。raw出力、trajectory、LMDB本体、checkpoint、logは
   追跡しない（`data/`、`runs/`、`artifacts/`）。

## 3. 全体レイアウト

```text
uma_pyscf/
├── pyproject.toml              # P2.0。src-layout、console entry point定義
├── README.md                   # P2.0でinstall/test手順を追記
├── docs/
│   ├── project_plan.md
│   ├── roadmap.md
│   ├── plans/
│   ├── lab_notes/
│   └── decisions/              # Gate 1以降の判断記録（ADR形式、連番）
├── configs/
│   ├── environments/           # Part I A1から使用中（host inventory）
│   ├── dft/                    # P2.0: Gate 1で固定したlabel protocol
│   ├── sampling/               # P2.2: scan/displacement/MD候補生成条件
│   ├── datasets/               # P2.4–P2.6: dataset定義、QC閾値、split定義
│   ├── finetune/               # P2.7: 学習hyperparameterとbase checkpoint参照
│   ├── models/                 # P2.7: model registry（学習済みmodelの台帳）
│   └── evaluation/             # P2.7–P2.8: metric、holdout、base比較
├── src/uma_pyscf/
│   ├── core/                   # P2.0: units、spin、id、atomic I/O、例外
│   ├── schemas/                # P2.1: versioned manifest/record schema
│   ├── calculators/            # P2.3: GPU4PySCF adapter、provenance、retry
│   ├── sampling/               # P2.2: 決定論的構造候補生成
│   ├── qc/                     # P2.4: electronic/geometry QC、ledger
│   ├── datasets/               # P2.5–P2.6: canonical record、ASE/LMDB、split
│   ├── training/               # P2.7: fairchem config合成、training record
│   ├── inference/              # P2.7: model registry、条件付き推論、prediction record
│   ├── evaluation/             # P2.8: parity、relative energy、retention
│   └── cli/                    # P2.0〜: thin subcommand（実装は各moduleへ委譲）
├── scripts/                    # batch/PBS/GPU wrapper。config読込→library呼出のみ
├── tests/
│   ├── unit/                   # 依存なしで常時実行（src/をmirror）
│   ├── integration/            # pyscf/ase/fairchem必要。無ければskip
│   └── fixtures/               # 小さな versioned fixture（第9節の上限内）
├── validation/                 # Part I実験。凍結、src非依存を維持
├── data/                       # Git非追跡: canonical records、LMDB shard
├── runs/                       # Git非追跡: label計算・training実行の作業領域
└── artifacts/                  # Git非追跡: checkpoint、大型図表
```

project_plan第7節との差分は次の2点で、採用時にdecision recordへ理由を残す。

- `src/uma_pyscf/core/`の追加。multiplicity↔spin変換を「一か所に集約する」
  というproject_plan第6節の要求に対する、その一か所の置き場である。
- `src/uma_pyscf/cli/`の追加。scripts/を薄く保つ原則を、entry point実装の
  置き場として明文化するもの。

## 4. src/uma_pyscf モジュール責務

| Module | 責務 | 置かないもの |
|---|---|---|
| `core/units.py` | 単位定数と変換（Hartree/eV、Bohr/Å）。数値はCODATA値を一度だけ定義 | 分野固有ロジック |
| `core/spin.py` | multiplicity↔`2S`変換、electron count/parity検査、`<S^2>`target | SCF実行 |
| `core/ids.py` | structure ID・fingerprint生成（sha256）、命名規則 | 乱数 |
| `core/io.py` | atomic write、checksum、JSON/YAML読み書き | schema知識 |
| `core/errors.py` | fail-closed用の例外階層（`ValidationError`、`ProvenanceError`等） | — |
| `schemas/` | structure manifest、label record、dataset manifest、split manifestの
typed schemaとvalidator。`schema_version`の管理と後方互換規則 | 計算・変換の実装 |
| `calculators/` | GPU4PySCF adapter（Gate 1確定protocolの実行）、provenance収集、
retry policyの適用、resource見積り | QC判定、dataset書出し |
| `sampling/` | Tier 1–3の構造候補生成（seed固定）、衝突/重複filter、
charge/spin sibling展開、親子関係の記録 | DFT実行 |
| `qc/` | electronic/geometry QC判定、rejection/retry ledger、dataset release gate集計 | 閾値の値（configs/datasets/へ） |
| `datasets/` | canonical record組立、ASE変換、LMDB書出し、load-back検証、split生成 | 学習実行 |
| `training/` | fairchem config合成、training record（seed/checksum/commit）保存、
学習済みmodelのregistry登録 | metric計算、推論 |
| `inference/` | model registryの読込とmodel_idによるcheckpoint選択、
charge/multiplicity条件付きのUMA/fairchem calculator wrapper（単位変換込み）、
prediction record（構造・条件・energy/forces・model_id・単位）の書出し | metric計算（evaluation/へ）、学習 |
| `evaluation/` | label/simulation/retention metric、parity集計、報告表生成。
入力はlabel recordとprediction record | 推論実行（inference/のrecordを読む） |
| `cli/` | `uma-pyscf <subcommand>`。引数解釈とconfig読込のみで、処理は各moduleへ | 科学ロジック |

### 機能から見た対応

利用者視点の機能と実装の対応を固定する。

| やりたいこと | 入口 | 実装経路 |
|---|---|---|
| 教師データ作成 | `uma-pyscf sample` → `label` → `qc` → `dataset` | sampling → calculators → qc → datasets。QCを通らないlabelはdatasetに入らない（fail closed） |
| 追加学習とmodel保存 | `uma-pyscf train-config` ＋ fairchemの学習CLI | training/がconfig合成とtraining recordを作り、完了したcheckpointを`configs/models/`のregistryへ登録。checkpoint本体は`artifacts/models/<model_id>/`（Git非追跡）、registryにchecksum |
| modelを選んで推論 | `uma-pyscf infer --model <model_id>` | inference/がregistryからcheckpointを解決し、charge/multiplicityを必須入力としてprediction recordを`runs/infer/<model_id>/`へ書く |
| 都度の応用計算 | `scripts/`の薄いscript（＋notebook） | libraryのcalculator/推論APIを呼ぶだけの使い捨て。再利用が2回目に見えた時点でtest付きでsrcへ昇格 |
| 推論結果のDFT検証→追加学習データ化 | `uma-pyscf select` → `label` → `qc` → `dataset` | 下記「active learningの経路」 |
| 応用計算しながら学習候補を収集（on-the-fly収集） | `scripts/`のMD/opt script＋`MonitoredCalculator` | inference/の監視wrapperがflagしたframeを、下記「on-the-fly収集」経由でactive learning経路へ（非同期） |

### Active learningの経路（P2.9）

「推論結果のいくつかをPySCFで検証し、追加学習データとして保存する」機能は、
専用の別pipelineを作らず、既存経路の合成として実装する。

1. `sampling/selection.py`がprediction record（と既存label）を読み、
   選択基準（ensemble不一致、base/fine-tuned差、force外挿、未充足category等）で
   検証候補構造を選ぶ。選択configは`configs/sampling/`に置く。
2. 選ばれた構造は**通常のlabel経路そのまま**（同一versionのDFT config →
   calculators → qc）でGPU4PySCFにより検証・ラベル化する。
3. 合格recordは新しいdataset versionとしてdatasets/へ入る。provenanceに
   「どのmodel_idの推論から、どの選択基準・configで選ばれたか」を必須記録する。
4. 再学習は新dataset versionに対する通常のtraining/経路。評価は固定holdoutで行う。

検証済み推論結果を別枠のdatasetに貯めない理由: 追加データも通常データと同じ
QC・split・provenance規則を通らないと、Gate 2の品質保証が二重管理になるため。

### On-the-fly収集（P2.9の拡張mode）

応用計算（MD、opt、scan）の実行中に学習候補を拾う機能は、
**「収集はon-the-fly、再学習は非同期のbatch」**として実装する。

1. `inference/`に`MonitoredCalculator`（通常calculatorのwrapper）を置く。
   各stepで監視信号 — force外れ値、energy不連続、base modelとの乖離
   （2 model並走）、最近接原子間距離、学習分布からの組成・座標外挿指標 —
   をstep logへ記録し、trigger閾値を超えたframeをflagする。
2. flagされたframeは、Tier 3 sampling候補と同じcandidate manifest形式で
   `runs/infer/<model_id>/<実行名>/flagged/`へ書かれる。simulation自体は
   止めない（trigger発火はlogとflagのみ）。
3. 以後は通常のactive learning経路（selection → label → QC → dataset新version
   → 再学習）へ合流する。provenanceには軌道ID・step・発火したtrigger・
   使用model_idを必須記録する。
4. trigger閾値・記録間隔・flag上限は`configs/sampling/otf_*.yaml`で管理する。
   高温衝突frameの上限（Part II計画Tier 3と同じ制約）を適用する。

**軌道内での即時再学習（VASP MLFF/FLARE型の真のon-the-fly）は初期scope外**
とする。理由:

- UMA規模のfine-tuningは分〜時間単位で、fs刻みのMD loopに同期できない。
- GPU4PySCFのsingle point（ωB97M-V/def2-TZVPD）も分単位で、step内で
  ラベルを返せない。
- 軌道の途中でmodelが替わると、その軌道の再現性とprovenanceが壊れる。
- QC・reviewを飛ばしてdatasetへ入る経路を作らない（fail closed）。

将来必要になった場合は、step毎にmodel versionを記録する明示的なmodeとして、
decision record付きで別途導入する。

### 利便性の規約

都度の応用計算から使いやすいことを、次のAPI/CLI規約で担保する。

- **1行で使えるcalculator**: `from uma_pyscf.inference import load_calculator`
  → `calc = load_calculator("recommended")`でASE calculatorが返る。
  chargeとmultiplicityは`Atoms.info`の必須keyとし、無ければ例外（fail closed）。
- **model alias**: registryに`recommended`と`latest`のaliasを持つ。実体は
  model_idへの参照で、評価（P2.8）に合格したmodelだけを`recommended`へ
  昇格させる。昇格は人のreviewを挟み、自動化しない。
- **一覧CLI**: `uma-pyscf models list` / `uma-pyscf datasets list`で
  registry・dataset manifestを表形式で確認できる。
- **入力形式**: 構造入力はxyz/extxyz/ASE Atomsを受け付け、入口で
  canonical schemaへ正規化する。応用scriptが独自parserを持たない。
- **cycle orchestrator**: `uma-pyscf al-cycle --config`はselect→label→qc→
  datasetを順に呼ぶ薄い合成commandとする。再学習の開始と`recommended`昇格は
  含めない（人の判断点として残す）。

## 5. configs/ の規約

- 形式はYAML、file名は`<内容>_v<N>.yaml`（例: `configs/dft/omol_wb97mv_tzvpd_v1.yaml`）。
- **既存versionは変更しない**。条件を変えるときは`v<N+1>`を追加し、
  参照側manifestが使ったversionを記録する。
- 全fileに共通header: `schema_version`、`created`、`derived_from`（任意）、
  そして数値には単位をkey名で明示（`conv_tol`のような無単位量を除く）。
- `configs/dft/v1`はGate 1 decision recordから機械的に書き起こす。
  Conditional GOの制限（元素、原子数、memory上限等）は`configs/dft/`の
  該当fileと`configs/datasets/`のQC ruleの両方に、機械判定可能な形で置く。
- `configs/environments/`はPart Iで導入済みの形式を継続する。

## 6. scripts/ と CLI

- console entry pointは`uma-pyscf`一つとし、subcommand（`label`、`qc`、
  `dataset`、`split`、`train-config`、`evaluate`等）は各Milestoneで追加する。
- `scripts/`にはhost固有wrapper（PBS投入、GPU host逐次実行、同期）だけを置き、
  中身は「環境変数/引数を読み、configを指定してCLIを呼ぶ」以上のことをしない。
- Part Iの`run_suite.py`が持つ運用概念（子プロセス隔離、attempt ledger、
  stop-on-failure、resume）は、P2.3で`calculators/`＋CLIとして再実装する。

## 7. Milestoneとの対応（作成タイミング）

| Milestone | 新規作成 | 完了時にあるべきtest |
|---|---|---|
| P2.0 | `pyproject.toml`、`src/uma_pyscf/{__init__,core/}`、`cli/`骨格、`tests/unit/core/`、`configs/dft/v1`、lint/型チェック設定 | clean環境でinstall→unit test全通過。spin/units/atomic writeのtest |
| P2.1 | `schemas/`、`tests/unit/schemas/`、`tests/fixtures/`（36構造の正規化結果subset） | round-trip、単位変換、spin parity、旧schema拒否 |
| P2.2 | `sampling/`、`configs/sampling/` | 同一config/seedで同一manifest再生成。親子関係とfilterのtest |
| P2.3 | `calculators/`、`scripts/`のGPU/PBS wrapper、`runs/`運用規約 | adapter単体（pyscf mock）、retry ledger、resume。integrationはGPU hostで |
| P2.4 | `qc/`、`configs/datasets/`のQC閾値 | 各QC ruleの合否fixture、release gate集計 |
| P2.5 | `datasets/`（ASE/LMDB）、`tests/integration/` | 変換round-trip、shard checksum、破損検出 |
| P2.6 | `datasets/splits.py`、`configs/datasets/`のsplit定義 | leakage検査（親子・sibling同group）、split再現性 |
| P2.7 | `training/`、`inference/`、`configs/finetune/`、`configs/models/`（registry）、`configs/evaluation/`のbase評価 | fairchem config合成のsnapshot test、training record完全性、registry round-trip、base UMA推論のfixture検証（単位・charge/multiplicity入力） |
| P2.8 | `evaluation/`、`configs/evaluation/` | metric数学のunit test、category別集計、prediction record読込 |
| P2.9 | `sampling/selection.py`と`inference/`の`MonitoredCalculator`の追加のみ（新directoryなし） | 選択strategyの決定論test、選択provenanceの完全性、trigger発火とflag書出しのunit test |

## 8. validation/ からの移植方針

移植は「Gate 1後、対象Milestoneの実装時に、test付きで再実装」を原則とする。
validation/側のfileは変更せず凍結する。

| validation/資産 | 移植先 | 扱い |
|---|---|---|
| `multiplicity_to_pyscf_spin`、`target_s2`、parity検査 | `core/spin.py` | P2.0で移植。挙動同一をtestで固定 |
| `write_json`のatomic write | `core/io.py` | P2.0で移植 |
| manifest/XYZ検証の考え方 | `schemas/` | P2.1で新schemaとして再設計（コード流用しない） |
| `compare.py`のRMSE/MAE/max数学 | `evaluation/label_metrics.py` | P2.8で移植 |
| `run_suite.py`のattempt ledger・子プロセス隔離 | `calculators/`＋`qc/ledger.py` | P2.3で概念を移植 |
| `collect_environment.py` | 当面validation/に残す | 形式は`configs/environments/`で共通 |
| PBS template類 | `scripts/` | host条件を確認して再作成 |
| ORCA入出力（`prepare_orca.py`等） | 移植しない | 教師engineはGPU4PySCFに統一。ORCA比較はPart I資産のまま |

## 9. 横断規約

### 単位・符号・スピン

P2.1のschema実装で最終確定する前提で、次を既定とする。

- canonical recordは**計算native単位を保持**する: energy Hartree、
  gradient Hartree/Bohr、座標Å。key名に単位を含める（現行validationと同じ）。
- `gradient_*`と`forces_*`は別fieldとし、符号反転は`datasets/`のexport層の
  **一か所**でのみ行う。fairchem向け出力はeV、eV/Å。
- source of truthはmultiplicity（`2S+1`）。PySCF `spin=2S`は常に導出値として
  併記し、逆方向の入力は受け付けない。
- chargeとmultiplicityは全recordの必須fieldであり、default値を持たない。

### fixtureとtestの規模

- `tests/fixtures/`は1 fileあたり100 KB以下、合計5 MB以下を目安とし、
  実DFT結果は正規化済みJSONのsubsetのみ置く。
- `tests/unit/`は追加依存なし（stdlib＋numpy程度）で常時全通過。
  pyscf/cupy/ase/fairchemを要するtestは`tests/integration/`に置き、
  依存が無い環境では自動skipする。

### 命名

- dataset ID: `ds_<domain>_<連番3桁>`（例: `ds_sigehcl_001`）。manifestに
  生成config version、件数、checksumを必須記録。
- split manifest: `split_<dataset_id>_<axis>_v<N>.json`（axis: parent、
  composition、charge、multiplicity、reaction、severity）。
- model ID: `uma_ft_<dataset_id>_<連番3桁>`（例: `uma_ft_ds_sigehcl_001_001`）。
  registryにbase checkpoint、dataset ID/split、training config version、
  checkpoint checksum、licenseを必須記録。base modelもregistryに
  `uma_base_<name>`として登録し、推論・評価のmodel指定を一本化する。
- 学習実行: `runs/train/<dataset_id>/<config_v>/<seed>/`。
- 推論実行: `runs/infer/<model_id>/<実行名>/`（prediction record置き場）。

## 10. Gate 1で確定してから反映する事項

本書では枠だけ定め、値はGate 1 decision recordから転記する。

1. `configs/dft/v1`の内容（density fitting有無、auxiliary basis、grid、
   SCF閾値、memory/batch上限）
2. Conditional GO制約の機械判定rule（対象元素、原子数上限、GPU memory上限）
3. `calculators/`のresource estimatorが使う実測係数（Part Iの performance
   table から）
4. retry policyの許可リスト（Part I C4で観測した失敗モードに基づく）

## 11. 未決事項

- build backend（setuptools src-layoutを既定候補とするが、P2.0着手時に決定）
- lint/型チェックの採用組合せ（ruff＋mypyを既定候補）
- LMDB shardの目標サイズとhash方式（P2.5で実測して決定）
- `docs/decisions/`のADR template（最初のGate 1 recordで様式を固定）
