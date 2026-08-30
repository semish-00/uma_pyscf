# Part II: UMAファインチューニング実装計画

- 文書状態: 実行中（Gate 1 Conditional GO採択済み）
- 基準日: 2026-08-31
- 開始条件: GPU4PySCF Gate 1がGOまたはConditional GO（2026-08-31に充足）
  （Gate 1非依存のP2.0/P2.1基盤のみ
  [decisions/0001](../decisions/0001-start-part2-foundation-before-gate1.md)で前倒し）
- 初期対象: H/Si/Ge/Clの非周期分子、charge/spin条件付き`omol` fine-tuning

## 1. 目的

Gate 1で固定したGPU4PySCF protocolを用い、構造生成、DFT label、品質管理、dataset変換、
UMA fine-tuning、科学的評価までを再現可能なpipelineとして実装する。

単発の学習成功ではなく、次を満たすことを成果とする。

- 同じmanifestからdatasetを再構築できる。
- 各labelの計算条件とQC履歴を追跡できる。
- split leakageを防げる。
- base UMAとの比較とforgetting評価を自動化できる。
- charge/multiplicityを変えた状態面を評価できる。

## 2. Pilot scope

### 化学scope

- 元素: H / Si / Ge / Cl
- 主要分子: SiHx、GeHx、SiClx、GeClx、Si–Ge–H–Cl混合
- state: neutral/cation/anion、singlet/doubletを中心に科学的に妥当なmultiplicity
- geometry: minimum近傍、結合scan、角度変形、解離方向、反応候補、分子間配置
- 規模: Gate 1で検証済みの最大8原子。8原子超と有限surface clusterは別gateで拡張

### 初期dataset規模

- engineering smoke: 50–200構造
- training pilot: 1,000–5,000構造
- 拡張は計算件数ではなくholdout metricとcoverageで判断

### Scope外

- 新しい汎用基礎modelの事前学習
- 全83元素への一般化
- 周期slab、bulk、band、キャリア濃度を直接条件入力するmodel
- アーキテクチャ変更による長距離静電相互作用の根本解決
- commercial/production deployment

## 3. 実装原則

### Configuration first

計算・sampling・dataset・training・evaluation条件をversioned configに置き、CLI引数やscriptへ
科学条件を散在させない。configにはschema versionと明示的な単位を持たせる。

### LibraryとCLIを分離

- 再利用ロジック: `src/uma_pyscf/`
- 実行入口: `scripts/`またはconsole entry point
- notebookは探索・可視化に限定し、唯一の実装にしない
- PBS/GPU batch scriptは薄く保ち、configをlibraryへ渡す

### Fail closed

charge/spin parity、SCF convergence、単位、atom order、gradient/force符号、dataset必須fieldが
不明な場合は処理を停止する。警告だけでtraining datasetへ混入させない。

## 4. Target architecture

```text
structure manifest
        │
        ▼
sampling / structure generation
        │
        ▼
GPU4PySCF labeling ── raw output store
        │
        ▼
normalization + QC ── rejection / retry ledger
        │
        ▼
canonical dataset records
        │
        ├── split manifests
        ├── ASE-readable intermediate
        └── ASE-LMDB / fairchem input
                 │
                 ▼
          UMA fine-tuning
                 │
                 ▼
       domain + retention evaluation
```

## 5. Repository implementation

```text
configs/
├── dft/                 # Gate 1で固定したlabel protocol
├── sampling/            # scan、displacement、MD候補生成
├── datasets/            # composition、state、QC、split
├── finetune/            # UMA checkpointとtraining hyperparameters
└── evaluation/          # metric、holdout、base comparison

src/uma_pyscf/
├── schemas/             # versioned manifest/record schema
├── calculators/         # GPU4PySCF adapter、provenance
├── sampling/            # deterministic structure candidates
├── datasets/            # canonical record、ASE/LMDB conversion
├── qc/                  # SCF/spin/geometry/units/duplicates
├── training/            # fairchem config composition
└── evaluation/          # parity、relative energy、force、retention
```

最初から全directoryを空で作らず、各milestoneで必要なmoduleとtestを同時に追加する。
directory・module粒度の詳細設計と作成タイミングは
[本番リポジトリ構成設計](03_production_repository_structure.md)を参照する。

## 6. Milestone P2.0: 基盤scaffold

### 実装

- `pyproject.toml`
- package layoutとtest runner
- formatter/linter/type check方針
- config/schema versioning
- CLI共通error handlingとstructured logging
- local/GPU hostのenvironment文書

### 完了条件

- clean environmentからinstall/testできる。
- sample configをloadしてvalidationできる。
- validation実験コードを`src`からimportしない。

## 7. Milestone P2.1: Canonical data model

各label recordの必須field候補:

- stable structure ID
- atomic numbers、Cartesian positions、atom order
- total charge
- spin multiplicity
- PySCF spin `2S`
- electronic-state/initial-guess provenance（必要な場合）
- energy
- Cartesian forcesまたはgradientと明示的な符号
- units
- engine/package versions
- functional、basis、ECP、aux basis
- grid、VV10 grid、density fitting、SCF threshold
- convergence、iteration、`<S^2>`
- parent structure、sampling method、random seed
- raw result logical locationとchecksum
- QC status、rejection/retry history

### 設計判断

- canonical内部単位とfairchem出力単位を分ける。
- gradientとforceを別名で保持し、暗黙の符号反転を禁止する。
- multiplicityとPySCF spinを両方保存し、どちらがsourceかを明確にする。

### 完了条件

- JSON schemaまたは同等のtyped schemaがある。
- round-trip test、unit conversion test、spin parity testがある。
- 既存36構造のnormalized resultを読み込める。

## 8. Milestone P2.2: Structure generation

### Tier 1: deterministic local perturbation

- bond length scan
- angle/dihedral scan
- normal-modeまたはCartesian displacement
- 複数振幅とseed固定
- collision、重複、孤立fragment filter

### Tier 2: reaction-oriented candidates

- reactant/product interpolation
- dissociation/association coordinate
- TS候補周辺
- 同一geometryのcharge/spin sibling

### Tier 3: finite-temperature/diverse candidates

- xTBまたは既存MLIPでMD候補を生成
- 生成potentialをprovenanceに記録
- descriptor/距離/組成で多様性選択
- 高温衝突frameの上限を設ける

高温MDのみをdataset sourceにしない。制御scanで自由度を覆い、MDは反応的・多分子領域を補完する。

### 完了条件

- 同一config/seedから同じ候補manifestを再生成できる。
- 親子関係を用いてsplit leakageを防げる。
- DFT投入前にgeometry QC reportを生成できる。

## 9. Milestone P2.3: GPU4PySCF label pipeline

### 実装

- manifest batch reader
- resource estimatorとbatch grouping
- GPU device allocation
- scratch/output管理
- atomic result write
- retry policy
- normalized result変換
- resume/skip completed
- summaryとfailure ledger

### Retry policy

標準設定での初回結果を保存したうえで、許可された再試行だけを順番に適用する。

例:

1. SCF max cycle増加
2. 初期guess変更
3. level shift/damping等のreview済み設定
4. stability/broken-symmetry処理（対象caseのみ）

異なる数値protocolで得た結果を同一datasetへ入れる場合は、Gateで事前承認された等価性検証を必要とする。

### 完了条件

- 50–200構造のengineering setを中断・再開可能に処理できる。
- failure原因をcase別に集計できる。
- rawとnormalized recordのchecksum対応が取れる。

## 10. Milestone P2.4: Quality control

### Electronic-structure QC

- SCF convergence
- finite energy/gradient
- `<S^2>` deviation
- electron/spin parity
- force max/norm
- energy/forceの外れ値
- retry protocol

### Geometry/data QC

- atom orderとID
- minimum distance
- disconnected fragments（意図した場合を除く）
- duplicate/near-duplicate
- parent/trajectory leakage
- unitとforce sign
- charge/multiplicity coverage

### Dataset release gate

- accepted/rejected/retried件数
- category/元素/原子数/charge/spin分布
- energy/force分布
- rejection理由
- raw location/checksum

をdataset cardに記録する。

## 11. Milestone P2.5: ASE/fairchem dataset変換

### 実装

- canonical recordからASE-readable structureへ変換
- energy/force、charge、multiplicity、task情報を保持
- ASE-LMDB生成
- shard、checksum、record count
- load-back verification

fairchem/UMAの正確な必須keyとversion依存仕様は、実装時に採用versionの公式例と照合し、
adapter testで固定する。推測したfield名をdataset標準として先行固定しない。

### 完了条件

- 既知のsmall fixtureを変換し、全fieldをlosslessに読み戻せる。
- record数、energy、forces、charge、multiplicityをsourceと照合できる。
- corrupted/incomplete shardを検出できる。

## 12. Milestone P2.6: Dataset split

最低限、次の評価splitを用意する。

| Split axis | 目的 |
|---|---|
| parent/scaffold holdout | 近接geometryのleakage防止 |
| composition holdout | 未知組成への一般化 |
| charge holdout | 未知chargeへの一般化 |
| multiplicity holdout | 未知spinへの一般化 |
| reaction-family holdout | 反応系列の外挿 |
| nonequilibrium severity | 平衡近傍から遠い構造の評価 |

同一geometryのcharge/spin siblingは、通常splitでは同じgroupへ置く。charge/spin generalizationを
測る専用splitだけは意図的に分け、その設計を通常metricと区別する。

## 13. Milestone P2.7: UMA baselineとfine-tuning

### Baseline

- 採用するUMA checkpoint名、checksum、license、taskを記録
- 学習前に全holdoutでbase UMAを評価
- inference unitsとcharge/multiplicity入力をfixtureで確認

### Training ladder

1. 50–200構造でoverfit smoke
2. 小datasetでtraining/validation plumbing確認
3. 1,000–5,000構造pilot
4. 必要な場合のみhyperparameter sweep

最初は既存`omol`taskへの限定fine-tuningを採用する。新しいPySCF task embeddingの追加は、
既存経路でcross-code shiftを扱えないことが実証された場合の別decisionとする。

### Loss

- energy + forcesを基本
- energy/force loss weightをversion管理
- 原子数・組成によるenergy baselineの扱いを明示
- ORCA/PySCF labelを無造作に一つのdatasetへ混ぜない

### 再現性

- seed
- checkpoint checksum
- exact dataset manifest/split
- code commit
- environment
- training config
- hardware

をtraining recordに保存する。

## 14. Milestone P2.8: Evaluation

### Label metric

- energy MAE/RMSE
- force component MAE/RMSE
- per-structure force RMSE/max
- 同一組成内relative energy
- charge差、ionization/electron-affinity候補
- spin gap
- reaction barrier/profile

### Simulation metric

- geometry optimization成功率
- optimized structure error
- reaction path/scanの順位と形状
- MD短時間安定性（必要な段階のみ）

### Retention/forgetting

- base UMAとfine-tuned modelを同じholdoutで比較
- 対象domain改善と元domain劣化を別々に報告
- replay dataの有無を比較する場合、sourceとlicenseを記録

### 完了条件

- base UMAに対する改善をconfidence intervalまたは複数seedで確認
- 代表caseのparity plotだけでなく、category別metricを報告
- failure caseを構造・charge・spin別に分類

## 15. Gate 2とGate 3

### Gate 2: Dataset pilot GO

- engineering setのpipelineが再現可能
- QC、provenance、splitが完全
- label coverageが目的scopeを満たす
- failure/retry biasを説明できる

### Gate 3: Model iteration GO

- 対象domainでbase UMAを一貫して改善
- relative energyとforcesの双方で改善
- charge/spin状態を意図どおり区別
- forgettingが許容可能、またはreplay等で制御可能

改善が一部categoryだけの場合はConditional GOとしてscopeを限定する。

## 16. Milestone P2.9: Active learning

候補選択:

- model ensemble disagreement
- base/fine-tuned UMAの差
- UMA–GPU4PySCF error
- charge/spin面の接近・入れ替わり
- force/energy外挿
- 未充足の組成・構造category

loop:

1. candidate生成
2. uncertainty/diversity選択
3. GPU4PySCF label
4. QC/dataset version更新
5. retraining
6. fixed holdout評価

停止条件はDFT計算件数ではなく、対象科学metricの改善飽和とする。

## 17. 表面＋気相分子への拡張

気相pilotのGate 3通過後、有限clusterとして次を段階的に追加する。

- 小さなSi/Ge cluster + hydride/chloride
- 終端したsurface cluster + gas molecule
- 吸着前後、接近、解離、反応候補
- cluster size convergence

有限clusterの端・終端に由来するartifactを評価し、周期slabの代替と無条件にみなさない。
周期表面、帯電slab、外部電場は別project/gateとする。

## 18. 想定リスクと緩和策

| Risk | 影響 | 緩和策 |
|---|---|---|
| GPU4PySCF数値差 | label不整合 | Gate 1、固定protocol、overlap set |
| SCF/state収束bias | dataset欠損・誤state | `<S^2>`、retry ledger、state review |
| absolute energy offset | 混合labelの系統誤差 | 同一engine統一、relative metric、baseline分析 |
| sampling bias | 適用範囲不足 | deterministic scan + MD + diversity |
| split leakage | 過大評価 | parent/reaction/trajectory group split |
| catastrophic forgetting | 汎用性能低下 | base holdout、replay比較、低学習率/限定更新検討 |
| 長距離相互作用限界 | charge系の誤差 | 距離別評価、scope明示、architecture問題を分離 |
| raw data消失 | 再現不能 | server主保存、manifest/checksum、backup方針 |
| license制約 | checkpoint共有不可 | license記録、配布範囲review |

## 19. 実装順序チェックリスト

- [x] Gate 1の採用protocolを本計画へ反映
- [x] package/config/schema scaffold
- [x] canonical recordとunit/spin test
- [x] deterministic structure generation
- [x] GPU4PySCF batch label MVP（unit test・local dry-run）
- [x] QC/retry/provenance（engineering QC、releaseはfail closed）
- [x] SoftBank GPU sample→label→QC integration smoke（job 1797122）
- [ ] 50–200構造engineering dataset
- [ ] ASE/fairchem変換とload-back test
- [x] split generatorとleakage test
- [ ] base UMA evaluation
- [ ] overfit smoke
- [ ] 1,000–5,000構造pilot
- [ ] fine-tuningと複数seed評価
- [ ] retention/forgetting report
- [ ] Gate 3判断
- [ ] active learningまたはsurface cluster拡張

## 20. 成果物

- install可能な`uma_pyscf` package
- versioned config/schema
- GPU4PySCF label pipeline
- QC/retry/provenance system
- dataset manifest/card/split
- ASE-LMDB変換器
- base UMA evaluation
- training config/checkpoint record
- domain/retention evaluation report
- Gate 2/Gate 3 decision records
