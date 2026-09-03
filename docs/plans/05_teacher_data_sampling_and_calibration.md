# Teacher-data sampling, calibration, and oracle-pool plan

- 文書状態: 採択・第一実装increment完了
- 基準日: 2026-09-01
- 対象scope: H/Si/Ge/Cl、非周期分子、原則8原子以内
- 教師protocol: GPU4PySCF ωB97M-V/def2-TZVPD energy/analytic forces
- 関連Gate: Gate 1 Conditional GO、Gate 2 dataset pilot、Gate 3 model value
- 入力evidence: Deep Research sampling survey、engineering-50、MF0 UMA/PFP dry-run

## 1. Decision

教師データは単一のMDまたは単一のactive-learning scoreから作らない。次の四層を分離する。

1. **candidate generation**: deterministic scan、normal/local displacement、reaction path、
   lightweight-model MD、未知反応候補をportfolioとして生成する。
2. **candidate assembly**: model errorを見ず、source quota、parent/reaction/trajectory quota、
   state-aware duplicate除去でcalibrationとoracle poolを固定する。
3. **teacher labeling**: release候補labelは固定GPU4PySCF protocolだけで生成する。
   CPU PySCFは層化sentinel、境界監査、direct fallback比較に使う。
4. **model-aware acquisition**: UMA latent、geometry/graph novelty、PFP–UMA disagreementは、
   独立calibrationで真のGPU4PySCF errorとの関係を確認した後だけ選抜へ使う。

GPU4PySCF予算は潤沢と仮定する。目的はDFT件数の最小化ではなく、generator biasと
selection biasを測定できる厚いoracle poolを作ることである。online active learningを
急がず、まずacquisition非依存にlabelしたpoolでretrospective comparisonを行う。

## 2. Existing evidence and fixed constraints

### 2.1 Completed engineering evidence

- GPU4PySCF production候補はCPU PySCF directに対して十分なenergy/gradient parityを示した。
- 50件のneutral-singlet engineering setは50/50でlabel/QC/ASE-LMDBを通過した。
- base UMA-S-1.2はその狭いsetでGPU4PySCFに近かった。
- 10,000-step modelはtrainへoverfitしたがholdoutを大幅に悪化させた。これは
  training plumbingの確認であり、production modelではない。
- PFP v9.0.0/R2SCAN_PLUS_D3は候補生成と独立criticとして実行可能だが、strong
  distortionでGPU4PySCFとの差が増えた。
- 既存NEB/IRC 4 trajectoryから78件をimportし、UMA/PFP predictionとparent/trajectory
  quota付きselectionをdry-runした。

### 2.2 Constraints that remain binding

- GPU4PySCF density fitting + explicit MINAO、grid、version、containerを変更しない。
- PFP labelをcanonical teacher datasetへ混ぜない。
- PFPは帯電状態を扱わないため、neutral-compatible geometryの生成・補助信号に限定する。
- non-default charge/multiplicityはstate registryが`approved`になるまでreleaseへ入れない。
- 同じparent、reaction、trajectory、scan、geometry-state familyをframe単位でsplitしない。
- engineering-50はregression fixtureであり、acquisition calibrationやfixed testに再利用しない。

## 3. Dataset stages and budgets

### C0: geometry/acquisition calibration — 180 GPU4PySCF labels

scoreを一切見ず、500–1,000程度の候補から次のquotaで180件を固定する。

| source category | labels | purpose |
|---|---:|---|
| equilibrium/local/NMS | 45 | minimum近傍とsoft/hard local distortion |
| internal scan/dissociation/association | 36 | 制御されたbond変化とfragment asymptote |
| interpolation/NEB/IRC | 36 | reaction pathとbarrier近傍 |
| moderate-temperature MD | 27 | anharmonic basinと複数mode結合 |
| strong-distortion/high-temperature MD | 18 | label可能なhigh-energy tail |
| curated graph edit/unknown reaction | 18 | 既知path外のbond break/form候補 |
| **total** | **180** | |

この比率は開始値であり、普遍的な最適値ではない。C0選抜はsource quota、geometry QC、
state-aware duplicate、parent/trajectory上限だけを用いる。

### C0-S: state review — 24–36 engineering labels

C0とは別artifactにし、SiH3/GeH3等の候補stateを複数geometryで比較する。

- charge/multiplicity sibling
- SCF root、occupation、`<S^2>`、state ordering
- GPU density-fit、GPU directまたはCPU PySCF sentinel
- 必要なcaseだけORCA independent reference

承認前は学習datasetへ入れない。state比率を先にproduction quotaとして固定しない。

### T0: dedicated fixed test — 200 GPU4PySCF labels

C0でQCとacquisition signalを評価した後、scoreを計算する前にparent/reaction listとmanifestを
freezeする。C0、engineering-50、training poolとgroupを共有しない。

- interpolation strata: 既知chemistryの未見distortion/path
- extrapolation strata: 未見parent/reaction/topology
- equilibrium、scan、reaction path、MD、high-energy tailを分けてmetricを報告

### P0: acquisition-independent oracle pool — 1,000 GPU4PySCF labels

10,000–30,000 cheap candidatesから、C0と同じblind portfolio ruleで1,000件をlabelする。
全件にbase UMA、neutral-compatible subsetにPFPを実行し、後から同じoracle pool上でpolicyを
比較する。GPU4PySCF budgetが潤沢なので、各policyのunionだけをlabelする初期案は採らない。

retrospective comparisonは次の順で行う。

1. high-error capture、coverage、failure enrichmentだけでsignalをscreenする。
2. random、diversity-only、best model-aware/combinedの3 armへ絞る。
3. 100 / 400 / 800 labels、各3 training seedでlearning curveを作る。
4. T0でbase UMA、fine-tuned model、retentionを比較する。

### P1: expansion — up to 5,000 labels

P0とGate 3を通過した場合だけonline active learningへ進む。各source/stateのfloorを維持し、
残りをcoverage deficit、fixed-test error、electronic-boundary auditへ動的配分する。

## 4. Candidate-generation portfolio

### 4.1 Deterministic controls

- Cartesian displacement: sigma/severityを固定し、seedを記録する。
- normal-mode/Wigner: Hessian source、state、temperature/amplitude、imaginary-mode処理を記録する。
- bond/angle/dihedral scan: coordinateとmoving atom groupを明示する。
- dissociation/association: fragment identity、distance、orientationを記録し、fragmentを
  geometry failureとして自動棄却しない。

### 4.2 Reaction paths

- endpoint interpolation、NEB/CI-NEB、IRCを同じ`reaction_id`へ束ねる。
- pathはframe indexの等間隔だけでなく、mass-weighted arc lengthを基本baselineとする。
- endpoint、最高energy/climbing image、その前後を必ず保持できる将来拡張を許す。
- 一つのreaction pathから近接frameがquotaを占有しないよう`max_per_trajectory`を使う。

最初のreaction registryは8 familyとする。C0本体の36件はengineering-50と
parent/reactionを共有しない前半4 familyから9件ずつ固定する。後半4 familyは
ユーザー指定のreaction-coverage extension 36件とし、C0のacquisition calibrationや
T0 fixed testには数えない。

| tier | reaction family | charge / multiplicity | initial frames | handling |
|---|---|---|---:|---|
| C0 independent | SiHCl3 -> SiCl2 + HCl | 0 / 1 | 9 | C0 path quota |
| C0 independent | GeHCl3 -> GeCl2 + HCl | 0 / 1 | 9 | C0 path quota |
| C0 independent | SiH3Cl -> SiHCl + H2 | 0 / 1 | 9 | C0 path quota |
| C0 independent | GeH3Cl -> GeHCl + H2 | 0 / 1 | 9 | C0 path quota |
| coverage extension | SiH4 -> SiH2 + H2 | 0 / 1 | 9 | engineering-50 overlapを記録 |
| coverage extension | GeH4 -> GeH2 + H2 | 0 / 1 | 9 | engineering-50 overlapを記録 |
| coverage extension | Si2H5 -> Si2H3 + H2 | 0 / 2 | 9 | C0-Sでstate承認後にlabel |
| coverage extension | Ge2H5 -> Ge2H3 + H2 | 0 / 2 | 9 | C0-Sでstate承認後にlabel |

Si2H3/Ge2H3は奇数電子のラジカルなので、中性doubletを主状態とする。quartetは
C0-Sのstate orderingを監査するが、承認前にteacher datasetへ入れない。
Si2H5/Ge2H5からのH2脱離は、反応物とradical生成物を同じdoublet PESで扱える
最初の経路として採用する。直接のSi2H4 -> Si2H3 + HとGe同族反応は、解離後の
spin couplingが増えるため初回からは除外する。

各pathは、反応物・生成物を同じelectronic stateで最適化し、原子対応を固定した上で
CI-NEB/stringを実行する。endpointは原則`fmax < 0.01 eV/angstrom`、可能なら
`0.001 eV/angstrom`を目標とし、endpoint、TS近傍、中間領域からmass-weighted
arc lengthで9件を選ぶ。

### 4.3 Lightweight-model MD

初回はneutral singletだけをPFP v9.0.0/R2SCAN_PLUS_D3で生成する。

- isolated moleculeなのでNVT Langevinを使い、NPTは使わない。
- Hを含むため初期timestepは0.5 fs、短いpreflightでenergy drift/explosionを確認する。
- 300/600/900/1200 K、複数seed、短いtrajectoryを開始値とする。
- global RNG、velocity RNG、thermostat RNGを固定する。
- estimator/calculatorをrun間で共有しない。in-place geometry変更後はcalculatorをresetする。
- 保存frame数ではなくdecorrelated coverageを数える。

PFP MDだけに依存しない。base UMA MDはstate-aware候補源として将来比較できるが、UMA blind
spotを継承するためdeterministic sourceとgraph sourceを残す。charged/open-shell trajectoryは
state-aware cheap backendを別途検証してから追加する。

### 4.4 Unknown-reaction candidates

初回は一般reaction-network engineを導入せず、H/Si/Ge/Cl向けにreviewしたbond-edit whitelistを
使う。

1. graph上で1 bond break、1 bond form、またはpaired editを列挙
2. electron/spin parityと粗いvalence ruleで明白な不整合だけ除く
3. cheap constrained relaxation
4. interpolation/string/NEB候補へ変換
5. 通常のgeometry QCとteacher label経路へ合流

AFIR、Chemoton、metadynamics、nanoreactor、adversarial samplingはC0/P0のcoverage deficitが
示された場合のsecond-stage generatorとする。

## 5. Structure identity, grouping, and provenance

### 5.1 Identity

- geometry duplicate: composition + rotation/translation invariant geometry fingerprint
- calculation identity: geometry + charge + multiplicity + method
- same geometry/different stateはduplicateではない。
- geometry dedupはstate-agnostic viewも記録し、state pairの存在を監査できるようにする。

### 5.2 Indivisible groups

少なくとも次をgroup keyへ含める。

- `parent_id`
- `reaction_id`
- `trajectory_id`
- `path_id`
- `scan_id`
- `graph_edit_family`
- `geometry_state_family`

最終splitはこれらの連結成分を分割しない。

### 5.3 Required candidate provenance

- generator/category/version/config checksum/seed
- source parent/reaction/trajectory/frame
- cheap model/version/mode/runtime（使用した場合）
- geometry hash、state、sampling severity
- selection mode、quota、rank、selected reason
- teacher protocol、runtime、retry、QC、raw/checksum

## 6. Geometry and electronic-boundary policy

不安定構造を三群に分ける。

| class | meaning | handling |
|---|---|---|
| `valid_high_energy` | 大energy/forceだが有限で意図した構造 | label/train候補、source quotaでtail支配を防ぐ |
| `electronic_ambiguous` | root、occupation、spin、SCF継続性が曖昧 | release保留、failure/state ledgerへ保存 |
| `hard_invalid` | atom overlap、NaN/Inf、parser/integrator破綻 | label対象外、理由を保存 |

大force、大energy、SCF difficultyだけをhard reject理由にしない。5–10%のboundary-audit budgetを
持ち、失敗領域をdatasetから不可視化しない。element-pair distance floorはC0のrepulsive scanと
failure分布からQC v2でfreezeする。

## 7. CPU PySCF role

CPU資源は次に使う。

1. C0/P0のsource × severity × stateを層化し、5–10%をGPU density-fitと比較する。
2. strong distortion、dissociation、large gradient、SCF retry caseを優先監査する。
3. GPU direct fallbackとCPU density-fit/directを分離し、差の原因を記録する。
4. GPU queueが混雑した場合の独立label laneとして使えるが、engine/protocol差をmanifestに残す。

CPU/GPU labelを無条件に一つのtraining partitionへ混ぜない。Gate 1で等価性を確認した固定laneだけを
同一datasetへ採用し、境界caseで新たな差が見つかった場合は別protocolとして隔離する。

## 8. Cheap signals and calibration

全C0 recordにbase UMAを実行し、neutral-compatible subsetだけPFPを実行する。

- model-independent: pair-distance、SOAP等、graph edit/topology、source/state coverage
- UMA: energy/forces、structure latent候補、atom-level tail novelty
- PFP: force magnitude/direction disagreement、same-composition energy rank
- teacher outcome: UMA→GPU4PySCF error、runtime、retry、QC/failure

欠損PFP scoreを0で埋めない。利用可能featureだけでweightを再正規化する。

signal評価はSpearmanだけでなく、top-10% recall、precision@k、AUROC/AUPRC、source/state別の
一貫性を使う。数値thresholdはC0実測前にproduction保証値としてfreezeしない。

UMA latent hookはfairchem内部APIへ依存するためcritical pathにしない。まずmodel-independent
descriptorと既存UMA/PFP予測でC0を実行できる状態を作り、latent extractionは並行experimental
adapterとして追加する。

### 8.1 Holdout-guided resampling without test contamination

fixed T0の同じrecord、parent、reaction family周辺をtrainingへ追加した場合、T0はその後の
generalization testとしては使わない。誤差駆動samplingにはC0/P0のdiagnostic partitionを使い、
対象recordと新規候補は同じ`reaction_id` / `parent_id`のindivisible groupへ束ねる。

最初のshadow experimentは次の3 armを同じlabel budgetで比較する。

1. source-stratified random
2. base UMAのrelative-energy/force residual上位の信頼できるgeometry周辺のlocal shell
3. residual上位とgeometry diversityの複合

local shellは一点への過集中を避け、例えばCartesian 0.02/0.05 angstrom、反応座標の
小さな前後変位、及びpath上の隣接区間で構成する。各1 parent/reactionからの上限と
duplicate QCを強制する。元holdoutのteacher labelに電子状態の疑いがある場合は、残差の大きさに
関わらず`electronic_ambiguous`へ隔離し、state/stability監査で承認されるまでshell生成源に
使わない。

現在の10,000-step engineering modelは既知holdoutと未見反応経路の両方を悪化させているため、
そのモデルの残差をacquisition scoreに使わない。base UMAと今後Gate 3を通過した複数seedモデルの
共通苦手領域を優先し、一つの過学習model固有の誤差を増幅しない。

## 9. Repository design

既存の一方向依存`core -> schemas -> sampling/inference/qc/datasets -> cli`を維持する。
Deep Research案の新しいtop-level `features/`、`acquisition/` packageは作らない。

```text
configs/sampling/
  calibration_portfolio_v1.yaml
  ... source-specific generator configs
src/uma_pyscf/sampling/
  portfolio.py                 # multi-manifest assembly, quotas, compact receipt
  portfolio_cli.py
  trajectory_import.py         # path thinning strategies
  generate.py                  # deterministic local/scan operations
src/uma_pyscf/inference/
  uma.py                       # unlabeled UMA predictions; latent adapter later
validation/matlantis_pfp/
  ...                          # PFP client, MD/optimization, disagreement builder
runs/calibration/<id>/
  pool/ portfolio/ labels/ qc/ predictions/ analysis/
```

PFP clientをproduction packageへimportしない。Matlantis側はversioned trajectory/prediction recordを
出力し、production側はfile schemaを介して読む。

## 10. First implementation increment

1. [x] 既存MF0 candidate prediction/selection/trajectory importを独立commitで保全する。
2. [x] `assemble-portfolio`を実装する。
   - 複数candidate manifestとSHA-256を読む
   - source quotaをfail closedで満たす
   - state-aware duplicateを跨いで除く
   - deterministic parent round-robinを使う
   - `max_per_parent`/`max_per_trajectory`を適用する
   - selected candidate manifestとversioned reportをatomic writeする
3. [x] 180件配分を表すconfig fixtureとunit/CLI testsを追加する。
4. [x] trajectory thinningへmass-weighted arc-length baselineを追加する。
5. [ ] source-specific generatorを、local/scan、path、MD、graphの順で追加する。
   C0 local 45件、internal scan/dissociation 36件、independent reaction path 36件は
   label/QC完了。strong-distortionは18/18 DFT収束、暂定QCで17 accepted、1件を
   gradient boundary監査へ保留した。閾値を変えず0.72圧縮replacementをlabel/QC acceptedとし、
   strong-distortion quotaは18/18 acceptedとなった。moderate-temperature MDは親バランス回復を含む27/27
   label/QC accepted。C0は162/180で、残りはcurated graph-edit/unknown-reaction 18件である。
   8 reactionのendpoint生成と、Si2H3/Si2H5/Ge2H3/Ge2H5に対する24件の
   doublet/quartet GPU4PySCF auditは完了した。全12 geometry pairでdoubletが低く
   S2も良好だが、state承認はCPU/direct stability sentinel後とする。
   PFP Langevin MD runnerとengineering preflight configは実装済みだが、C0用の独立
   parent/trajectoryとreview済みgraph editは未実装である。

第一incrementは候補を実際に180件labelしたことを意味しない。実source manifestがquotaを
満たさない限りportfolio assemblyはfail closedする。次の実行単位は500–1,000件の候補源生成と、
そのmanifestを入力にしたC0 freezeである。

## 11. Gates

### C0 exit

- 180件がsource quotaどおり、score非依存に固定される。
- GPU4PySCF label/QC/failureが100%説明される。
- CPU/GPU sentinel差がsource/severity別に説明される。
- QC v2とdistance/severity policyをfreezeできる。
- cheap signalの有効domainまたは無効性を報告できる。

### Gate 2

- C0、C0-S、T0、P0のmanifest/checksum/groupingが完全。
- release対象stateがapprovedである。
- split leakage auditが通る。
- label/failure/source/state/force/energy coverageを説明できる。

### Gate 3

- T0でbase UMAより対象domainを改善する。
- relative energyとforcesの双方で改善する。
- equilibrium retentionと元domain forgettingを報告する。
- 複数seedで方向が再現する。

train loss低下や構造件数だけをGate metricにしない。

## 12. Deferred decisions

次はC0 evidenceが出るまで決めない。

- UMA latentの層、pooling、距離尺度
- PFP disagreementのproduction weight
- high-temperature上限
- metadynamics/AFIR/Chemoton導入
- online active-learning score
- 5,000件時の動的allocation
- fine-tuning head/backbone/replay recipe

これらは現在の実装を止めるblocking decisionではない。
