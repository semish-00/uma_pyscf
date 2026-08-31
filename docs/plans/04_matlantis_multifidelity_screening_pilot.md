# Matlantis/PFP多フィデリティ教師データ選抜pilot計画

- 文書状態: 提案（未採択）
- 基準日: 2026-09-01
- 位置づけ: P2.9 active learningの前段に置くengineering pilot
- production状態: 本書だけではdataset release、Gate 2通過、追加学習modelの昇格を認めない
- 更新: 2026-09-01のDeep Research後、label予算・calibration順序・productionへの昇格条件は
  [教師データsampling・calibration・oracle pool計画](05_teacher_data_sampling_and_calibration.md)
  が本書に優先する。本書はMF0 engineering experimentの設計・記録として保持する。
- 入力 evidence:
  [PFP engineering-50実行記録](../lab_notes/2026-08-31_matlantis_pfp_engineering_50.md)と
  2026-08-31受領のmulti-fidelity調査報告

## 1. 判断

広範な追加文献調査は行わず、pilot計画へ進む。

採用する初期方針は次の通り。

> PFPは候補構造の生成と選抜信号に使い、UMAへ追加する教師labelは
> 固定protocolのGPU4PySCF labelだけにする。

最初からPFP labelをUMAの教師データへ混ぜる方法、multi-head学習、
delta learningは採用しない。これらはgold-only baselineを上回れる可能性を
示す根拠がpilotから得られた場合だけ、別decisionとして検討する。

追加で必要なのは新しいサーベイではなく、次の限定確認である。

1. 現在の契約条件で、PFP出力を内部の選抜特徴量・比較結果・学習候補生成へ
   利用し、必要なprovenanceとともに保存できることを確認する。
2. UMA uncertaintyを構成する独立model/checkpointが科学的に有効か確認する。
   有効なensembleを用意できなければ、pilot初回はそのpolicyを除外する。
3. 現行10,000-step engineering overfitの結果を評価し、pilotで比較対象にする
   UMA modelを固定する。

## 2. この判断の根拠

### 2.1 ローカル実績

`engineering_50_v1`のneutral singlet 50構造について、PFP v9.0.0
`R2SCAN_PLUS_D3`は50/50成功し、resume時には50/50をskipできた。したがって、
PFP推論を外部の安価なscoring段として接続する実行可能性は確認済みである。

一方、同一組成内でoffsetを除いたGPU4PySCFとの比較では、PFPのenergy MAEは
0.03837 eV、force-component MAEは0.07715 eV/angstromだった。狭い既存splitでは
base UMAの方が良く、PFP値を教師labelとして直接足す根拠にはならない。

この50構造はpipeline検証用のengineering setであり、学習・holdout双方をすでに
観測している。active-learning policyの科学的test setには再利用しない。

### 2.2 一次資料の限定確認

- universal MLIPでconfiguration spaceを生成し、第一原理計算でrelabelしてから
  物質固有modelを学習する流れは、2026年の
  [Hänseroth et al.](https://arxiv.org/abs/2606.23214)が直接支持する。
- multi-fidelity学習にはpositive transferの実績がある一方、低・高fidelity間の
  alignment、force label、backbone更新方法に結果が依存する
  [Gardner et al.](https://arxiv.org/abs/2506.14963)。
- 同時multi-fidelity学習の有効例もあるが、GGA/meta-GGA等の個別条件での結果であり、
  PFPからGPU4PySCFへの移植性を保証しない
  [Kim et al.](https://doi.org/10.1021/jacs.4c14455)。

したがって、候補生成・選抜を先に検証し、異なるfidelityのlabel混合は後段に
切り分けるのが最小リスクである。

## 3. 検証仮説

pilotでは次の順に仮説を検証する。

- **H1: screening** — PFPとUMAのforce不一致は、GPU4PySCFに対するUMAの
  大誤差構造をrandomより高い確率で拾う。
- **H2: incremental value** — PFP不一致は、単純な幾何多様性だけでは得られない
  追加情報を持つ。
- **H3: generation** — PFP optimization/短時間MDが、Cartesian displacementだけ
  では得にくい、QC合格かつ非重複の反応・歪み構造を低コストに供給する。
- **H4: learning value** — H1–H3を通過した選抜policyは、同じGPU4PySCF label数で
  randomまたはdiversity-onlyよりUMAの固定test誤差を改善する。

H1–H3は追加学習なしのoffline acquisition評価で検証する。H4の再学習比較は、
前段がGOになった場合だけ実施する。

## 4. 初期scopeと非対象

初回pilotは次に限定する。

- 元素: 現行scopeのH/Si/Ge/Cl
- 系: 非周期分子
- electronic state: neutral singlet
- target label: 固定GPU4PySCF protocolのenergy/forces
- PFP: 固定した`model_version`、`calc_mode`、client versionを全recordへ保存
- energy不一致: 組成別offsetを除いた相対energyまたはrankだけを使用
- force不一致: component差だけでなく、原子ごとの大きさと方向差を保存

charged/open-shell構造は、PFPがcharge/multiplicityを条件として扱わないため、
初回のPFP energy/force disagreement policyには入れない。将来PFPで幾何候補を
生成する場合も、state-aware UMAとGPU4PySCFで再評価し、PFP値を同じ電子状態の
予測と解釈しない。

初回scope外は次の通り。

- PFP labelとGPU4PySCF labelの混合学習
- multi-head/fidelity-token学習
- delta learning
- 本番のon-the-fly再学習
- black-box multi-fidelity Bayesian optimization
- 少数の全成功記録だけから学習するcost/failure予測model

## 5. リポジトリへの配置

新しい`mf_active_learning/`packageは作らない。
[本番リポジトリ構成設計](03_production_repository_structure.md)の既存経路へ
recordで接続する。

```text
configs/sampling/
  mf_pfp_screening_pilot_v1.yaml       # pool、score、quota、budget
src/uma_pyscf/schemas/
  ...                                  # candidate/prediction/selection record拡張
src/uma_pyscf/sampling/
  selection.py                         # policy、quota、dedup、budget
src/uma_pyscf/evaluation/
  acquisition.py                       # ranking、capture、bootstrap
validation/matlantis_pfp/
  ...                                  # Matlantis側runner。srcからimportしない
runs/al/<cycle_id>/
  pool/ scores/ selections/ labels/    # Git非追跡の実行物
```

Matlantis側はversioned prediction recordを書き出す。production側はそのrecordを
schema検査して読み、PFP clientを直接importしない。選抜後は既存の
`label -> qc -> dataset`へ合流し、別系統の教師datasetを作らない。

最低限必要なrecord fieldは次の通り。

- `structure_id`、`parent_id`、`trajectory_id`、`frame_index`
- composition、charge、multiplicity、geometry fingerprint
- generator名、generator config checksum、seed
- predictor名、model version、calc mode、client version
- energy/forces、単位、wall time、API retry、成功/失敗理由
- 各acquisition score、policy、rank、quota、選抜理由
- GPU4PySCF label config version、QC結果、dataset version

## 6. 実行段階

### MF0: baselineと評価境界を固定

1. 10,000-step engineering overfitを評価し、base UMA、200-step失敗model、
   10,000-step modelの位置づけを確定する。
2. active-learning候補と親構造が重ならない固定testを新設する。
3. testはpolicy scoreを見ずに親構造単位で固定し、selection実装から読めない場所・
   manifestに分離する。
4. PFP出力の内部利用・保存条件を契約文書または窓口回答で記録する。

`engineering_50_v1`はschema、metric、resumeのregression fixtureとしてのみ使う。

### MF1: 400候補の共通poolを作る

目標400構造を、同じ親構造群から次の目安で生成する。

| 生成方法 | 目標数 | 目的 |
|---|---:|---|
| Cartesian displacement（sigma 0.02/0.04/0.08/0.12 angstrom） | 160 | 既存samplingとの連続性 |
| bond stretch/compression scan | 80 | 局所的な反応・強歪み |
| reaction interpolation/NEB近傍 | 80 | barrier周辺のcoverage |
| PFP optimization trajectoryの間引き | 40 | basin間・緩和経路 |
| 短時間PFP MDの間引き | 40 | 有限温度の非調和構造 |

全候補に既存geometry QC、duplicate除去、親構造ごとの上限を適用する。
400件に満たない場合は失敗理由を保存し、条件を緩めて数合わせしない。

### MF2: 共通poolをscoreする

全候補について少なくともbase UMAとPFPを実行する。MF0で有効と判定した場合だけ
fine-tuned UMAも加える。

初回のscoreは次とする。

- geometry novelty: pair-distance等の決定論的descriptorによる距離
- PFP disagreement: 組成内相対energy差、force RMS差、最大原子force差、
  force cosine/angular差
- base/fine-tuned disagreement: 有効なfine-tuned modelがある場合のみ
- UMA uncertainty: 独立seedを3本以上用意し、固定calibration setで
  uncertainty-errorの正相関を確認できた場合のみ

学習途中checkpointを独立modelとして数えない。uncertaintyが準備できない場合は
欠損値を0で埋めず、そのpolicy自体をpilot初回から外す。

### MF3: unionだけをGPU4PySCFでlabelする

各policyは20件を選ぶ。

1. random
2. diversity-only
3. PFP-disagreement-only
4. diversity + PFP disagreement
5. UMA uncertainty + diversity + PFP disagreement（MF2の条件を満たす場合のみ）

重複を除いたunionをGPU4PySCFへ送り、初回のHF上限を100件とする。
policyごとに親構造・trajectory上限を設け、近接frameによる見かけの高成績を防ぐ。
全件を通常のQCへ通し、不合格recordも失敗分析用ledgerへ残す。

### MF4: 再学習なしで選抜信号を評価

GPU4PySCF labelを正解として、各policy/scoreについて次を親構造bootstrap付きで
評価する。

- UMA energy/force errorとのSpearman相関
- high-error上位20%を対象にしたAUPRC
- `recall@20`とtop-k error capture
- 選抜集合の親構造、生成法、幾何descriptorのcoverage
- PFP score追加によるdiversity-onlyからの差分
- PFP、UMA、GPU4PySCFのwall time、成功率、retry、失敗category

絶対energy offsetはmetricにしない。主判定はforce errorと同一組成内の
relative-energy errorで行う。

### MF5: 条件付きfine-tuning ablation

MF4がGOの場合だけ、random、diversity-only、採択combined policyを比較する。

- 追加HF budget: 20、40、80の累積learning curve
- training seed: 各条件3本以上
- 初期dataset、training config、step数、評価testを全policyで共通化
- fixed untouched testに加え、元UMAのretention setも評価
- 代表値だけでなくseed分散と親構造bootstrap区間を報告

## 7. 判定Gate

閾値はGPU4PySCF label実行前にconfigへ固定し、結果を見て変更しない。

### Gate MF-A: 実行健全性

- record、checksum、単位、PFP version/calc modeが100%追跡可能
- candidate生成とscoreが固定seedで再現可能
- parent/trajectory漏洩なし
- PFP、UMA、GPU4PySCFの失敗がledgerへ残る

一つでも満たさなければSTOPし、科学比較を行わない。

### Gate MF-B: screening価値

次の3条件中2条件を満たし、親構造bootstrapの80%以上で改善方向が一致すればGOとする。

1. PFP disagreementとUMAのGPU4PySCF force errorのSpearman rhoが0.30以上
2. 20%の選抜budgetでhigh-error上位20%の40%以上を捕捉
3. combined policyがdiversity-onlyよりAUPRCまたはcaptureを0.05以上改善

NO-GOでもPFP構造生成のH3が有効なら、PFPをgeneratorだけに残し、
disagreement scoreは廃止する。

### Gate MF-C: 教師データ選抜価値

同じHF budgetで、combined policyがrandomとdiversity-onlyの双方に対して、

- fixed testのforce MAEをseed中央値で10%以上改善し、
- relative-energy MAEを悪化させず、
- retention metricを5%以上悪化させない

場合に限り、P2.9の標準policy候補とする。昇格には別decision recordを必要とする。

## 8. 実装順序と成果物

| 順序 | 作業 | 主成果物 | 停止点 |
|---:|---|---|---|
| 1 | MF0 baseline/契約/test固定 | model評価、test manifest、確認記録 | 前提不成立なら停止 |
| 2 | schemaとconfig | validator、unit test、pilot config | Gate MF-A準備 |
| 3 | common pool生成 | candidate manifest、生成統計 | QC不足なら再設計 |
| 4 | UMA/PFP score | prediction/score records | version欠落なら停止 |
| 5 | selection dry-run | policy別manifest、union予算 | label前review |
| 6 | GPU4PySCF union label | label/QC ledger | Gate MF-A/B |
| 7 | offline acquisition評価 | acquisition report、cost table | MF-B NO-GOなら終了 |
| 8 | 条件付き学習ablation | learning curves、retention report | Gate MF-C |

最初の実装単位は「MF0と、GPU4PySCFを走らせないselection dry-runまで」とする。
そのreview後に最大100件のHF label予算を確定する。

## 9. 計画上の注意

- 現行50件は全成功かつ5–8原子中心で、cost/failure modelを学習するには狭すぎる。
  初回は実測tableと層別集計に留める。
- PFPのPES softeningは探索coverageを増やす可能性がある一方、非物理構造も増やす。
  強歪みframeの上限とgeometry QCは選抜scoreより前に適用する。
- policyを同じunion上で比較するoffline評価はlabel費用を節約できるが、
  fine-tuning後の価値を直接証明しない。MF-Cを省略してproduction採択しない。
- 固定test、selection union、training追加集合は親構造単位で分離する。
  trajectory frame単位のrandom splitは禁止する。
