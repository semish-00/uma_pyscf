# UMA × GPU4PySCF ファインチューニング実現性調査

- 日付: 2026-08-12
- 状態: 初期調査
- 目的: GPU4PySCFで量子化学教師データを生成し、Meta FAIR ChemistryのUMAをファインチューニングする計画の実現性を評価する。

## 要約

本計画は十分に実現可能である。ただし、OMol25を再現するような新しい汎用モデルを作るのではなく、電荷・スピン状態が重要な限定領域に対してUMAを追加学習する計画として進めるべきである。

有力な対象は次のとおり。

- イオン化・電子付加・酸化還元
- ラジカル反応
- スピンギャップおよびスピンクロスオーバー
- 遷移金属錯体
- 同一構造の複数電荷・スピン面
- 帯電反応中間体や遷移状態

これらは、全電荷および任意のスピン多重度を入力として扱えない現行PFPに対する差別化候補となる。ただし、材料、周期系、対応元素数、計算可能原子数を含む全用途でUMAがPFPより優位という意味ではない。

## UMAとOMol25について確認できたこと

「UMAの事前学習教師データがすべてωB97M-V/def2-TZVPD」という理解は正確ではない。

UMAは複数のDFTデータセットと計算条件を統合したマルチタスクモデルである。分子用の`omol`タスクがOMol25に対応し、このタスクの計算条件がORCA 6によるωB97M-V/def2-TZVPDである。

`omol`タスクでは次の入力を使用する。

- 原子番号
- 原子座標
- 全電荷
- 全スピン多重度
- DFTタスク
- 元素組成

全電荷とスピン多重度は単なるメタデータではなく、モデル内部の埋め込みとMixture of Linear Expertsのルーティングに使われる。

OMol25の主な範囲は次のとおり。

- 140M超のDFT計算（2026年版論文）
- 約83Mのユニークな分子系
- 83元素
- 2～350原子
- 平均約50原子
- 全電荷 -10～+10
- スピン多重度 1～11
- エネルギーと原子力
- 分子、金属錯体、電解液、 biomolecule fragment、反応構造など

参考資料:

- [UMA公式ガイド](https://fair-chem.github.io/uma/)
- [UMA論文](https://arxiv.org/html/2506.23971)
- [OMol25論文](https://arxiv.org/html/2505.08762)
- [UMAモデルカード](https://huggingface.co/facebook/UMA)

## UMAのファインチューニング

FAIRChemはUMAの公式ファインチューニング経路を提供している。

- 独自ASE-readableデータからASE-LMDBを生成可能
- energy-only、energy + forces、energy + forces + stressを選択可能
- 公式テンプレートはローカル1 GPUから開始可能
- 簡易経路でサポートされるのは一度に1つのUMAタスク

本計画では`omol`タスクを使うのが自然である。ただし、GPU4PySCFによるラベルがORCA 6のOMol25ラベルと完全には一致しない場合、得られるモデルは厳密には「GPU4PySCF版ωB97M-Vに適応したomolモデル」となる。

新しい`pyscf-wb97m-v`タスク埋め込みを追加する方法は、公式の簡易ファインチューニング経路を超える。初期段階では既存`omol`への限定的ファインチューニングを採用し、ベースUMAに対するcatastrophic forgettingを評価する。

参考資料:

- [FAIRChem Fine-tuning](https://fair-chem.github.io/fine-tuning/)
- [FAIRChem custom datasets](https://fair-chem.github.io/ase-dataset-creation/)

## GPU4PySCFで教師データを作れるか

必要な主要機能はGPU4PySCFに存在する。

- ωB97M-V
- VV10非局所相関
- def2-TZVPD
- RKS/UKS
- 解析的勾配
- density fitting
- unrestricted DFTのgradient/Hessian
- ASE連携

したがって、エネルギーと力を持つUMA用教師データの生成は可能である。

一方、公式READMEには、def2-TZVPDを使ったdensity fittingはCPUメモリによって約168原子が目安と記載されている。初期対象は20～100原子程度が妥当である。遷移金属、重元素、100原子超の系は、基底、ECP、補助基底、CPUメモリおよびSCF安定性を個別に検証する。

参考資料:

- [GPU4PySCF公式リポジトリ](https://github.com/pyscf/gpu4pyscf)
- [PySCF DFTガイド](https://pyscf.org/user/dft.html)
- [PySCF GPUガイド](https://pyscf.org/user/gpu.html)

## 最重要の実装上の注意: spinの定義

UMAとPySCFでは`spin`の意味が異なる。

- UMA/OMol: スピン多重度 `2S + 1`
- PySCF `mol.spin`: `2S = N_alpha - N_beta`

したがって、通常の変換は次のとおり。

```text
pyscf_spin = uma_spin_multiplicity - 1
uma_spin_multiplicity = pyscf_spin + 1
```

この変換をデータモデルの一か所に集約し、各スクリプトで個別実装しない。電子数とのparityチェックも必須とする。

参考資料:

- [PySCF spin and charge](https://pyscf.org/user/gto.html?highlight=coordinate)

## ORCA 6とGPU4PySCFの差

同じ汎関数・基底名を指定しても、OMol25とGPU4PySCFの教師ラベルが完全一致するとは限らない。

OMol25では次の設定が使われている。

- ORCA 6.0.0
- LibXC版ωB97M-V
- def2-TZVPD
- def2 ECP
- RI-J
- COSX
- DEFGRID3
- tight SCF convergence
- `thresh = 1e-12`
- `tcut = 1e-13`
- 非一重項および一部の一重項でUKS
- 一部の一重項ではスピン対称性を破る初期推測

GPU4PySCFとの差が生じ得る項目:

- 数値積分格子とpruning
- density fitting/COSXの近似
- 補助基底
- SCF収束アルゴリズム
- 初期密度行列
- RKS/UKSの選択
- broken-symmetry解の探索
- ECPの実装
- VV10の数値評価
- 力とエネルギーの数値的一貫性

本格的なデータ生成前に、OMol25の公開ORCA出力から50～100構造を選び、GPU4PySCFで再計算してクロスコード比較を行う。

比較項目:

- 絶対エネルギー
- 同組成内の相対エネルギー
- 原子力
- 電荷差・イオン化エネルギー・電子親和力
- スピンギャップ
- SCF反復回数と収束率
- `S^2`およびスピン汚染
- 最大原子力

絶対エネルギーにほぼ定数または元素組成依存の差があるだけなら、element referenceで吸収できる可能性がある。構造依存・スピン依存の差が大きい場合は、ORCA版`omol`とPySCF版ラベルを明確に区別する必要がある。

## Matlantis/PFPとの比較

2026-08-12時点で確認したPFP v9理論資料では、全計算モードについて系全体の電荷は0であり、非中性系はスコープ外とされている。

スピンについては次のように整理する。

- PBE/R2SCANではスピン分極を考慮する。
- 複数の磁気状態を比較できた場合は、より安定な状態を教師データに採用する。
- すべての磁気状態を網羅したわけではなく、強磁性状態のみ計算したケースも存在する。
- WB97XD分子モードはωB97X-D/6-31G(d)。
- O2などの二原子分子を除き、主にsingletまたはdoubletで、triplet以上を考慮していない。

したがって、比較時の推奨表現は次のとおり。

> UMAのomolタスクは、全電荷とスピン多重度を明示入力し、同一幾何構造の異なる電荷・スピン面を区別できる。現行PFPは中性系を前提とし、利用者が任意の全電荷・スピン多重度を指定してポテンシャル面を選択する用途を想定していない。

比較上の注意:

- PFPは周期材料や大規模系で強みを持つ。
- PFP v9はR2SCANを含み96元素を対象とする。
- UMAの`omol`は非周期分子データで学習されている。
- 優位性は「帯電・開殻分子化学」に限定して検証する。

参考資料:

- [Matlantis PFP v9リリース](https://matlantis.com/ja/news/release-260716/)

## UMA側に残る科学的限界

電荷とスピンを入力できることは、長距離電気相互作用や電子状態を厳密に扱えることと同義ではない。

UMA-S/Mは6 Åの直接カットオフを持つ。message passingにより実効受容野は広がるが、十分に離れた複数成分は非相互作用として扱われる場合がある。UMA論文でも長距離相互作用と未見charge/spinへの一般化が制限として明記されている。

OMol25の評価では、UMA-S-1.2でも次の課題が残る。

- 金属錯体を含むスピンギャップの誤差が大きい。
- 電荷状態間のIE/EAも通常の同一状態内エネルギー差より難しい。
- カットオフ外の非共有結合・静電相互作用は難しい。

これは本計画の機会でもあるが、ファインチューニングのみでアーキテクチャ由来の長距離限界を解消できるとは仮定しない。

## 推奨ロードマップ

### Phase 0: 再現性基盤

- fairchem、PySCF、GPU4PySCF、CUDA、CuPy、LibXCのバージョンを固定する。
- UMA checkpoint名、checksum、ライセンス受諾情報を記録する。
- 単位、電荷、spin変換、ECP、基底、補助基底を設定ファイル化する。
- GPUサーバーとローカルで同一Git commitを使う。

### Phase 1: クロスコード検証

- OMol25から50～100点を選ぶ。
- ORCA 6とGPU4PySCFのenergy/forceを比較する。
- closed-shell、open-shell、charged、metal-containingを含める。
- 許容可能な差と品質管理基準を定義する。

### Phase 2: 小規模教師データ

- 1用途、限定元素集合から開始する。
- 1,000～5,000点を目安にpilot datasetを作る。
- 同一幾何構造の複数charge/spinを意図的に含める。
- 平衡構造だけでなく、歪み、反応経路、遷移状態近傍を含める。

### Phase 3: UMA-Sファインチューニング

- まずenergy + forcesで学習する。
- ベースUMAと比較する。
- 学習領域の改善だけでなく、一般性能の退化も測る。
- 必要ならOMol25のreplay dataを混ぜる。

### Phase 4: 科学的評価

- snapshotのランダム分割を避ける。
- 分子骨格、反応系列、金属中心、配位子系列ごとに分割する。
- 未知組成、未知charge、未知spinを分けて評価する。
- energy/force MAEだけでなく、相対エネルギー、spin gap、反応障壁、最適化成功率で評価する。

### Phase 5: Active learning

- UMAとGPU4PySCFの不一致が大きい構造を追加する。
- charge/spin面の入れ替わり近傍を重点的に取得する。
- DFT計算数ではなく、対象タスクの検証誤差を停止基準にする。

## データとGitの方針

Gitで追跡するもの:

- ソースコード
- 計算・学習設定
- 環境lock file
- 小さなテストfixture
- データmanifest
- checksum
- データ生成コード
- 集計された評価表・図
- 重要な判断記録

Gitで追跡しないもの:

- 生のSCF出力
- wavefunction/checkpoint
- GBW、density matrixなどの大容量電子構造ファイル
- ASE DB/LMDB本体
- trajectory
- 学習checkpoint
- 実行log全量
- キャッシュ

GPUサーバーを生データの主保存先とし、ローカルには必要なsubsetのみ同期する。ローカルに置いた計算結果も原則としてgitignoreする。各データセットは、実体ではなくmanifest、生成条件、件数、checksum、保存場所の論理名によって同定する。

## 想定するリポジトリ構成

```text
uma_pyscf/
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .gitignore
├── .agents/
│   └── skills/
│       └── uma-pyscf-workflow/
│           ├── SKILL.md
│           ├── agents/
│           │   └── openai.yaml
│           ├── references/        # 必要になった場合のみ
│           ├── scripts/           # 必要になった場合のみ
│           └── assets/            # 必要になった場合のみ
├── configs/
│   ├── dft/
│   ├── sampling/
│   ├── finetune/
│   └── evaluation/
├── docs/
│   ├── lab_notes/
│   ├── decisions/
│   ├── roadmap.md
│   └── scientific-protocol.md
├── src/uma_pyscf/
│   ├── calculators/
│   ├── sampling/
│   ├── datasets/
│   ├── qc/
│   ├── training/
│   └── evaluation/
├── scripts/
├── tests/
│   └── fixtures/
├── data/                           # 原則gitignore
├── runs/                           # gitignore
└── artifacts/                      # gitignore
```

## Skill構成についての判断

skill内の`scripts/`、`references/`、`assets/`は現在の公式構成に含まれるが、すべて任意である。空ディレクトリを慣例的に作る必要はない。

本プロジェクトでは次の順序が適切。

1. `SKILL.md`だけで、科学的ワークフローと必須チェックを定義する。
2. UIメタデータが必要なら`agents/openai.yaml`を追加する。
3. 詳細な計算条件表やスキーマが大きくなったら`references/`へ分離する。
4. 同じ変換・検証コードを繰り返し書く状態になったら、決定的な実装を`scripts/`へ追加する。
5. 出力テンプレートなどが必要になった場合のみ`assets/`を追加する。

このプロジェクトでskill内scriptの候補になり得るもの:

- UMA multiplicityとPySCF spinの変換・検証
- 計算設定manifestの検査
- ASEデータの必須フィールド検査
- energy/force単位の検査
- SCF品質フィルタ
- dataset split leakageの検査

一方、GPU4PySCF計算ジョブ本体、dataset生成pipeline、学習CLIはプロジェクトの主要コードなので、skill内ではなく`src/uma_pyscf/`またはリポジトリ直下の`scripts/`に置く。skill内scriptはCodexがワークフローを安定して実行するための補助に限定する。

参考資料:

- [OpenAI公式 Build skills](https://learn.chatgpt.com/docs/build-skills)
- [OpenAI公式 AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md)

## ライセンス上の注意

- `fairchem`コードはMIT License。
- UMA checkpointはHugging Face上でgated access。
- checkpointはFAIR Chemistry Licenseで提供され、商用・非商用利用が許可される一方、地域制限とacceptable-use条件がある。
- 学習済み派生checkpointの共有条件もFAIR Chemistry Licenseに従う必要がある。
- OMol25データセット自体はCC BY 4.0。

実装開始前に、利用主体、公開予定、商用利用可能性を踏まえてライセンス文面を保存・確認する。

## 次の作業候補

1. Gitリポジトリを初期化する。
2. `AGENTS.md`と`CLAUDE.md`を作成する。
3. `.gitignore`に科学計算データとcheckpointの除外規則を定義する。
4. `README.md`と`docs/roadmap.md`を作成する。
5. Python環境方針を決め、`pyproject.toml`を作成する。
6. ORCA–GPU4PySCFクロスコード検証のprotocolを作成する。
7. 実際の反復ワークフローが見えてからrepo skillを作成する。

