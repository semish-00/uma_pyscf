# ORCA–PySCFクロスコード検証総括とUMAファインチューニング適合性評価

- 日付: 2026-08-13
- 対象元素: H / Si / Ge / Cl
- 対象系: 気相小分子、閉殻・中性二重項、Si–Ge混合分子、制御された非平衡構造
- 判定: **CPU PySCFラベルを用いた小規模UMAファインチューニング試験へ条件付きGO**

## 要旨

ORCA 6.0.0とCPU PySCF 2.14.0で、同一geometry、charge、multiplicity、
ωB97M-V/def2-TZVPDのenergyとanalytic Cartesian gradientを比較した。
H2による配管確認から始め、SiH4、GeH4、SiCl4、GeCl4、結合伸縮、ランダム変位、
中性二重項ラジカル、Si–Ge–H–Cl混合分子まで合計36構造を検証した。

全36構造で両codeのSCFとgradient計算が正常終了し、設定した暫定gateを全件通過した。
29構造の本検証suiteについて、同一組成内の相対energy RMSEは
`0.00233 kcal/mol`、全468 gradient成分のRMSEは
`1.2705e-5 Eh/bohr = 0.000653 eV/Å`だった。コード差は今回の化学領域では非常に小さい。

したがって、**ORCAラベルで事前学習されたUMAのOMol系taskを、PySCF系ラベルで
ファインチューニングすること自体に、現時点で大きな整合性問題は見えていない**。
ただし、この結論はCPU PySCFに対するものであり、実際の教師データ生成候補である
GPU4PySCFの検証、実際のOMol事前学習ORCA入力の再現、荷電分子、結合解離・反応領域、
より大きな系は未検証である。本格的な教師データ生成開始ではなく、次段階の
GPU4PySCF三者比較と小規模fine-tuning pilotを正当化する結果と位置づける。

## 共通計算条件

### ORCA

- ORCA 6.0.0
- `WB97M-V def2-TZVPD EnGrad`
- `VeryTightSCF DEFGRID3 SCNL`
- `NORI NOCOSX`（クロスコード診断のためRI/COSXを除外）
- `NoAutoStart`
- `%pal nprocs`をPBS割当と一致
- node-local `/tmp` scratch

### CPU PySCF

- PySCF 2.14.0
- LibXC 7.0.0
- RKS（閉殻）/ UKS（開殻）
- ordinary DFT grid level 5
- VV10 nonlocal grid level 5
- `grid_response=True`
- density fittingなし
- SCF convergence `1e-10 Eh`

ORCAのgrid名とPySCFのgrid levelは等価ではないため、名称ではなくenergy/gradientの
収束とクロスコード差を実測して判断した。今回のORCA条件は近似を外した診断laneであり、
OMol事前学習時のproduction設定そのものと断定しない。

## 検証セット

### 初期7構造

- H2
- SiH4 / GeH4の四面体seed
- SiH4 / GeH4の第1結合を1.25倍にした非平衡構造
- SiCl4 / GeCl4の四面体seed

### `si_ge_h_cl_ladder_v1` 29構造

| Category | 構造数 | 内容 |
|---|---:|---|
| bond scan | 12 | SiH4、GeH4、SiCl4、GeCl4の第1結合を0.85、1.15、1.30倍 |
| radical | 4 | SiH3、GeH3、SiCl3、GeCl3の中性二重項・平面seed |
| mixed | 3 | H3Si–GeH3、H3Si–GeCl3、Cl3Si–GeH3 |
| random displacement | 10 | 5親構造へσ=0.04/0.12 Åのseed固定Gaussian変位 |

ランダム変位では剛体並進を除去し、原子間距離filterを適用した。構造生成条件とrandom
seedはsuite manifestおよびXYZ commentに保存している。

## 結果

### 全36構造の集計

| Metric | Result |
|---|---:|
| 正常終了・SCF収束 | 36 / 36 |
| 暫定cross-code gate | 36 / 36 PASS |
| absolute total-energy difference RMSE | 0.00868 mEh |
| absolute total-energy difference MAE | 0.00634 mEh |
| gradient component count | 564 |
| gradient component RMSE | 1.2520e-5 Eh/bohr |
| gradient component RMSE | 0.000644 eV/Å |

absolute total energyの差は元素組成と原子数に依存し得るため、異なる組成を混ぜたRMSEだけで
potential energy surfaceの一致を評価しない。同一組成内で各seedを差し引いた相対energyと、
各Cartesian gradient成分を主要指標とした。

### 29構造suiteの主要指標

| Metric | Result |
|---|---:|
| relative-energy parity points | 23 |
| relative-energy RMSE | 0.002332 kcal/mol |
| relative-energy MAE | 0.001734 kcal/mol |
| gradient components | 468 |
| gradient component RMSE | 1.2705e-5 Eh/bohr |
| gradient component RMSE | 0.000653 eV/Å |
| gradient component MAE | 0.000436 eV/Å |

### Category別

| Category | n | absolute ΔE RMSE [mEh] | gradient RMSE [Eh/bohr] | gradient RMSE [eV/Å] |
|---|---:|---:|---:|---:|
| bond scan | 12 | 0.00946 | 1.5319e-5 | 0.000788 |
| radical | 4 | 0.00801 | 8.1013e-6 | 0.000417 |
| mixed Si/Ge/H/Cl | 3 | 0.00885 | 6.0429e-6 | 0.000311 |
| random displacement | 10 | 0.00755 | 1.2799e-5 | 0.000658 |

29構造suite内の最大差は次の通りだった。

- absolute energy difference: `1.5868e-5 Eh`（SiCl4 bond ×0.85）
- gradient RMS difference: `3.0123e-5 Eh/bohr`（SiCl4 bond ×1.30）
- gradient maximum component difference: `5.4546e-5 Eh/bohr`
  （H3Si–GeCl3 random σ=0.04 Å）

いずれも暫定許容値（energy `5e-5 Eh`、gradient RMS `2e-4 Eh/bohr`、
gradient max `5e-4 Eh/bohr`）より十分小さい。

### 開殻状態

中性二重項の理想値は `<S^2>=0.75` である。

| Molecule | PySCF `<S^2>` | ORCA `<S^2>` | PySCF deviation |
|---|---:|---:|---:|
| SiH3 | 0.752914 | 0.752914 | 0.002914 |
| GeH3 | 0.753473 | 0.753473 | 0.003473 |
| SiCl3 | 0.753255 | 0.753255 | 0.003255 |
| GeCl3 | 0.754412 | 0.754411 | 0.004412 |

両codeの`<S^2>`はほぼ一致し、今回の4構造では顕著なspin contaminationは見られない。

## UMAファインチューニングへの解釈

### 条件付きGOと判断する理由

1. **force/gradientのコード差が小さい**
   - 構造学習を直接駆動するgradient成分が、閉殻、開殻、非平衡、混合元素系で一貫している。
2. **同一組成内の相対energyがよく一致する**
   - 結合伸縮とランダム変位でpotential energy surfaceの形状差が小さい。
3. **開殻referenceが一致する**
   - 少なくとも中性二重項ではUKSのspin状態とgradientが揃っている。
4. **Clを含む系でも差がgate内にある**
   - Hのみの小系に限定した偶然の一致ではなく、Si/Ge/H/Cl領域へ拡張して確認できた。

以上から、ORCA由来の事前学習表現をPySCFラベルで更新したとき、教師codeの変更だけが原因で
大きな矛盾したenergy/force signalが入る可能性は低いと考える。

### 「問題なさそう」の範囲

現時点で支持できるのは、次の限定された主張である。

> H/Si/Ge/Clからなる小さな気相分子の閉殻および中性二重項について、
> 明示したωB97M-V/def2-TZVPD数値条件では、ORCA 6.0.0とCPU PySCF 2.14.0の
> energy/gradient差は小さく、ORCAラベル事前学習モデルをPySCFラベルで
> fine-tuneするpilotを妨げるクロスコード不整合は観測されなかった。

「GPU4PySCFで大規模教師データを生成して問題ない」「OMol事前学習と完全に同一」とまでは、
まだ結論しない。

## ファインチューニング時の注意

### 1. ORCAラベルとPySCFラベルを無造作に混ぜない

事前学習済みweightをPySCF datasetでfine-tuneすることと、一つのtraining dataset内で
ORCA/PySCFのabsolute energyをsource情報なしに混合することは別問題である。
後者では小さなcode-dependent energy zeroが、原子数と組成に応じて系統誤差になる可能性がある。

推奨方針:

- fine-tuning datasetは一つの固定PySCF/GPU4PySCF protocolで統一する。
- label provenanceとしてengine、version、grid、density fitting、charge、multiplicityを保存する。
- ORCA datasetも併用する場合、overlap setから元素別energy offsetまたは組成別baselineを推定する。
- energyだけでなくforce lossと相対energy評価を必須にする。

### 2. 大きな系でのextensive errorを確認する

今回のabsolute energy差は小さいが、分子サイズとともに線形的に蓄積する可能性をまだ除外していない。
Si/Ge中心が複数ある分子・clusterで、`ΔE`を原子数および元素countに回帰し、元素別offsetの有無を調べる。

### 3. 事前学習能力のforgettingを別に評価する

クロスコードlabelが一致していても、狭いSi/Ge/H/Cl datasetへ強くfine-tuneすると、元モデルの
汎用性を失う可能性がある。PySCF-domain validationに加え、元のORCA-domain holdoutまたは
代表的な一般分子setを保持して、fine-tuning前後の劣化を測る。

## 未検証事項

1. **CPU PySCF–GPU4PySCF一致**
   - 本番教師データ経路に対する最重要gate。
2. **実際のOMol事前学習ORCA入力**
   - RI-J/COSX、積分threshold、nonlocal correlationなどの完全なproduction条件。
3. **荷電状態**
   - cation/anion、同一geometry・異なるcharge、diffuse density、SCF安定性。
4. **多重度の拡張**
   - singlet/triplet競合、同一geometryの複数spin state、より強いspin contamination。
5. **反応・解離領域**
   - 0.85–1.30倍より外側、結合切断、radical生成、TS候補、分子間衝突。
6. **大分子・表面cluster**
   - 原子数依存のenergy offset、メモリ・計算時間、SCF収束。
7. **真の周期表面**
   - 今回は分子Gaussian basis計算であり、周期slabやband計算を検証していない。

## 推奨する次段階

### Gate A: GPU4PySCF三者比較

- 同じ29構造をGPU4PySCFでenergy/gradient計算する。
- CPU PySCF–GPU4PySCFを最初に比較する。
- CPU/GPUが一致した後でGPU4PySCF–ORCAを解釈する。
- production候補のdensity fittingとgrid設定を一軸ずつ変えて収束を確認する。

### Gate B: charge/multiplicity matrix

- SiH3+ / SiH3-、GeH3+ / GeH3-など小さな荷電系。
- 同一geometryで複数charge/multiplicityを設定する。
- `<S^2>`、SCF安定性、energy/gradient差を保存する。

### Gate C: reactive nonequilibrium set

- 各代表結合の0.7–2.0倍scan。
- Si–Ge、Si–Cl、Ge–Cl、Si–H、Ge–Hの解離方向。
- xTB/MLIP高温MD由来候補を多様性選択し、既存potential由来biasを抑える。

### Gate D: small fine-tuning pilot

- PySCF/GPU4PySCFのみで統一した小規模training setを作る。
- structure/charge/spin/compositionでtrain/validation/testを分割する。
- fine-tuning前後について次を比較する。
  - PySCF holdoutのrelative-energy/force RMSE
  - ORCA overlap setのrelative-energy/force RMSE
  - 元domain holdoutに対するforgetting
  - charge/multiplicityを変えたときのstate分離

## 再現用ファイル

- 初期結果note: `docs/lab_notes/2026-08-13_h2_orca_pyscf_crosscode_result.md`
- suite投入note: `docs/lab_notes/2026-08-13_si_ge_h_cl_ladder_submission.md`
- suite manifest: `validation/orca_gpu4pyscf/suites/si_ge_h_cl_ladder_v1.json`
- parity用relative-energy CSV:
  `validation/orca_gpu4pyscf/analysis/orca_pyscf_relative_energy_parity.csv`
- parity用gradient CSV:
  `validation/orca_gpu4pyscf/analysis/orca_pyscf_gradient_rms_parity.csv`
- CSV再生成script: `validation/orca_gpu4pyscf/export_parity_csv.py`
- 小さなnormalized resultとraw計算結果: `validation/orca_gpu4pyscf/runs/`（Git非追跡）

## 結論

今回の結果は、**ORCAで作成された事前学習labelとPySCF labelの間に、Si/Ge/H/Cl分子の
fine-tuningを阻害するほどの不連続は見られない**ことを示している。特に相対energyとgradientの
一致は良好であり、PySCF教師データによるUMA fine-tuningの実証へ進む根拠として十分である。

一方で、本番判断の直前に必要なgateはGPU4PySCFとの同等性確認である。したがって現時点の
プロジェクト判断は「実現可能性は高い、small pilotへ進む。ただし大量教師データ生成は
GPU三者比較とcharge/spin検証の後」とする。
