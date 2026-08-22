# ORCA–PySCF–GPU4PySCFクロスコード検証基盤の初期実装

- 日付: 2026-08-12
- 状態: scaffold実装・ローカルCPUスモーク完了
- 対象: `validation/orca_gpu4pyscf/`

## 結論

クロスコード検証は、UMA本体コードやSkillから分離した独立実験として
`validation/orca_gpu4pyscf/` に配置した。

比較はORCAとGPU4PySCFの二者比較ではなく、次の三角比較とする。

1. CPU PySCF ↔ GPU4PySCF
2. CPU PySCF ↔ ORCA
3. GPU4PySCF ↔ ORCA

この構成により、GPU実装に由来する差と、量子化学コード間の差を分離できる。

## ORCAはCPUかGPUか

OMol25再現に用いるORCA 6.0.0のωB97M-V/def2-TZVPD計算はCPU計算として扱う。

ORCAの公式並列実行方法はOpenMPIを用いたCPUマルチプロセスであり、入力中の
`PAL`または`%pal nprocs`によって並列モジュールを起動する。ORCA本体を
`mpirun`で直接起動してはいけない。

したがって、GPUサーバー上でORCAを動かすこと自体は可能だが、ORCA用にGPUを
予約する利点は基本的にない。同じノードまたはクラスタ内で、ORCAはCPU、
GPU4PySCFはGPUを使用する役割分担が自然である。

公式資料:

- [ORCA 6.0 Calling the Program (Serial and Parallel)](https://www.faccts.de/docs/orca/6.0/manual/contents/calling.html)

## 参照した既存プロジェクト

正しい参照先は次のローカルcheckoutである。

`/Users/sekisho/aixtal/tech/シミュレーション/反応性熱流体/ho-pyscf-repro`

このディレクトリは読み取り専用で参照し、変更を加えていない。

確認できた実績:

- PySCF計算は`ssh ujilab`先のPBS/Torque計算ノードで実行する。
- 既存の実行Pythonは`/home/seki/miniconda3/envs/qcthermo/bin/python`。
- ujilabではPySCF 2.14を使用している。
- ωB97M-Vのエネルギー、解析勾配、Hessianまで既存プロジェクトで検証済み。
- meta-GGAの数値格子感度が実際に観測されており、標準presetではPySCF grid level 5を使用している。
- PySCFの`spin`は多重度ではなく`n_alpha - n_beta = 2S`である。
- 開殻ではRKSではなくUKSを使用し、`<S²>`を記録する運用がある。
- 実計算とdry-runを分離し、dry-runではPySCFをimportしない設計が有効だった。

今回の実装ではこれらの知見を採用したが、`qcthermo`のコード自体はコピーせず、
クロスコード比較に必要な最小限の独立CLIとして実装した。

## 実装したファイル

```text
validation/orca_gpu4pyscf/
├── README.md
├── protocol.md
├── common.py
├── run_pyscf.py
├── prepare_orca.py
├── parse_orca.py
├── compare.py
├── configs/
│   └── h2_wb97mv_def2tzvpd.json
├── structures/
│   └── h2.xyz
├── jobs/
│   ├── run_pyscf_cpu_pbs.sh
│   └── run_orca_cpu_pbs.sh
├── tests/
│   ├── fixtures/h2.engrad
│   └── test_validation.py
└── runs/
    └── .gitignore
```

`runs/`以下にはORCA出力、PySCF結果、比較レポートを保存するが、Git追跡しない。

## データ契約

manifestでは少なくとも次を明示する。

- 一意なcase ID
- XYZ構造
- 全電荷
- スピン多重度
- 汎関数と基底関数
- SCF収束条件
- PySCF通常格子とVV10用非局所格子
- ORCAのCPU数、メモリ、キーワード
- 暫定比較許容値

実行前に原子番号から電子数を求め、電子数とPySCF spinの偶奇が一致するか検査する。

変換規則:

```text
pyscf_spin_2s = multiplicity - 1
```

正規化結果ではエネルギーをhartree、勾配をhartree/bohrで保存する。教師データの
forceは勾配の負であるため、forceとgradientを直接比較しない。

## PySCF/GPU4PySCF側の重要設定

ωB97M-VはVV10非局所相関を含む。PySCFでは通常DFT格子`grids`とは別に
`nlcgrids`が存在するため、両者のlevelをmanifestで明示する。

今回の初期値:

- `grids.level = 5`
- `nlcgrids.level = 5`
- `conv_tol = 1e-10`
- `max_cycle = 200`
- `grid_response = True`
- 最初の診断laneではdensity fittingを使用しない

GPU4PySCFはωB97M-V、VV10、RKS/UKS、解析勾配を公式にサポートしている。
近年のreleaseではVV10 gradientとgrid responseに関する修正も続いているため、
GPU環境ではバージョンを必ず固定・記録する。

公式資料:

- [PySCF DFT and VV10](https://pyscf.org/user/dft.html)
- [GPU4PySCF repository](https://github.com/pyscf/gpu4pyscf)
- [GPU4PySCF releases](https://github.com/pyscf/gpu4pyscf/releases)

## ORCA側の初期診断条件

初期H2入力は次の方針で生成する。

```text
WB97M-V def2-TZVPD EnGrad
VeryTightSCF DEFGRID3 SCNL NORI NOCOSX NoAutoStart
```

- `SCNL`: VV10非局所相関をself-consistentに扱う。
- `DEFGRID3`: ORCAの高品質数値格子。
- `NORI NOCOSX`: 最初の診断ではRI/COSX近似を外す。
- `NoAutoStart`: 古い軌道ファイルの意図しない再利用を避ける。

ORCAではhybrid DFTにRIJCOSXが既定で使われる。これは高速で実用的だが、最初から
有効にすると「コード差」と「RI/COSX近似差」を分離できない。そのため最初は
近似なしで比較し、次にRIJCOSXを有効にしたproduction候補laneを追加する。

注意点として、この初期条件は比較を診断しやすくするためのものであり、OMol25の
ORCA production recipeを再現したものではない。後続調査でOMol25はORCA 6.0.0、
RI-J/COSX、DEFGRID3、tight convergence、`thresh=1e-12`、`tcut=1e-13`と確認した。
実際の公開`orca.inp`を取得し、SCNL/NLを含む完全な入力を照合してからproduction
recipeを固定する。

公式資料:

- [ORCA ωB97M-V and DFT](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/DensityFunctionalTheory.html)
- [ORCA Numerical Integration](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/numericalintegration.html)
- [ORCA RI/COSX](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/RI.html)

## ローカル検証結果

### コード検証

- Python構文検査: 合格
- PBS shell構文検査: 合格
- unittest: 9件合格
- PySCFをimportしないdry-run: 合格
- ORCA入力生成: 合格
- `.engrad` fixture parsing: 合格
- 同一fingerprint強制と比較gate: 合格

### CPU PySCFスモーク

ujilabへの投入前に、ローカルに存在した別プロジェクト用環境を使用してH2だけを
実行し、runnerの接続を確認した。この値は本番benchmarkではなく、コード経路の
スモーク結果としてのみ扱う。

環境:

- PySCF 2.13.0
- LibXC 7.0.0
- CPU 1 thread
- RKS ωB97M-V/def2-TZVPD
- `grids.level = 5`
- `nlcgrids.level = 5`
- `grid_response = True`

結果:

- SCF収束: yes
- energy: `-1.1607679696541828 hartree`
- H0 z-gradient: `+0.0010727773913136174 hartree/bohr`
- H1 z-gradient: `-0.0010727773913133953 hartree/bohr`
- wall time: 約44.2秒（再実行値。初回は約45.2秒）

生成結果は`runs/`以下にあり、Git追跡対象外である。

## 現時点の未解決点

1. 公開`orca.inp`との照合によるOMol25の完全なproduction入力。
2. ORCA実行ファイルをujilabまたは別CPU機のどこに置くか。
3. ORCA licenseと利用条件、および自動バッチ実行可能なinstallation形態。
4. GPU機のGPU型、CUDA、CuPy、cuTENSOR、PySCF、GPU4PySCFの互換構成。
5. CPU PySCF 2.14とGPU側PySCF/GPU4PySCFの対応version。
6. density fittingを教師データproductionで使うか。
7. 最終的なenergy/gradient許容値。現在のmanifest値は暫定である。
8. ORCAとPySCFの絶対energy差が残る場合の汎関数実装・格子・基底・積分threshold分解。

## 次の実行順

1. このrepositoryをujilabにcloneまたはpullする。
2. ujilab login nodeではdry-runだけを行う。
3. `run_pyscf_cpu_pbs.sh`でH2のCPU PySCFジョブを投入する。
4. GPU機でCUDA互換性を確認し、同一manifestをGPU4PySCFで実行する。
5. ORCAのinstallation先を決め、CPUノードで同一manifestを実行する。
6. CPU PySCF↔GPU4PySCFを先に比較する。
7. CPU/GPUが一致してからORCAとの差を調べる。
8. H2を通過後、H2O、CH3、O2、H2O+、OH−へ拡張する。
9. 安定したノウハウだけをSkillへ追加する。

## Skillへの昇格候補

現時点では検証コード自体をSkillへ入れない。次が複数ケースで確認された時点で、
短い手順または決定論的検査としてSkillへ昇格する。

- multiplicityからPySCF spinへの変換と電子数parity検査
- ωB97M-Vで`grids`と`nlcgrids`を両方固定する規約
- VV10へのD3/D4二重付与を防ぐ規約
- ORCA driverを`mpirun`で起動しない規約
- force/gradientの符号と単位変換
- cross-code結果fingerprintと比較gate
