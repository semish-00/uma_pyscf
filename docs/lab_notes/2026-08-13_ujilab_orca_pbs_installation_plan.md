# ujilab向けORCA 6.0.0導入・PBS実行設計

- 日付: 2026-08-13
- 状態: ORCA/OpenMPI導入・PBS H2スモーク完了
- 対象: ORCA 6.0.0 / OpenMPI 4.1.6 / OpenPBS 23.06.06

## 決定

ujilabでは次の構成を採用する。

1. ORCA 6.0.0本体を`/home/seki/uma_pyscf/software/orca/6.0.0`へユーザー導入する。
2. OpenMPI 4.1.6を`/home/seki/uma_pyscf/software/openmpi/4.1.6`へユーザー導入する。
3. ORCAとOpenMPIの版をディレクトリで固定し、globalな`orca`や`mpirun`に依存しない。
4. OpenMPIのcompileはPBS計算ノード上のローカル`/tmp`で行う。
5. ORCA実行ファイルは共有NFS上から絶対パスで起動する。
6. 計算のworking directoryは計算ノード固有の`/tmp`とする。
7. 最初はsingle-nodeだけを使い、4、8、16 processの実測後に標準値を決める。

ORCA installer、展開済みORCA、raw outputはGitで追跡しない。

## `mpirun`を直接書かない理由

ORCAの並列実行にはOpenMPIが使われるが、PBS scriptから起動するのはserialな
ORCA driver 1個である。

```text
PBS job
  └─ /full/path/orca input.inp
       └─ input中の %pal nprocs N を読む
            └─ ORCAがOpenMPIで並列moduleをN process起動
```

したがって正しい起動は次である。

```bash
/home/seki/uma_pyscf/software/orca/6.0.0/orca input.inp "--bind-to core"
```

次は誤りである。

```bash
mpirun -np 4 /home/seki/uma_pyscf/software/orca/6.0.0/orca input.inp
```

後者はdriver自体を4個起動する形になる。ORCA 6.0公式manualもdriverを
`mpirun`で起動しないよう明記している。一方でOpenMPI runtimeは不要という
意味ではなく、driverが内部で並列moduleを起動するために必要である。

## ujilabで追加確認した事実

- Ujilab1/3: Intel Xeon Gold 6338、64物理core、SMTなし
- 各node: 2 socket、32 core/socket、2 NUMA node
- `/home`: login nodeからNFS共有、約118 TB空き
- `/tmp`: 各計算nodeのlocal ext4、約741 GB空き
- GCC/G++/GFortran 9.4.0、GNU Make 4.2.1
- `/usr/openpbs/include/tm.h`と`/usr/openpbs/lib/libpbs.so`が存在
- OpenPBSでは`select=1:ncpus=8:mpiprocs=8`形式の既存稼働例あり
- system側の`mpirun`はIntel MPI 2021.10
- OpenMPI module/packageはなし

共有homeへの一回のbinary loadより、頻繁で大容量なORCA temporary I/OをNFSに
流さないことの方が重要である。このためORCA/OpenMPIはNFS上に一度だけ導入し、
job scratchだけをlocal `/tmp`へ置く。

## インストール方式の比較

| 方式 | 評価 | 理由 |
|---|---|---|
| `/home/seki/opt`へ版固定 | 採用 | root不要、全計算nodeから同じ内容が見える |
| 各nodeの`/opt`へ個別導入 | 不採用 | root依存、node間drift、Ujilab2 down時の保守が複雑 |
| condaのOpenMPI | 不採用 | ORCAが要求するruntimeとのABI・plugin混在リスク |
| system Intel MPIを流用 | 不可 | OpenMPI buildのORCAとは異なるMPI実装 |
| ORCAをjobごとに`/tmp`へ全コピー | 当面不採用 | 毎回の転送とNFS負荷が増える。実測でbinary loadが問題なら再検討 |
| ORCA working filesをhomeに保存 | 不採用 | NFS I/O競合と大量temporary fileの原因 |

## PBS資源設計

初期H2は次を揃える。

```text
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=16gb
%pal nprocs 4 end
%maxcore 2000  # MB/process
OMP/MKL/OPENBLAS threads = 1
```

`%maxcore`はjob全体ではなく1 processあたりである。H2設定ではMaxCore予算は
`4 × 2000 = 8000 MB`で、PBSの16 GB要求に余裕がある。一般ケースでも
`nprocs × maxcore`をPBS memoryの70～75%以下に置くことを本プロジェクトの
安全側の初期規約とし、ORCAの`%scf DryRun true`によるprocess単位memory見積もりも
利用する。

ORCA公式manualはRI-DFTでは概ね16 processまでが有効な目安で、それ以上では
overheadが大きくなると説明している。ωB97M-Vはhybridだが、まず4/8/16で測定し、
64 core一括使用はベンチマークで有利と確認されるまで行わない。

## 実装した安全策

`run_orca_cpu_pbs.sh`に次を実装した。

- modern OpenPBSの`select/ncpus/mpiprocs/mem`指定
- `ORCA_ROOT`と`OPENMPI_ROOT`の版固定default
- `/usr/bin/orca`誤使用の明示的拒否
- OpenMPIが厳密に4.1.6か確認
- ORCAの未解決shared libraryを`ldd`で検出
- manifestにORCA 6.0.0を固定し、output記載versionとの差をparserで拒否
- manifestの`nprocs`が`PBS_NCPUS`を超えないことを確認
- BLAS/OpenMP threadを1に固定
- node-local `/tmp`に一意なjob scratchを作成
- ORCA driverへ`--bind-to core`を渡す
- 成否にかかわらず主要outputを共有homeへ回収
- 検証済みpathだけをcleanup
- `KEEP_SCRATCH=1`による診断時の一時保持

OpenMPI build用PBS scriptには次を実装した。

- 公式OpenMPI 4.1.6 tar.gzのSHA-256固定
- 計算nodeのlocal `/tmp`でcompile
- GCC/G++/GFortranを明示
- `--with-tm=/usr/openpbs`でOpenPBS/TM連携を組み込み
- non-emptyなinstall prefixの上書き拒否
- version確認

## 追加・更新ファイル

- `validation/orca_gpu4pyscf/jobs/run_orca_cpu_pbs.sh`
- `validation/orca_gpu4pyscf/jobs/build_openmpi_416_pbs.sh`
- `validation/orca_gpu4pyscf/jobs/test_openmpi_416_pbs.sh`
- `validation/orca_gpu4pyscf/setup/ujilab.md`

## 実施結果（2026-08-13）

ORCA Forum公式配布から利用者が取得した次のinstallerを使用した。

```text
orca_6_0_0_linux_x86-64_avx2_shared_openmpi416.run
SHA-256: 5dcdf31f01ec92d09d7f60c6c677c74b868d4113ce046fdbee55e0c8580f1b6f
```

macOSのdownload metadataが公式Forumの該当file-download URLを示すこと、Makeself
内蔵MD5検査に合格すること、ujilab転送後のSHA-256がlocalと一致することを確認した。
installerと展開物はいずれも所有者だけがアクセスできる権限にした。

`setup`はshell startup fileへPATHを追記する実装だったため実行せず、`--noexec`で
次へ展開した。PBS runnerが絶対pathと環境変数を毎回設定するため、利用者の
`~/.bashrc`等は変更していない。

```text
/home/seki/uma_pyscf/software/orca/6.0.0
```

代表的なMPI module `orca_autoci_mpi`は、次へ正しくlinkした。

```text
libmpi.so.40     => /home/seki/uma_pyscf/software/openmpi/4.1.6/lib/libmpi.so.40
libopen-rte.so.40 => /home/seki/uma_pyscf/software/openmpi/4.1.6/lib/libopen-rte.so.40
libopen-pal.so.40 => /home/seki/uma_pyscf/software/openmpi/4.1.6/lib/libopen-pal.so.40
```

未解決shared libraryはなかった。

PBS job `1347.Ujilab`をUjilab1へ4 processで投入した。ORCA計算自体は4.048秒で
normal terminationし、次を得た。

```text
ORCA version: 6.0.0
energy: -1.160767976664 Eh
gradient z (H1/H2): +0.001072511417 / -0.001072511417 Eh/bohr
```

jobのexit status 1は量子化学計算の失敗ではなく、実際のORCA 6.0.0 `.engrad`が
座標block後の`The end of the file` markerを持たないのに、fixture由来のparserが
それを必須にしていたためである。parserをEOF終端にも対応させ、回帰testを追加した。
local/ujilabのPython 3.11で全11 testが合格し、既存出力から`result.json`を生成した。
修正後のend-to-end確認としてPBS job `1348.Ujilab`を再投入し、wall time 4秒、
exit status 0、ORCA 6.0.0 normal termination、`result.json`生成まで完了した。

## 次のgate

以下を順に進める。

1. 同一H2をCPU PySCFで実行し、ORCAとのenergy/gradient差を比較する。
2. ORCA 1 processと4 processの再現性・timingを比較する。
3. 小分子test setへ拡張する。
4. GPU4PySCFを同一protocolへ接続する。
5. 代表的な中規模系で4/8/16 process scalingを測る。

## ローカル検証

- 2本のPBS shell script: `bash -n`合格
- cross-code unittest: 11件合格
- ORCA input dry generation: 合格
- ORCA 6.0.1を6.0.0 manifestへ混入させたfixture: 意図どおり拒否

## 参考資料

- [ORCA 6.0 serial and parallel execution](https://www.faccts.de/docs/orca/6.0/manual/contents/calling.html)
- [ORCA 6.0 energy/gradient and memory estimation](https://www.faccts.de/docs/orca/6.0/manual/contents/typical/energygradients.html)
- [OpenMPI 4.1 official releases](https://www.open-mpi.org/software/ompi/v4.1/)
- [OpenMPI runtime-system configure options (`--with-tm`)](https://docs.open-mpi.org/en/main/installing-open-mpi/configure-cli-options/runtime.html)
- [OMol25 calculation details](https://arxiv.org/html/2505.08762)
