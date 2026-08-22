# ujilabへのOpenMPI 4.1.6導入とORCA導入準備

- 日付: 2026-08-13
- 作業ルート: `/home/seki/uma_pyscf`
- 状態: OpenMPI導入・PBSスモーク完了、ORCA公式installer待ち

## 結論

ORCA 6.0.0 `shared_openmpi416`版の実行基盤として、OpenMPI 4.1.6をujilabの
ユーザー領域へ導入し、OpenPBS上で4 rankが正常起動することを確認した。

```text
/home/seki/uma_pyscf/
├── installers/                    # Git追跡外、mode 700
│   └── openmpi-4.1.6.tar.gz       # mode 600
├── software/                      # Git追跡外
│   ├── openmpi/4.1.6/
│   └── orca/                      # ORCAは未導入
├── validation/
└── docs/
```

## ujilabマニュアルとの照合

利用者提供の`並列計算機マニュアル.pdf`全31ページを確認した。今回に関係する
記載は次のとおり。

- 標準MPIはIntel MPIである。
- PBSのMPI例は`nodes=1:ppn=64`を要求して`mpirun -np 64`を実行する。
- OpenMPの場合は`OMP_NUM_THREADS`を割当core数へ合わせる。
- job scriptは`PBS_O_WORKDIR`へ移動して実行する。
- 2022年当時はOpenPBS 20.0.0、64 core/node、約256 GB/nodeと記載されている。

現在の実機はOpenPBS 23.06.06で、Ujilab1/3は各64物理core、約252 GBである。
manualは一般MPI programの起動法を示しているが、ORCAは公式仕様に従いdriverを
`mpirun`で包まない。Intel MPIはORCAのOpenMPI buildには流用しない。

## 既存MPIの再利用調査

新規導入前に次を確認した。

- system/oneAPI: Intel MPI 2021.10
- module command: なし
- dpkg/loader/standard paths: OpenMPIなし
- conda: MPICHのlibraryあり
- `/home/seki/advance_phase/mpi`: `mpiexec.hydra`を要求する別MPI環境で、
  `ompi_info`なし。OpenMPI 4.1.6ではない

したがって、ORCA公式buildに対応するOpenMPI 4.1.6のユーザー導入が必要と判断した。

## 公式sourceの取得

取得元:

```text
https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
```

Open MPI Project公式ページ掲載SHA-256:

```text
44da277b8cdc234e71c62473305a09d63f4dcca292ca40335aab7c4bf0e6a566
```

HTTPSで取得した`.part`ファイルをSHA-256検証し、一致後に正式名へ変更した。
第三者mirrorやpackage repositoryは使用していない。

## OpenMPI build

PBS job:

- job ID: `1344.Ujilab`
- node: Ujilab1
- allocation: 8 CPU、8 GB
- wall time: 9分17秒
- CPU time: 20分03秒
- exit status: 0

主要configure条件:

```text
--prefix=/home/seki/uma_pyscf/software/openmpi/4.1.6
--with-tm=/usr/openpbs
CC=/usr/bin/gcc
CXX=/usr/bin/g++
FC=/usr/bin/gfortran
```

build directoryはUjilab1のlocal `/tmp`を使用し、完了後に削除した。installed
runtimeは次を報告した。

```text
mpirun (Open MPI) 4.1.6
mca:ras:tm:version:"component:4.1.6"
```

## PBS MPIスモーク

初回job `1345.Ujilab`は、ujilabのjob環境に`PBS_NCPUS`が設定されないことを
発見して、MPI起動前にexit 1となった。allocation自体は4 CPUだった。

runnerを`PBS_NODEFILE`の行数へfallbackするよう修正し、再投入した。

- job ID: `1346.Ujilab`
- node: Ujilab1
- allocation: 4 CPU
- wall time: 1秒
- exit status: 0
- PBS nodefile: `ujilab1`が4行
- rank 0/1/2/3: CPU 0/32/1/33へbinding

これによりOpenMPI 4.1.6、OpenPBS/TM integration、single-node 4-rank起動、
core bindingが正常であることを確認した。

## ORCAの状態

FACCTs/ORCA Forum公式配布のORCA 6.0.0 Linux x86-64 AVX2
`shared_openmpi416`版を受領し、SHA-256転送照合後に次へ導入した。

```text
/home/seki/uma_pyscf/software/orca/6.0.0
```

ORCAのMPI moduleが本noteで導入したOpenMPI 4.1.6の`libmpi.so.40`、
`libopen-rte.so.40`、`libopen-pal.so.40`へlinkすることを確認した。PBS job
`1347.Ujilab`でH2のORCA 6.0.0計算が4 process、4.048秒でnormal terminationし、
energyとanalytic gradientを取得した。詳細は
`2026-08-13_ujilab_orca_pbs_installation_plan.md`を参照する。

## 参考資料

- [OpenMPI 4.1公式release](https://www.open-mpi.org/software/ompi/v4.1/)
- [ORCA 6.0 parallel execution](https://www.faccts.de/docs/orca/6.0/manual/contents/calling.html)
- [ORCA Forum](https://orcaforum.kofo.mpg.de/)
