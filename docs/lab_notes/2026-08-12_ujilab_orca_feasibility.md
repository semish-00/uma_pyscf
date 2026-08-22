# ujilabでのORCA並列計算実現性調査

- 日付: 2026-08-12
- 状態: 読み取り専用の環境調査完了、ORCA未導入
- 対象ホスト: `ssh ujilab`

## 結論

ujilabのCPU並列計算機上で量子化学ORCAを実行することは、ハードウェアと
ジョブスケジューラの面では現実的である。ただし、現時点では量子化学ORCAと
対応するOpenMPIが導入されていないため、まだ計算は実行できない。

本プロジェクトではORCA計算が必要である。GPU4PySCFだけでもUMAの追加学習用
データは生成できるが、それだけではOMol25のORCAラベルと同等であることを示せない。
まず少数のORCAアンカー計算を実行し、CPU PySCF・GPU4PySCF・ORCAの三角比較を
成立させる。差が十分に小さく安定していると確認できた後は、全教師点をORCAで
二重計算せず、定期的なスポット検証に絞る余地がある。

## 最重要の発見: `/usr/bin/orca`は別ソフト

ujilabには`/usr/bin/orca`が存在するが、これは量子化学計算プログラムではなく、
UbuntuのスクリーンリーダーOrcaである。

確認結果:

- `/usr/bin/orca`: Python script
- 所有パッケージ: Ubuntu package `orca`
- `orca --version`: GUI/display関連エラーとなり、量子化学ORCAの出力ではない
- `/home/seki`、`/opt`、`/usr/local`の限定探索では、別の実行可能な量子化学
  `orca`は見つからなかった

これは誤実行しやすいので、量子化学ORCA導入後も常に実行ファイルの絶対パスを
使う。例えば`ORCA_EXECUTABLE=/home/seki/opt/orca/6.0.0/orca`のように明示し、
単なる`orca`コマンドには依存しない。ORCA公式資料も、並列計算ではdriverを
絶対パスで起動するよう求めている。

## OMol25再現に必要な版

OMol25論文の2026年3月4日改訂版には、全計算を**ORCA 6.0.0**で行ったと明記
されている。したがって、クロスコード再現の基準版はORCA 6.0.0とする。

同論文で確認できる主要条件:

- ORCA 6.0.0
- LibXC実装のωB97M-V
- def2-TZVPD
- RI-JおよびCOSX
- tight convergence
- DEFGRID3
- `thresh = 1e-12`
- `tcut = 1e-13`
- 非一重項はUKS
- 遷移金属・ランタノイド錯体や結合解離が想定される一重項の一部もUKS
- 上記一重項ではβ HOMO–LUMOを20度回転させたbroken-symmetry初期推測

最新ORCAを利用した別laneを設けることはできるが、6.0.0の代わりにはしない。
版更新による数値差を混ぜないため、`6.0.0-reproduction`と`current-diagnostic`を
別環境・別fingerprintとして扱う。

参考資料:

- [OMol25論文: Calculation Details](https://arxiv.org/html/2505.08762)
- [OMol25 electronic-structure公開資料](https://github.com/facebookresearch/fairchem/blob/main/docs/molecules/datasets/omol25_elec.md)

## ujilabで確認した計算環境

### PBSと計算ノード

- OpenPBS 23.06.06
- queue: `workq`（enabled、started、default queue）
- `qsub`の絶対パス: `/usr/openpbs/bin/qsub`
- Ujilab1: free、64 CPUs、約252 GB memory
- Ujilab3: free、64 CPUs、約252 GB memory
- Ujilab2: state-unknown/down
- GPU resource: なし
- 調査時点のqueueとユーザーjob: なし

ORCAはこのCPUノードで実行する。初期検証は1ノード4コアから始め、8、16コアの
実測スケーリングを取る。RI-DFT/RIJCOSXでは通信とメモリ帯域の影響があるため、
最初から64コアを使うとは限らない。まずsingle-nodeに限定する。

### OS、CPU、ストレージ

- Linux x86_64、kernel 5.15
- login node CPU: Intel Xeon Silver 4208
- AVX2、AVX-512対応
- glibc 2.31
- `/home`: 約121 TB、約118 TB available、使用率3%
- `/tmp`: 存在し書き込み可能
- login nodeでは`/scratch`、`/work`、`/local_scratch`は見つからなかった

容量は十分である。一方、ORCAの一時ファイルは大きくなる可能性があるため、
共有homeへ直接ばらまかず、PBS job内で`$TMPDIR`または計算ノードの`/tmp`を使う。
計算ノードでの`$TMPDIR`の実体、空き容量、自動削除挙動は最初の短時間PBS probeで
確認する必要がある。

### Python/PySCF

- Python: `/home/seki/miniconda3/envs/qcthermo/bin/python`
- PySCF: 2.14.0

CPU PySCF側は既に利用可能であり、ORCA導入前でもH2の再計算とrunner検証を進められる。

### MPI

- OpenMPI package/module: 見つからない
- environment-modules: 見つからない
- Intel MPI 2021.10: `/opt/intel/oneapi/mpi/2021.10.0/bin/mpirun`

Linux版ORCA 6.0の並列実行はOpenMPI buildを使用する。Intel MPIは、OpenMPI向けに
動的リンクされたORCAの代替として流用しない。ORCA 6.0公式installerの例は
`orca_6_0_0_linux_x86-64_shared_openmpi416.run`であり、OpenMPI 4.1.6 buildである。
導入後に`ldd`と短いparallel smokeで、実際に読み込まれるMPI libraryを確認する。

参考資料:

- [ORCA 6.0 manual: installer example](https://www.faccts.de/docs/orca/6.0/manual/_downloads/f5a34d500b44971b2b057d96d7f899ca/orca.pdf)
- [ORCA 6.0 manual: serial and parallel execution](https://www.faccts.de/docs/orca/6.0/manual/contents/calling.html)

## 導入にroot権限は不要

公式installerはユーザー指定ディレクトリへ導入できるため、`/home/seki`配下への
ユーザーインストールで進められる見込みである。候補構成は次のとおり。

```text
/home/seki/opt/orca/6.0.0/
/home/seki/opt/openmpi/4.1.6/   # installerにruntimeが含まれない場合
```

ただし、ORCAのdownloadには登録と利用条件への同意が必要である。installerを
repositoryへ入れたり第三者へ再配布したりしない。プロジェクトの所属・用途、
共同研究や商用利用の有無に対して許諾範囲が適合するかを、導入前に利用者側で
確認する。生成した計算結果を教師データとして使うことについても、同じ確認事項に
含める。

参考資料:

- [ORCA 6.1公式installation guide](https://www.faccts.de/docs/orca/6.1/manual/contents/quickstartguide/installation.html)
- [ORCA forum terms / EULA](https://orcaforum.kofo.mpg.de/app.php/privacypolicy/policy)

## 推奨する導入・検証順

1. ORCAの利用条件が本プロジェクトに適合するか確認する。
2. 正規アカウントからLinux x86-64/OpenMPI 4.1.6版ORCA 6.0.0 installerを取得する。
3. installerはGit外の非公開領域からujilabへ転送する。
4. `/home/seki/opt/orca/6.0.0`などにユーザーインストールする。
5. `ldd`で欠落libraryとOpenMPI ABIを確認する。
6. 必要ならOpenMPI 4.1.6をユーザー領域に導入し、ORCA jobだけにPATHと
   `LD_LIBRARY_PATH`を限定して設定する。
7. login nodeではversion確認とinput生成だけを行う。
8. PBSで1コアH2 single point/gradientを実行する。
9. PBSで4コアH2を実行し、driverは絶対パスで直接起動する。driver自体を
   `mpirun`で包まない。
10. CPU PySCF結果と比較後、4/8/16コアでH2Oなどの短いスケーリング試験を行う。
11. closed-shell、open-shell、charged、metal-containingへ順に拡張する。

## 判定

| 項目 | 判定 | 備考 |
|---|---|---|
| CPU資源 | 可 | freeな64 CPU/約252 GBノードが2台 |
| PBS実行 | 可 | OpenPBS 23.06.06、`workq`稼働中 |
| 保存容量 | 可 | homeに十分な空き容量 |
| 量子化学ORCA | 未導入 | `/usr/bin/orca`はスクリーンリーダー |
| ORCA 6.0.0再現 | 導入後に可 | 正規installerが必要 |
| 並列runtime | 未準備 | Intel MPIのみ。OpenMPI 4.1.6を要確認・導入 |
| GPU | 不要 | ORCAはCPU、GPU4PySCFは別GPU機で実行 |
| node-local scratch | 要probe | ORCA導入後の最初のPBS jobで確認 |
| ライセンス適合性 | 要利用者確認 | download・導入前のgate |

総合判定は**条件付きで実行可能**である。障害は計算機性能ではなく、ORCA 6.0.0の
正規installer、利用条件の確認、およびOpenMPI 4.1.6 runtimeの準備に限られる。

## この調査で変更していないもの

- ujilab上のファイル
- ujilabの環境設定
- PBS queueおよびjob
- package/module構成

今回のSSH調査は読み取り専用で行った。
