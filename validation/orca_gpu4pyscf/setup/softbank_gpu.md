# SoftBank AIデータセンター GPU4PySCF環境

この文書は、Part I Workstream A1/A2をSoftBank AIデータセンターのA100 nodeで
再現する手順を記録する。一般的な接続方法と運用規約は
[`docs/lab_notes/2026-08-30_softbank_gpu_connection.md`](../../../docs/lab_notes/2026-08-30_softbank_gpu_connection.md)
を参照する。

## 固定する層

| 層 | 固定値 |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| host driver | 535.161.08 |
| base container | `nvcr.io/nvidia/pytorch:23.10-py3` |
| container CUDA | 12.2 |
| container Python | 3.10.12 |
| Python direct requirements | `requirements-gpu-cu122.in` |
| Python package directory | `/lustre/user140002/python/gpu4pyscf-cu122-py310-v1.8.1` |

Remote tagをjobごとに指定するとPyxisのimportが繰り返されるため、containerは一度だけ
Enroot squashfsへ変換し、checksumと共に`/lustre`へ保存する。Python packageは
base containerへ書き込まず、versionを含むimmutable directoryへ`pip --target`で
導入する。setup jobは既存directoryを上書きしない。

## 1. Repositoryとdirectory

```bash
ssh sb-gpu
cd /lustre/user140002/uma_pyscf
git status --short --branch
git pull --ff-only
mkdir -p \
  /lustre/user140002/containers \
  /lustre/user140002/python \
  /lustre/user140002/logs \
  /lustre/user140002/artifacts
```

## 2. Containerの固定

ログインサーバーで一度だけ実行する。

```bash
enroot import \
  -o /lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh \
  docker://nvcr.io#nvidia/pytorch:23.10-py3
sha256sum /lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh \
  > /lustre/user140002/containers/nvidia-pytorch_23.10-py3.sqsh.sha256
```

## 3. Python package directory

このクラスタでは Slurm のメモリ TRES が `1M` として構成されているため、
`--mem` / `--mem-per-cpu` は指定しない。指定すると、実メモリの空き状況とは
無関係に `Memory specification can not be satisfied` で投入が拒否される。

NGC PyTorch 23.10 の既存パッケージとの整合性を保つため、overlay側のNumPyは
`1.24.4`に固定する。NumPy 2.xではベースイメージ内のNumba/Matplotlibに対する
`pip check`が失敗する。

```bash
sbatch validation/orca_gpu4pyscf/jobs/setup_gpu4pyscf_softbank_slurm.sh
```

job IDを保存し、`squeue`と`/lustre/user140002/logs/uma-gpu-env-<job-id>.{out,err}`を
確認する。成功すると完全な解決versionが次へ保存される。

```text
/lustre/user140002/python/gpu4pyscf-cu122-py310-v1.8.1.lock.txt
```

## 4. A1/A2/C0

```bash
sbatch validation/orca_gpu4pyscf/jobs/run_gpu_smoke_softbank_slurm.sh
```

成果物はjob IDごとのdirectoryへ保存される。

```text
/lustre/user140002/artifacts/gpu4pyscf-a1-a2/<job-id>/
  environment.yaml
  gpu_smoke_check.json
  full_ladder_dry_run.log
  git_commit.txt
  git_status.txt
```

`gpu_smoke_check.json`の全checkが`passed`で、29-case dry runが
`dry_run_ok`になるまで5-case実計算へ進まない。

## 5. C1 five-case smoke

H2、SiH4、SiCl4、SiH3、H3Si-GeCl3をsuite記載順に実行し、最初の失敗で停止する。

```bash
sbatch validation/orca_gpu4pyscf/jobs/run_gpu_c1_softbank_slurm.sh
```

session summaryと実行時runtime fileのchecksumは次に保存される。各caseの正規化結果と
attempt ledgerはGit追跡外の`validation/orca_gpu4pyscf/runs/`に保存される。

```text
/lustre/user140002/artifacts/gpu4pyscf-c1/<job-id>/
```

## 6. C3 one-axis setting matrix

4代表系についてordinary grid、VV10 grid、density fittingを一軸ずつ変更した20件を
順次実行する。

```bash
python validation/orca_gpu4pyscf/generate_c3_matrix.py
sbatch validation/orca_gpu4pyscf/jobs/run_gpu_c3_softbank_slurm.sh
```

session summaryは次に保存される。

```text
/lustre/user140002/artifacts/gpu4pyscf-c3/<job-id>/session.json
```

候補は診断用で、matrix完走だけではproduction protocolへ昇格しない。特にdensity
fittingは同一設定のCPU PySCF結果と比較し、近似誤差とGPU port差を分離する。

## 7. C4 conditional full ladder

C3で選んだdensity-fitting + explicit MINAO条件付き候補を、direct source manifestを
変更せず29件へ展開する。generatorは初期runner履歴用suiteとfinal MINAO suiteを両方生成し、
本番候補の判定には`gpu_c4_density_fit_minao_ladder_v1.json`を使う。

```bash
python validation/orca_gpu4pyscf/generate_c4_density_fit_suite.py
sbatch \
  --export=ALL,SUITE_NAME=gpu_c4_density_fit_minao_ladder_v1.json,ARTIFACT_BASE=/lustre/user140002/artifacts/gpu4pyscf-c4-minao \
  validation/orca_gpu4pyscf/jobs/run_gpu_c4_softbank_slurm.sh
```

```text
/lustre/user140002/artifacts/gpu4pyscf-c4-minao/<job-id>/session.json
```

29件の完走だけでproduction freezeとはしない。既存のdirect CPU/ORCA結果との数値差、
C3 relative-energy結果、失敗率、runtimeをまとめてGate 1で判断する。

## 8. 失敗時

`gpu_smoke_check.py`は最初に壊れた層で後続checkをskipする。次の順で一軸ずつ直す。

1. CuPy import
2. SlurmからのGPU可視性
3. CuPy kernel
4. PySCF import
5. GPU4PySCF import
6. RKS ωB97M-V/VV10 gradient
7. UKS ωB97M-V/VV10 gradientと`<S^2>`

既存のpackage directoryを直接変更しない。versionを変更する場合は新しいdirectory名、
requirements、lock、inventoryを作成する。
