# SoftBank GPUサーバー接続・運用メモ

- 日付: 2026-08-30
- SSH alias: `sb-gpu`
- 状態: SSH、Git、Slurm、GPU4PySCF・fairchem固定containerまで実機確認済み
- 認証情報: この文書には保存しない

## 接続

ローカル端末から次のaliasだけを使用する。

```bash
ssh sb-gpu
```

`sb-gpu`は、必要に応じてアクセスサーバーへの`sb-tunnel`を起動し、ローカル転送を
経由してログインサーバーへ接続するよう`~/.ssh/config`で設定済みである。通常は既存の
トンネルとSSH認証状態が再利用されるため、パスワードやOTPを求められず接続できる。

認証が失効している場合は、Codexが対話SSHを開始し、パスワードまたはOTPの入力が
必要になった時点でユーザーが端末へ直接入力する。認証情報をチャット、ログ、スクリプト、
環境変数、リポジトリへ書かない。認証失敗を自動反復せず、1回失敗したら入力内容と
接続状態をユーザーが確認する。

## 確認済み環境

2026-08-30に次を確認した。

| 項目 | 確認結果 |
|---|---|
| ログインサーバー | `fcpv00334` |
| ユーザー | `user140002` |
| 専用partition | `140-partition` (`up`, `idle`) |
| GPU node | `fcdgx00081` |
| GPU | NVIDIA A100-SXM4-80GB x 8 |
| NVIDIA driver | `535.161.08` |
| Slurm | `23.02.7` |
| container runtime | Enroot/Pyxis |
| main storage | `/lustre`、契約領域500 GB |
| home storage | `/home/user140002`、20 GB |
| repository | `/lustre/user140002/uma_pyscf` |
| checkout | `claude/plan-review-implementation-iwwtp6`（計算前に`pull --ff-only`） |

GitHub HTTPSへの接続、対象branchのclone、Slurmによる1 GPU割当、公式CUDA containerの
Pyxis import、container内からのGPUとrepositoryの可視性を確認した。確認ジョブ3件は
すべて`COMPLETED`、exit code `0:0`だった。

## 再接続時の最小確認

ログインサーバーでは状態確認だけを行い、GPU計算を直接実行しない。

```bash
cd /lustre/user140002/uma_pyscf
git status --short --branch
sinfo --noheader -o '%P|%a|%l|%D|%t|%G|%N'
squeue
```

GPU割当を再確認する必要がある場合だけ、短いSlurm jobを使う。

```bash
srun -p 140-partition -N1 -n1 --gpus=1 --time=00:02:00 \
  nvidia-smi --query-gpu=index,name,driver_version,memory.total \
  --format=csv,noheader
```

`defq`は利用不可である。すべての`sbatch`/`srun`で`-p 140-partition`を明示し、
GPU数とtime limitも必ず指定する。

Slurm clientはlogin shellの環境初期化に依存する。`ssh sb-gpu "sbatch ..."`のような
非対話commandでは`sbatch`がPATHになく、実体pathを直接指定してもSlurm設定のDNS解決に
失敗した。Codexからの投入は対話PTYで`ssh sb-gpu`へ入り、そのpromptで`sbatch`を実行する。

## Repository更新

GPU側checkoutは計算実行用のcleanなconsumerとして扱う。通常の更新は次で行う。

```bash
cd /lustre/user140002/uma_pyscf
git status --short --branch
git pull --ff-only
```

GPU側で開発commitやforce操作を行わない。ローカルで実装・test・commit・pushし、
GPU側は`pull --ff-only`で同期する。

UMA評価用の固定環境は
[fairchem GPU環境固定記録](2026-08-31_fairchem_gpu_environment.md)と
[P2.7 base UMA評価運用](../operations/p2_7_uma_baseline.md)を参照する。Hugging Face tokenは
認証端末へ利用者が直接入力し、この接続メモやrepositoryには保存しない。

## Storageとcontainerの運用

- repository、container image、入力、永続結果、logは`/lustre/user140002`に置く。
- `/home`にはSSH・shell設定など小さなファイルだけを置く。
- GPU nodeの`/raid`はjob中の一時scratchに限定し、job終了前に必要な成果物を
  `/lustre`へ戻す。
- 計算はSlurmとEnroot/Pyxis containerを経由する。
- containerは`latest`を避け、tagまたはdigestを固定する。
- job scriptにはpartition、GPU数、CPU数、time limit、stdout/stderr、
  container image、mount、実行commitを明記する。
- このpartitionはSlurmのmemory TRESが`1M`として設定されているため、
  `--mem` / `--mem-per-cpu`を指定しない。

## 現在の実行段階

[Part I GPU4PySCF検証計画](../plans/01_gpu4pyscf_validation_plan.md)のA1/A2/C0/C1と
C3 one-axis matrixとC4 29-case ladderは完了した。final C4 Slurm job `1796359`は
29/29件初回成功、CPU direct比20.96xだった。実行記録は
[GPU4PySCF C3実行記録](2026-08-30_gpu4pyscf_c3.md)と
[GPU4PySCF C4実行記録](2026-08-30_gpu4pyscf_c4.md)を参照する。
