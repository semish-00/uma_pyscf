# fairchem GPU環境固定記録

- 日付: 2026-08-31
- 実行先: SoftBank AIデータセンター、A100-SXM4-80GB
- 用途: base UMA評価とengineering-only overfit smoke
- 状態: GPU runtimeとPython環境を固定済み、UMA checkpoint認証だけ手動操作待ち

## 固定環境

| 項目 | 固定値 |
|---|---|
| container | `nvcr.io/nvidia/pytorch:26.07-py3` |
| image path | `/lustre/user140002/containers/nvidia-pytorch_26.07-py3.sqsh` |
| image SHA-256 | `b7fda1fe99974e5901c48e3e3bfaac0c4349384ac3285339a85c4885273a6a20` |
| Python environment | `/lustre/user140002/python/fairchem-2.22-py312-v2` |
| freeze | `/lustre/user140002/python/fairchem-2.22-py312-v2.freeze.txt` |
| Python | 3.12.3 |
| fairchem-core | 2.22.0 |
| ASE | 3.26.0 |
| torch | 2.13.0+cu130 |
| torch CUDA runtime | 13.0 |
| GPU / driver | A100-SXM4-80GB / 535.161.08 |

Python環境はcontainerのsystem packageを継承しない独立venvとした。`pip check`は
`No broken requirements found`で、CUDA上のmatrix multiplicationまで実行した。最初に作った
`fairchem-2.22-py312-v1`はNGC内蔵alpha版torchvisionとstable torchの依存が衝突したため
採用しない。再現実行には必ず`v2`を使う。

## 実機probe

| Slurm job | 内容 | 結果 |
|---:|---|---|
| 1797343 | NGC 26.07 container、Python、torch、CUDA、GPU演算 | `COMPLETED` |
| 1797355 | clean venv、`pip check`、fairchem/ASE import、GPU演算 | `COMPLETED` |
| 1797360 | pretrained API、`uma-s-1p2`登録、HF token有無 | `COMPLETED` |

job 1797360では`uma-s-1p2`がfairchemの利用可能modelとして登録済みであることを確認した。
一方、Hugging Face tokenは未設定だった。checkpoint repositoryはgatedであり、利用者本人が
license条件を確認・承諾し、read tokenを端末から`hf auth login`へ入力する。tokenはチャット、
Slurm script、stdout/stderr、環境inventory、Gitへ保存しない。

## 次の実行

認証後、固定dataset `ds_sigehcl_001`に対して
`scripts/slurm/run_base_uma_engineering_50_softbank_slurm.sh`を投入する。評価artifactには
dataset manifest/split、全checkpoint cache fileのSHA-256、container SHA-256、package version、
GPU、Git commit、50件のpredictionとpartition別metricを保存する。

これはengineering evidenceであり、scientific releaseやproduction fine-tuningの承認ではない。
科学閾値とstate registry承認が完了するまでdecision 0004のfail-closedを維持する。
