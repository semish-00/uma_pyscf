# P2.7 base UMA評価運用

- 対象model: `uma-s-1p2`
- task: `omol`
- fairchem-core: `2.22.0`
- 対象dataset: verified ASE-LMDB `ds_sigehcl_001`（40 train / 10 holdout）
- 用途: fine-tuning前のengineering baseline

## 初回だけの手動認証

UMA checkpointはHugging Faceのgated repositoryから取得する。利用者本人が
`https://huggingface.co/facebook/UMA`で条件を確認・承諾し、read権限を持つtokenを発行する。
その後、GPU container内で次を実行してtokenを対話入力する。

```bash
hf auth login
```

tokenをチャットへ貼らず、command line引数、Slurm script、Git、共有logにも書かない。
認証の有無だけを検査し、token文字列自体は表示しない。licenseを承諾していない場合やtokenの
gated repository read権限がない場合は、計算を再試行せず認証状態を直す。

## 評価投入

ローカルの変更をcommit/pushし、SoftBank checkoutを`git pull --ff-only`で同期してから投入する。

```bash
ssh sb-gpu
cd /lustre/user140002/uma_pyscf
git status --short --branch
git pull --ff-only
sbatch scripts/slurm/run_base_uma_engineering_50_softbank_slurm.sh
```

scriptは次を固定する。

- NGC 26.07 imageと保存済みSHA-256
- 独立venv `fairchem-2.22-py312-v2`
- model cache `/lustre/user140002/models/fairchem/uma-s-1p2-2.22.0-v1`
- evaluation config `engineering_50_base_uma_s_1p2_v1.yaml`
- fixed dataset manifestとtrain/holdout partition

実行前にfairchem versionとCUDA availabilityを検査する。dataset shardはmanifest記録のSHA-256と
row数を照合し、各rowのrecord ID、`pbc=False`、`charge`、`spin`を確認してから推論する。
不一致時はartifactを発行せず失敗する。

fairchem-core 2.22.0では`get_predict_unit`の`cache_dir`引数がreference fileへは適用される一方、
checkpoint本体はimport時の`FAIRCHEM_CACHE_DIR`を使う。scriptは両者を同じmodel cacheへ固定し、
評価器も環境変数の不一致とcheckpoint-sized fileの欠落を拒否する。

## 評価artifact

出力は次に保存する。

```text
/lustre/user140002/runs/uma/base_eval_engineering_50_v1/<job-id>/evaluation.json
```

partitionごとにtotal energyのmean error、MAE、RMSE、per-atom MAE、force componentの
MAE/RMSE/max errorを記録する。組成offsetとgeometry依存の誤差を分けるため、同一組成内で
energy errorの平均を除いたcentered MAE/RMSEも併記する。centered metricはholdoutを学習へ
流用する補正ではなく、同一組成内の相対energy再現性を測る評価専用metricである。

model cacheの全file SHA-256、model source/license、seed、container、Python、ASE、fairchem、
torch/CUDA、GPU名、Git commitも同じartifactへ保存する。

## 判定後の順序

1. base UMAのtrain/holdout metricとpredictionを保存する。
2. 50件だけでtraining lossが十分低下するかを確認するoverfit smokeを行う。
3. 小datasetのtrain/validation plumbingを確認する。
4. 科学release条件を満たした後だけ1,000–5,000構造pilotへ進む。

engineering datasetはpipeline検証には使えるが、decision 0004が閉じている間はproduction modelの
学習済みcheckpointとしてreleaseしない。
