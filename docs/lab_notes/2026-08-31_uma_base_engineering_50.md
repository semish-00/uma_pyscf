# base UMA 50件評価実行記録

- 日付: 2026-08-31
- model: `uma-s-1p2`
- task: `omol`
- dataset: `ds_sigehcl_001`（40 train / 10 holdout）
- formal Slurm job: `1798793`
- Git commit: `8ebb3488c10aec9c555b4b51e4c4b3ad7971ed41`
- 状態: `COMPLETED`、exit code `0:0`、経過時間 `00:01:51`
- 用途: engineering-only fine-tuning前baseline

## 固定したmodel/runtime

`facebook/UMA`のgated accessを利用者本人がdevice認証し、checkpointをSoftBank上へ取得した。
推論は外部serviceではなくA100上でローカル実行した。

| 項目 | 値 |
|---|---|
| checkpoint bytes | 2,333,393,167 |
| checkpoint SHA-256 | `ba5c0d912efa22dc238e5fb1b5b7f66ee2e68c48c1b95b7cfd7fe1da5938398b` |
| fairchem-core | 2.22.0 |
| ASE | 3.26.0 |
| torch / CUDA | 2.13.0+cu130 / 13.0 |
| GPU | NVIDIA A100-SXM4-80GB |
| container SHA-256 | `b7fda1fe99974e5901c48e3e3bfaac0c4349384ac3285339a85c4885273a6a20` |

fairchem-core 2.22.0はcheckpoint本体とreference fileでcache指定経路が異なるため、
`FAIRCHEM_CACHE_DIR`と評価器の`model_cache_dir`を同一Lustre directoryへ固定した。
checkpoint-sized regular fileがない場合はartifactを発行しない。

## Metric

| Partition | Energy MAE (eV) | Energy MAE (eV/atom) | Same-composition centered MAE (eV) | Force component MAE (eV/A) |
|---|---:|---:|---:|---:|
| train (40) | 0.0308752 | 0.00535478 | 0.00735191 | 0.0254636 |
| holdout (10, SiCl4) | 0.00544185 | 0.00108837 | 0.000590527 | 0.00404154 |

formal artifact:

```text
/lustre/user140002/runs/uma/base_eval_engineering_50_v1/1798793/evaluation.json
SHA-256 52eaf82d7d34a9e954a06a8061b27936e45c294db90810a7c67ab3450f19c9e3
```

初回job 1798781も50件推論に成功したが、checkpoint本体がfairchemのdefault cacheへ入り、
artifactはreference fileだけをinventoryしたためformal evidenceには採用しない。cache固定後の
formal runとの差はprediction energy最大`1.20e-6 eV`、force component最大
`2.86e-6 eV/A`だった。

## 判定

固定ASE-LMDBからUMAをloadし、charge/multiplicity付き`omol`推論、energy/forcesの取得、
partition metric、checkpoint/runtime provenanceの発行まで完了した。base UMA evaluationを
engineering範囲で完了とし、次は同じ40/10 splitによる200-step overfit smokeへ進む。

このmetricは狭いneutral-singlet displacement setの結果であり、production精度や科学releaseを
保証しない。
