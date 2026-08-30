# 50構造ASE-LMDB変換実行記録

- 日付: 2026-08-31
- 実行先: SoftBank AIデータセンター、`fcdgx00081`
- Slurm job: `1797318`
- Git commit: `5d7a376818178ba6309e1edbbe786193f71abb92`
- 状態: `COMPLETED`、exit code `0:0`、batch経過時間 `00:01:02`
- source label job: `1797134`
- dataset ID: `ds_sigehcl_001`
- 用途: engineering-only dataset plumbing / overfit smoke

## 実装・version

`uma-pyscf dataset`と`verify-dataset`を実装し、SoftBankでは固定container
`nvidia-pytorch_23.10-py3.sqsh`内の独立overlayを使った。

```text
ase==3.26.0
ase_db_backends==0.11.0
lmdb==1.7.3
fairchem_core target==2.22.0
```

fairchemの公式実装を照合し、OMolのASE state keyを`charge`と`spin`に固定した。`spin`は
PySCFの`2S`ではなくmultiplicity `2S+1`である。学習時はdata configの
`a2g_args.r_data_keys`へ`[charge, spin]`を必須指定する。

## 結果

parent-group splitを変更せず、2 shardへ変換した。

| Partition | Records | Shards | Size | SHA-256 |
|---|---:|---:|---:|---|
| train | 40 | 1 | 53,248 bytes | `ed49a9aa20ad79bcdd000299613e25eb1a83f3b49fc0694bc8a0eaf2825111d1` |
| holdout | 10 | 1 | 20,480 bytes | `78564d37f604b65b68ad44fdcb24132495f61eebe020b93d956ca3011e0c1006` |

manifest:

- path:
  `/lustre/user140002/runs/label/engineering_50_v1/1797134/dataset/ds_sigehcl_001/dataset_manifest.json`
- SHA-256: `8b94e1308fc368a8f98428bd27e8ae0e0512a7f979ac200a73d9afc2aec90deb`
- size: 13,989 bytes
- schema: `uma-pyscf-ase-dataset-manifest-v1`

## 検証

export中のload-backと、publish後の`verify-dataset`の両方が成功した。次を全50件でsourceと照合した。

- atomic number、atom order、position、`pbc=False`
- energy Hartree→eV
- force `-gradient`、Hartree/Bohr→eV/Angstrom
- charge、multiplicity、task、record ID
- source record checksum

追加の独立集計でも、source record checksum不一致0、split checksum一致、2 shardのchecksum一致を
確認した。fixture integration testではsingletとdoubletを往復し、shard末尾を改変した破損が
checksum mismatchで学習前に拒否されることも確認済みである。

## 判定と次の作業

P2.5のsmall fixture lossless round-trip、実50件のrecord/field照合、shard count/checksum、
破損検出を満たした。教師データ作成pipelineはengineering範囲でfine-tuning inputまで到達した。

次はPython 3.11以上の`fairchem-core==2.22.0`学習環境を別に固定し、同じholdoutでbase UMA
`uma-s-1p2`を評価した後、50件のoverfit smokeを行う。production科学releaseは、pendingの
state registryと科学QC閾値がfreezeされるまで引き続き閉じる。
