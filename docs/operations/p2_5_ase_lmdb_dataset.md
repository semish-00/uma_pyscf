# P2.5 ASE-LMDB dataset運用

- 対象: QC accepted canonical label record
- 出力: fairchem向けASE-LMDB（energy + forces）
- task: `omol`
- release: manifestの`release`許可ではない。科学release判定は別Gateで行う

## 固定した互換契約

初回engineering datasetでは次を採用する。

| 項目 | 固定値 |
|---|---|
| ASE | `3.26.0` |
| ASE DB backends | `0.11.0` |
| fairchem-core target | `2.22.0` |
| dataset format | `ase-lmdb` |
| UMA task | `omol` |
| regression tasks | `ef` |
| energy | eV |
| forces | eV/Angstrom |
| position | Angstrom |
| boundary | aperiodic (`pbc=False`) |

canonical recordのenergyはHartree、gradientはHartree/Bohrのまま保持する。dataset export層だけが
ASEの`units.Hartree`と`units.Bohr`を使って変換し、`forces = -gradient`を一度だけ適用する。

fairchemのOMol入力では`Atoms.info["charge"]`がtotal charge、
`Atoms.info["spin"]`が**spin multiplicity (2S+1)**である。PySCFの`spin_2s`を`spin`へ渡さない。
各ASE rowには`charge`、`spin`、`multiplicity`、`task`、`record_id`、source record checksumを保存する。

## Export

dataset依存を入れる。

```bash
python3 -m pip install '.[dataset]'
```

新しいversioned output directoryを指定する。既存directoryへの上書きは拒否する。

```bash
python3 -m uma_pyscf.cli.main dataset \
  --config configs/datasets/engineering_50_ase_lmdb_v1.yaml \
  --split /path/to/split.json \
  --records /path/to/qc/records \
  --output-dir /path/to/ds_sigehcl_001
```

commandは次を完了してから`dataset_manifest.json`を発行する。

1. splitとrecord集合の完全一致、全recordの`qc.status=accepted`とSCF収束を確認
2. partitionごとにrecord ID順でASE-LMDB shardを作成
3. 全rowをload backし、原子順、座標、energy、forces、charge、multiplicity、taskをsourceと照合
4. source record、split、全shardのSHA-256とrecord countをmanifestへ保存
5. scratch directoryを最終output directoryへatomic rename

途中で失敗した場合はscratchを除去し、manifestも最終directoryも発行しない。

## 再検証

学習開始前にsourceとartifactをもう一度照合する。

```bash
python3 -m uma_pyscf.cli.main verify-dataset \
  --manifest /path/to/ds_sigehcl_001/dataset_manifest.json \
  --records /path/to/qc/records \
  --dataset-dir /path/to/ds_sigehcl_001
```

source recordの変更、shardの欠落・改変、row count不一致、field/value不一致はすべて失敗する。

## fairchem configの必須追記

fairchem 2.22.0の`AtomicData.from_ase`は、`r_data_keys`に明示された場合だけASE infoの
`charge`と`spin`を読み込む。公式fine-tuning helperが生成するdata YAMLをそのまま使わず、
train/validationの両方に次を持たせる。

```yaml
train_dataset:
  format: ase_db
  a2g_args:
    r_data_keys: [charge, spin]

val_dataset:
  format: ase_db
  a2g_args:
    r_data_keys: [charge, spin]
```

この指定がないconfigはOMol state入力を既定値へ落とすため、本projectでは学習開始前検証で拒否する。

## SoftBank 50件engineering run

```bash
sbatch scripts/slurm/run_dataset_engineering_50_softbank_slurm.sh
```

scriptは固定container内に独立dataset overlayを構築し、job 1797134のQC accepted recordsと
parent-group splitから`ds_sigehcl_001`を作る。export直後に`verify-dataset`も実行する。
