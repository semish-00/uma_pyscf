# P2.3 GPU label pipeline engineering smoke実行記録

- 日付: 2026-08-31
- 実行先: SoftBank AIデータセンター、`fcdgx00081`
- Slurm job: `1797122`
- Git commit: `58256f5e5130e4f9f3e1172db5c7df72d1274872`
- 状態: `COMPLETED`、exit code `0:0`、batch経過時間 `00:00:42`
- GPU: NVIDIA A100-SXM4-80GB x 1
- candidate: SiH4、bond 01 x 1.05、charge 0、multiplicity 1、5 atoms

## 実行結果

`sample -> label dry-run -> GPU4PySCF label -> production QC`を1件でend-to-end実行した。
label summaryは`completed=1, failed=0, blocked=0, skipped=0`、QC reportは
`accepted=1, rejected=0`だった。production protocolのrelease flagは設計通り`false`であり、
科学閾値、composition baseline、non-default state registryが固定されるまで
engineering利用に限定する。

計算は初回のprimary density-fitting attemptで収束した。

| Metric | Result |
|---|---:|
| SCF iterations | 8 |
| SCF wall time | 7.274958 s |
| gradient wall time | 1.058056 s |
| label wall time | 9.982622 s |
| energy | -291.8436480555339 Eh |
| max absolute gradient component | 0.013767891499309792 Eh/bohr |
| `<S^2>` | 0.0 |

## Protocolと実行環境

- protocol: `omol_wb97mv_tzvpd_v1`
- canonical protocol SHA-256:
  `34a77ae8fd809f893427fcfb1fe1400743e895c9afdb650215cb406bcb331404`
- method: ωB97M-V/def2-TZVPD、grid 5、VV10 grid 5、`grid_response=true`
- SCF: tolerance `1e-10`、max cycle 250、density fitting on
- initial density: explicit MINAO、CPU生成後にGPUへ変換
- Python 3.10.12、PySCF 2.14.0、GPU4PySCF 1.8.1、CuPy 13.4.1、cuTENSOR 2.2.0
- container SHA-256:
  `774bc1362b08db5bbbd08dd1dd3f7abf23610ed6f4091e85fcc6c8205c959aa9`
- Python lock SHA-256:
  `7f4dc93349b80173d5b5f9e1c1a4c0b92ef3cf50f045dd4c95955d5b73c76db8`

## Integrity

ledgerが記録したSHA-256と実ファイルを`sha256sum`で再計算し、一致を確認した。

- raw attempt:
  `51f3d83a9036a290fd13a2261609ae7d48bdbda372ebe3b32f96e868066b52fa`
- canonical record:
  `8e479e7c6cdebd1280c73ce57240882773997fb8d15b69aa9cf2c649628b04ff`

## 初回失敗と修正

job `1797121`は計算開始前に`/raid/user140002`の作成権限不足で失敗した。
SoftBank GPU機の`/raid`直下はユーザがdirectoryを作成できないため、commit
`58256f5`でcandidateごとの一時領域をcompute nodeのcontainer-local `/tmp`に変更した。
Pythonの`TemporaryDirectory`によりworker終了時に自動削除される。

## 成果物

```text
/lustre/user140002/runs/label/gpu_label_smoke_v1/1797122/
  provenance/
  input/
  label/attempt_ledger.json
  label/raw/gpu_label_smoke_v1_sih4_seed_bond01_x1p05/01_primary_density_fit.json
  label/records/gpu_label_smoke_v1_sih4_seed_bond01_x1p05.json
  label/summary.json
  qc/omol_wb97mv_tzvpd_conditional_qc_v1_report.json
```

## 判定

P2.3の1構造engineering smokeは完了。次は50–200構造engineering setで、
resume、failure ledger、resource tier、GPU memory、スループットを検証する。
