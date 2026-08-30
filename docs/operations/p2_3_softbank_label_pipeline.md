# P2.3 SoftBank GPU label pipeline

- 状態: Gate 1 Conditional GO後のengineering運用
- 対象: `omol_wb97mv_tzvpd_v1`
- release: 禁止（科学閾値、composition baseline、state registryのfreeze前）

## 実行モデル

`uma-pyscf label`はcandidate manifestを読み、各候補を独立したPython subprocessで
GPU4PySCFへ渡す。1候補ごとにraw attempt、canonical label record、checksum、attempt
ledgerをatomicに保存する。中断後に同じcommandを再実行すると、protocol fingerprintが
一致するcompleted recordを検証してskipする。

primaryはdensity fitting + explicit MINAOである。MINAO densityはCPU PySCF object上で
一度生成し、`to_gpu()`後の`kernel(dm0=...)`へ明示的に渡す。SCF不収束または確認済みの
SCF-root不一致だけが、同一method/gridのdirect fallbackへ進む。runtime/version/protocol
provenance欠落はfallbackせず失敗する。

## ローカルdry-run

```bash
uma-pyscf sample \
  configs/sampling/gpu_label_smoke_v1.yaml \
  --output-dir /tmp/uma-p23-input

uma-pyscf label \
  --config configs/dft/omol_wb97mv_tzvpd_v1.yaml \
  --manifest /tmp/uma-p23-input/gpu_label_smoke_v1_candidates.json \
  --output-dir /tmp/uma-p23-label \
  --dry-run
```

dry-runはPySCF/CUDAをimportせず、scope、resource tier、primary/fallback計画を表示する。
非default charge/multiplicityは`state_registry:`で始まる承認provenanceがなければblockする。

## SoftBank engineering smoke

GPU側checkoutはcleanなconsumerとし、ローカルでtest済みcommitをpushしてからpullする。

```bash
ssh sb-gpu
cd /lustre/user140002/uma_pyscf
git status --short --branch
git pull --ff-only
sbatch scripts/slurm/run_label_smoke_softbank_slurm.sh
```

Slurm clientはlogin shell初期化に依存するため、Codexからは対話PTYの`ssh sb-gpu`内で
投入する。jobは1件のSiH4候補についてsample → label → production QCを実行し、成果物を
次へ保存する。

```text
/lustre/user140002/runs/label/gpu_label_smoke_v1/<job-id>/
  provenance/
  input/
  label/
    raw/
    records/
    attempt_ledger.json
    summary.json
  qc/
  label_plan.json
```

成功条件はSlurm `COMPLETED`、label summaryの`completed=1, failed=0, blocked=0`、QCの
`accepted=1, rejected=0`、raw/record checksum整合である。QC acceptedはengineering判定で
あり、`release_status: engineering_only_pending_scientific_freeze`を解除しない。

## Resumeと失敗時

同じ`RUN_ROOT`でjobを再投入するとcompleted recordはskipされる。terminal failureを明示的に
再試行するときだけlabel CLIへ`--retry-failed`を付ける。config、manifest、protocol fingerprint
が既存ledgerと違う場合は同じrun directoryを再利用せず、新しい`RUN_ROOT`を使う。

raw label、ledger、QC reportは削除しない。`/raid`上のscratchだけをjob終了時に安全なprefixを
確認して削除する。
