# 50構造GPU label engineering set実行記録

- 日付: 2026-08-31
- 実行先: SoftBank AIデータセンター、`fcdgx00081`
- Slurm job: `1797134`
- Git commit: `d0632ae85a63d88311253ce3c42d98dcbcf35af8`
- 状態: `COMPLETED`、exit code `0:0`、batch経過時間 `00:11:36`
- allocation: A100-SXM4-80GB x 1、16 CPU
- sampling config: `engineering_50_v1`
- protocol: `omol_wb97mv_tzvpd_v1`

## 目的

P2.3の1構造smokeを50構造へ広げ、次を実機で検証する。

- 1 candidate / 1 Python processの連続実行
- 5原子tierと8原子tierの切り替え
- raw、canonical record、attempt ledgerの原子的な発行
- 同run directoryに対するresume
- production engineering QC
- ledger記録checksumと実ファイルの整合

## Set構成

SiH4、GeH4、SiCl4、GeCl4、H3Si-GeCl3の5組成に対し、各組成で
`sigma=0.04 A`のシード付きCartesian displacementを8件、`sigma=0.12 A`を2件
生成した。全てneutral singletで、geometry QCは50/50 acceptedだった。

| Tier | Candidates | Applied PySCF threads | Applied max memory |
|---|---:|---:|---:|
| 5 atoms | 40 | 8 | 24,000 MB |
| 8 atoms | 10 | 16 | 48,000 MB |

adapterが`pyscf.lib.num_threads()`でtierを明示適用し、実際のthread数とmemory上限を
label recordのengine provenanceに記録することを50/50件で確認した。

## Label、resume、QC

| Pass | completed | skipped | failed | blocked |
|---|---:|---:|---:|---:|
| first label pass | 50 | 0 | 0 | 0 |
| same-run resume pass | 0 | 50 | 0 | 0 |

全50件がprimary density-fitting attemptの1回で収束し、direct fallbackは発動しなかった。
resume passは新しいattemptを追加せず、50件すべてをskipした。QCは
`accepted=50, rejected=0`で、組成ごとに10件ずつacceptedだった。

## Performanceと収束

| Metric | Result |
|---|---:|
| label wall time sum | 593.402312 s |
| per-candidate mean | 11.868046 s |
| per-candidate median | 11.966868 s |
| per-candidate min | 7.433087 s |
| per-candidate max | 17.289352 s |
| SCF iterations min / median / max | 8 / 10 / 12 |
| max CuPy pool after a run | 885,552,640 bytes |

最も遅かったのは8原子の
`engineering_50_v1_h3si_gecl3_seed_disp0p04_s2026083501`で、17.289352秒だった。

## Integrity

50個のraw attemptと50個のcanonical recordを実ファイルからSHA-256再計算し、
attempt ledgerおよびrecord内の記録値と比較した。不一致は0件だった。

## SoftBank Slurm運用で得た知見

- nodeは実メモリと無関係にSlurmへ`RealMemory=1`、`CfgTRES=mem=1M`を広告する。
  `#SBATCH --mem=48G`は受付時に`Memory specification can not be satisfied`となるため
  指定しない。PySCF側の`max_memory` tierは別に適用・記録する。
- attempt ledgerはatomic replaceで更新する。job中にlogin nodeから頻繁に読むと、
  Lustre cacheが一時的に古いunlink済みinodeを見せる場合がある。軽量な進捗確認には
  `label/records/*.json`のファイル数を使い、最終集計はjob完了後に行う。

## 成果物

```text
/lustre/user140002/runs/label/engineering_50_v1/1797134/
  provenance/
  input/
  label/attempt_ledger.json
  label/raw/                         # 50 attempts
  label/records/                     # 50 canonical records
  label/summary_first_pass.json
  label/summary_resume_pass.json
  qc/omol_wb97mv_tzvpd_conditional_qc_v1_report.json
```

## 判定と次の作業

50構造engineering setは完了。P2.3のbatch実行、resume、resource tier、QC、
checksum integrityは実機で正常に動作した。ただし、今回はneutral singlet限定で、
自然発生の失敗やdirect fallbackはなかった。dataset releaseは引き続きfail closedとし、
次はcomposition baseline、production QC閾値、non-default state registryを科学reviewする。
