# Project roadmap

詳細な目的、Gate、運用原則は[プロジェクト計画書](project_plan.md)を参照する。

## 現在地

- 基準日: 2026-09-01
- 現在のphase: **Part II — C0 teacher-data calibration準備**
- 次のGate: **Gate 2 — pilot dataset品質**
- dataset release・UMA fine-tuning: 科学閾値、composition baseline、state registryまで保留

## Roadmap

| Milestone | 状態 | 次の具体作業 |
|---|---|---|
| 実現性調査 | 完了 | — |
| ORCA/Ujilab計算基盤 | 完了 | — |
| CPU PySCF–ORCA検証 | 完了 | — |
| GPU環境固定 | 完了 | A1/A2/C0完了。固定container、lock、inventoryを保存済み |
| 5-case GPU smoke | 完了 | 5/5初回成功、暫定CPU–GPU数値gate通過 |
| 29-case三者比較 | 完了 | density-fitting + explicit MINAO候補29/29初回成功、CPU direct比20.96x |
| Production DFT protocol | v1固定 | density fitting + explicit MINAO、direct fallback、scope/QC ruleをconfig化 |
| Gate 1 | Conditional GO採択 | decision 0003。release条件はoffset、科学閾値、state registry |
| Package基盤 | P2.0–P2.6完了 | schema/QC/split/ASE-LMDB実装済み、50件load-back検証済み |
| P2.3 label pipeline | 完了 | SoftBank Slurm job 1797122。1件completed、QC accepted、checksum一致 |
| 50–200構造engineering set | 完了 | job 1797134、50/50初回収束、resume 50/50 skip、QC 50/50 accepted |
| Release controls | 機構完了・科学review継続 | train-only baseline実測済み、state registry 12件は全てpending、decision 0004 |
| C0 calibration 180 | 81/180 label・QC完了 | reaction endpoint 8 family準備済み。次はC0 independent 4 path / 36件 |
| C0-S state review 24–36 | dimer tranche 24/24 label・QC完了 | Si2H3/Si2H5/Ge2H3/Ge2H5はdoubletを強く支持。CPU/direct stability sentinel後に承認判断 |
| T0 fixed test 200 | C0後 | 独立parent/reactionをscore計算前にfreeze |
| P0 oracle pool 1,000 | C0/T0後 | blindに全件labelし、acquisition policyをretrospective比較 |
| 1,000–5,000構造pilot | P0/Gate 3後 | online active learningは実測signal確認後に開始 |
| UMA fine-tuning | 10,000-step engineering overfit評価完了 | train適合・holdout崩壊を確認。base UMAをscreening基準として維持 |
| 科学的・retention評価 | Gate 2後 | relative energy、forces、forgetting |
| Active learning / cluster拡張 | 78件IRC poolのUMA/PFP selection dry-run完了 | C0/P0でsignalを校正するまでproduction選抜へ使わない |

## 詳細計画

- [Part I: GPU4PySCF検証計画](plans/01_gpu4pyscf_validation_plan.md)
- [Part II: UMAファインチューニング実装計画](plans/02_uma_finetuning_implementation_plan.md)
- [Part II準備: 本番リポジトリ構成設計](plans/03_production_repository_structure.md)
- [P2.9前段: Matlantis/PFP多フィデリティ教師データ選抜pilot計画](plans/04_matlantis_multifidelity_screening_pilot.md)
- [教師データsampling・calibration・oracle pool計画](plans/05_teacher_data_sampling_and_calibration.md)

このファイルは進捗だけを更新する。scope、品質基準、設計を変更するときは、
`project_plan.md`および該当Part計画も更新し、必要に応じて`docs/decisions/`へ理由を残す。
