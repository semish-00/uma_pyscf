# Project roadmap

詳細な目的、Gate、運用原則は[プロジェクト計画書](project_plan.md)を参照する。

## 現在地

- 基準日: 2026-08-31
- 現在のphase: **Part II — P2.3 GPU4PySCF label pipeline engineering smoke**
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
| Package基盤 | P2.0–P2.2、P2.4、P2.6完了 | 577 unit test、schema/QC/split実装済み |
| P2.3 label pipeline | MVP実装・GPU smoke待ち | sample→label→QCをSoftBank Slurmで1件実行 |
| 50–200構造engineering set | P2.3 smoke後 | 中断・resume、failure率、memory tierを確認 |
| 1,000–5,000構造pilot | release条件のreview後 | H/Si/Ge/Cl、8原子以内から開始 |
| UMA fine-tuning | Gate 2後 | base評価→overfit smoke→pilot |
| 科学的・retention評価 | Gate 2後 | relative energy、forces、forgetting |
| Active learning / cluster拡張 | 後続 | fixed holdoutの改善で判断 |

## 詳細計画

- [Part I: GPU4PySCF検証計画](plans/01_gpu4pyscf_validation_plan.md)
- [Part II: UMAファインチューニング実装計画](plans/02_uma_finetuning_implementation_plan.md)
- [Part II準備: 本番リポジトリ構成設計](plans/03_production_repository_structure.md)

このファイルは進捗だけを更新する。scope、品質基準、設計を変更するときは、
`project_plan.md`および該当Part計画も更新し、必要に応じて`docs/decisions/`へ理由を残す。
