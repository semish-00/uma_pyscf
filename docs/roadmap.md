# Project roadmap

詳細な目的、Gate、運用原則は[プロジェクト計画書](project_plan.md)を参照する。

## 現在地

- 基準日: 2026-08-22
- 現在のphase: **Part I — GPU4PySCF validation**
- 次のGate: **Gate 1 — GPU4PySCFを教師label engineとして採用できるか**
- 大量dataset生成・UMA fine-tuning: Gate 1まで保留

## Roadmap

| Milestone | 状態 | 次の具体作業 |
|---|---|---|
| 実現性調査 | 完了 | — |
| ORCA/Ujilab計算基盤 | 完了 | — |
| CPU PySCF–ORCA検証 | 完了 | — |
| GPU環境固定 | 未着手 | GPU/driver/CUDA/package inventory |
| 5-case GPU smoke | 未着手 | H2、SiH4、SiCl4、SiH3、混合分子 |
| 29-case三者比較 | 未着手 | CPU–GPUを優先して評価 |
| Production DFT protocol | 未着手 | grid/density fitting収束 |
| Gate 1 | 未着手 | GO / Conditional GO / NO-GO |
| Package基盤（Gate 1非依存部分） | P2.0–P2.1完了、P2.2進行中 | sampling/geometry QC実装（decisions/0001, 0002） |
| Dataset/QC実装 | Gate 1後 | label pipeline MVP |
| 1,000–5,000構造pilot | Gate 1後 | H/Si/Ge/Cl、charge/spin |
| UMA fine-tuning | Gate 1後 | base評価→overfit smoke→pilot |
| 科学的・retention評価 | Gate 1後 | relative energy、forces、forgetting |
| Active learning / cluster拡張 | 後続 | fixed holdoutの改善で判断 |

## 詳細計画

- [Part I: GPU4PySCF検証計画](plans/01_gpu4pyscf_validation_plan.md)
- [Part II: UMAファインチューニング実装計画](plans/02_uma_finetuning_implementation_plan.md)
- [Part II準備: 本番リポジトリ構成設計](plans/03_production_repository_structure.md)

このファイルは進捗だけを更新する。scope、品質基準、設計を変更するときは、
`project_plan.md`および該当Part計画も更新し、必要に応じて`docs/decisions/`へ理由を残す。
