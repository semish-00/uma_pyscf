# GPU4PySCF検証（Part I）向けツール整備記録

- 日付: 2026-08-22
- 対象: `validation/orca_gpu4pyscf/`
- 状態: GPU計算は未実行。GPUホストで実行する準備をローカルで完了
- 前提: GPU機への計算投入自体は別途（Codex/ローカル）で行う

## 目的

[Part I計画](../plans/01_gpu4pyscf_validation_plan.md)のうち、GPU実機なしで
先行実装できる部分を整備した。対象はWorkstream A（環境固定）、Workstream B
（runner検査項目の一部）、Workstream C0/C1（dry runとsmoke ladder）、および
Gate 1判定用のCPU–GPU比較集計である。テストは`tests/`に追加し、PySCFなしで
全24件が通る。

## 追加・変更したもの

| 項目 | ファイル | 計画上の対応 |
|---|---|---|
| 環境inventory収集 | `collect_environment.py` | A1 |
| installation smoke test | `gpu_smoke_check.py` | A2 |
| 5-case smoke suite | `suites/gpu_smoke_v1.json` | C1 |
| 非PBS逐次suite runner | `run_suite.py` | C0/C1/C4 |
| 結果fileのatomic write | `common.py` `write_json` | B（中断時の不完全結果防止） |
| to_gpu後のgrid level検査 | `run_pyscf.py` | B（VV10/ordinary grid維持確認） |
| GPU provenance拡充 | `run_pyscf.py` | A1/性能metric（device名、memory、SCF/gradient別wall time） |
| engine pair対応の集計 | `summarize_suite.py`、`export_parity_csv.py` | Gate 1比較（CPU–GPU優先） |

## GPUホストでの実行手順

checkout後、`validation/orca_gpu4pyscf/`で次を順に実行する。

```bash
# 1. A1: 環境inventoryを保存（configs/environments/gpu4pyscf-<host>.yaml）
python collect_environment.py

# 2. A2: スタックを層別に検証（CuPy→GPU可視→kernel→PySCF→GPU4PySCF→RKS/UKS gradient）
python gpu_smoke_check.py --output runs/gpu_smoke_check.json

# 3. C0: 29 manifestの解決とelectron/spin parity検査（PySCF importなし）
python run_suite.py suites/si_ge_h_cl_ladder_v1.json --device gpu --dry-run

# 4. C1: 5-case smoke。最初の失敗で停止する
python run_suite.py suites/gpu_smoke_v1.json --device gpu

# 5. smoke全件成功後にのみ29-case ladderを実行
python run_suite.py suites/si_ge_h_cl_ladder_v1.json --device gpu

# 6. Gate 1優先比較: CPU–GPU
python summarize_suite.py suites/si_ge_h_cl_ladder_v1.json --root ../.. \
  --left-engine gpu4pyscf --right-engine pyscf-cpu \
  --write-comparisons --output runs/gpu_vs_cpu_summary.json
python export_parity_csv.py --left-engine gpu4pyscf --right-engine pyscf-cpu
```

CPU側の`runs/<case>/pyscf-cpu/result.json`は既存のUjilab計算をGPUホストへ
同期するか、比較をrawの置き場所（GPUサーバー主保存の方針）で行う。

## 設計上の判断

- `run_suite.py`は各caseを子プロセスで実行する。native crashやGPU memory
  leakが後続caseへ波及しないためで、C1の「1件ずつ実行」を機械的に強制する。
- 失敗attemptは`runs/<case>/<engine>/attempts.jsonl`へ追記のみで残し、
  `result.json`は成功時のみatomicに書く。C4の「failed caseは上書きせず
  attemptを残す」に対応する。
- `run_pyscf.py`は`to_gpu()`変換後にordinary/VV10 grid levelが指定値の
  ままかを検査し、変わっていれば計算を拒否する（fail closed）。
- 集計スクリプトのengine pairはCLI引数で選ぶ。CPU–GPUが一致しない状態で
  GPU–ORCA差を解釈しない、という計画の優先順位を運用で崩さないため、
  defaultは従来のpyscf-cpu vs orcaに固定した。

## 未着手（GPU実機が必要なもの）

- A1/A2の実際の実行とversion固定、lock file作成
- C2 diagnostic lane（density fittingなしのGPU実行可否確認を含む）
- C3 production-candidate laneの設定比較
- charge/spin mini-matrixの計算実行と科学的状態選択のreview
  （候補suite `charge_spin_mini_v1`と生成器は同日追加済み。
  `state_selection_status: pending_scientific_review`）
- Gate 1 decision record（metric集計は`gate1_metrics.py`を同日追加済み）

Part II（`src/uma_pyscf/`のscaffold、canonical schema実装）はGate 1がGOに
なるまで着手しない方針を維持する。
