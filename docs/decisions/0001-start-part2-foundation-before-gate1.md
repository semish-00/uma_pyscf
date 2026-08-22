# 0001: Part II基盤（P2.0/P2.1のGate 1非依存部分）をGate 1前に実装開始する

- 状態: 採用
- 日付: 2026-08-22
- 判断者: プロジェクトオーナー（セッション指示による）
- 関連: [project_plan](../project_plan.md) §2、[Part II計画](../plans/02_uma_finetuning_implementation_plan.md) §6–7、
  [本番リポジトリ構成設計](../plans/03_production_repository_structure.md)

## 文脈

従来の計画では、Part II（`src/uma_pyscf/`パッケージ）の実装開始条件は
Gate 1のGO/Conditional GOだった。一方、GPU4PySCF検証のGPU実機実行は
別線（Codex/ローカル）で進行するため、本セッション側の作業がGate 1待ちで
停止する。P2.0（package scaffold、core）とP2.1（canonical schema）は
Gate 1の判定結果（DFT数値設定・制限事項）に依存しない。

## 決定

1. Gate 1判定を待たずに、次の**Gate 1非依存の基盤**を実装する。
   - P2.0: `pyproject.toml`、`src/uma_pyscf/core/`（units、spin、ids、io、
     errors）、`cli/`骨格、`tests/unit/`、lint/型チェック設定
   - P2.1: `src/uma_pyscf/schemas/`（canonical label record、structure
     manifest、crosscode result importer）
2. 次は**引き続きGate 1（またはGate 2/3）まで開始しない**。
   - `configs/dft/`のproduction protocol値の確定
   - `calculators/`の本番label pipeline（P2.3）と大量label生成
   - dataset生成・split・学習・評価の実行（P2.4以降の実行系）
3. 実装体制: 実装計画とオーケストレーションはFable 5セッション、
   実装本体はOpus以下のモデルのsubagentが行い、レビューを経てcommitする。

## 理由

- P2.0/P2.1の内容（単位、スピン変換、atomic I/O、record schema）は
  Gate 1のどの判定結果でも変わらない。
- validation/で確立した挙動（spin変換、atomic write）をtest付きで
  移植する作業は、GPU検証と独立に品質を固定できる。
- Gate 1がNO-GOの場合でも、これらの基盤はCPU PySCF縮小案・ORCA継続案の
  いずれでも再利用できるため、手戻りリスクが小さい。

## 影響

- roadmapに「Package基盤（Gate 1非依存部分）」の進行中項目を追加。
- 構成設計（03）の「Gate 1前に`src/`以下を作成しない」を
  「Gate 1の結果に依存する部分は作成しない」に置き換える。
- Gate 1判定時には、確定したprotocol・制約を`configs/dft/`と
  本設計へ反映してからP2.2以降を開始する（従来どおり）。
