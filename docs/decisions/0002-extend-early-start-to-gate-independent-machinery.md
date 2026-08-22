# 0002: 前倒し実装のscopeをGate 1非依存の機構全般へ拡張する

- 状態: 採用
- 日付: 2026-08-22
- 判断者: プロジェクトオーナー（「独立で進められるものは進める」指示による）
- 関連: [0001](0001-start-part2-foundation-before-gate1.md)、
  [本番リポジトリ構成設計](../plans/03_production_repository_structure.md)

## 決定

[0001](0001-start-part2-foundation-before-gate1.md)のP2.0/P2.1限定を拡張し、
次の原則で前倒し対象を判定する。

> **機構（machinery）はGate 1非依存なら前倒し可。実行（execution）と
> 科学条件の確定はGateに従う。**

前倒し可（Gate 1の判定結果に内容が依存しないもの）:

- P2.2: 決定論的構造候補生成、geometry QC、charge/spin sibling展開
- P2.6の分割機構: parent/composition/charge/multiplicity group split生成器
  （schema上のrecordに対して動作する純データ処理）
- schema・registry・変換器などrecord形式に閉じた実装

引き続きGateに従う（内容や実行がGate判定・実データ・レビューに依存）:

- `configs/dft/`のproduction protocol値の確定（Gate 1）
- P2.3 calculatorsの本番label pipelineと大量label生成（Gate 1）
- dataset releaseと学習・評価の実行（Gate 1/2/3）
- charge/spin状態の科学的選択の承認（別途review）
- fairchem/UMAに依存する`inference/`実体（checkpoint入手と依存導入時）

## 理由

GPU検証は別線（Codex/ローカル）で進むため、record形式に閉じた機構の実装は
Gate 1と並行しても手戻りが生じない。0001と同じ判断根拠を、milestone番号
ではなく判定原則として一般化した。

## 影響

- 実装順序はconfigs/schemaの依存関係に従う（P2.2 → 分割機構の順を基本）。
- 各前倒しmilestoneのcommitは本decisionを参照する。
- 構成設計（03）第7節の「作成タイミング」は、前倒し分についても
  「moduleとtestを同時に追加する」規則を維持する。
