# 0003: Gate 1 — GPU4PySCF label engineを条件付き採用する

- 状態: **採択（Conditional GO）**
- 提案日: 2026-08-30
- 採択日: 2026-08-31
- 判断者: プロジェクトオーナー（本セッションの「進めてください」により採択）
- 関連: [Part I検証計画](../plans/01_gpu4pyscf_validation_plan.md)、
  [C3実行記録](../lab_notes/2026-08-30_gpu4pyscf_c3.md)、
  [C4実行記録](../lab_notes/2026-08-30_gpu4pyscf_c4.md)、
  [charge/spin実行記録](../lab_notes/2026-08-30_gpu4pyscf_charge_spin.md)

## 判定

**Conditional GO**とする。

GPU4PySCFをH/Si/Ge/Cl分子系の教師label engineとして条件付き採用し、Part IIのlabel
pipeline MVPと小規模pilotへ進める。ただし、以下のproduction protocolと機械判定可能な
制約を満たす場合に限る。production protocolは
[`omol_wb97mv_tzvpd_v1.yaml`](../../configs/dft/omol_wb97mv_tzvpd_v1.yaml)、
engineering QCは
[`omol_wb97mv_tzvpd_conditional_qc_v1.yaml`](../../configs/datasets/omol_wb97mv_tzvpd_conditional_qc_v1.yaml)
として機械化する。

## 固定するproduction protocol

| 項目 | 固定値 |
|---|---|
| engine | GPU4PySCF 1.8.1 / PySCF 2.14.0 |
| method | ωB97M-V / def2-TZVPD |
| ordinary grid | level 5 |
| VV10 grid | level 5 |
| grid response | on |
| density fitting | on |
| SCF convergence | `1e-10`、最大250 cycle |
| initial density | CPU PySCF objectでexplicit MINAOを生成し、device変換前後で共用 |
| environment | 固定NGC container、CUDA/CuPy/cuTENSOR lock、A100 |
| fallback | 同一method/gridのdensity fitting off（direct） |

## 条件と機械判定rule

1. **Composition-dependent energy offsetを明示する。** density fittingはdirectに対し
   最大6.66e-5 Ehのabsolute shiftを示し、29件中8件が暫定5e-5 Ehを僅かに超えた。
   異なる組成のraw total energyを、そのoffsetを無視して直接比較しない。dataset/model側で
   atomic referenceまたはcomposition baselineを管理し、same-composition relative energyと
   gradientを主要QCにする。
2. **Initial densityを固定する。** `run_pyscf.py`はCPU側で指定guessを一度生成し、
   `kernel(dm0=...)`へ明示的に渡す。resultにguessと生成位置を記録し、暗黙guessへ戻さない。
3. **Open-shell QCを必須にする。** convergence、charge、multiplicity、`<S^2>`、target deviation、
   initial guess、attempt履歴を保存する。charge/spin matrixの状態選択は
   `pending_scientific_review`のままであり、承認前にteacher dataへ混ぜない。
4. **検証scope外へ自動拡張しない。** 現時点の証拠はH/Si/Ge/Cl、非周期分子、最大8原子、
   ωB97M-V/def2-TZVPDに限る。周期系、slab、100原子級、他元素・他methodは別gateを要求する。
5. **Versionとprovenanceを固定する。** container、Python overlay、package lock、GPU、runtime
   file checksum、input fingerprint、job/session summaryが欠けるresultをrelease対象にしない。
6. **Direct fallbackを保持する。** SCF不収束と確認済みSCF-root不一致は、同一method/gridの
   directへ再試行できる。spin contamination、gradient外れ値、未検証scopeは自動fallbackせず
   reviewへ送り、設定を緩めたresultを同一datasetへ無条件に混ぜない。

最終数値thresholdとspin contamination許容値は今回の採択ではfreezeしない。検証用暫定値を
engineering QCに明記し、dataset release品質保証値としては扱わない。

## 根拠

| Evidence | Result |
|---|---:|
| A1/A2 environment and smoke | 7/7 checks、固定lock/inventory保存 |
| C1 direct CPU–GPU | 5/5暫定gate通過 |
| C3 one-axis GPU matrix | 20/20初回成功 |
| density-fit CPU–GPU sentinel | 4/4暫定gate通過 |
| same-composition relative-energy error | worst 0.0146 kcal/mol |
| C4 final full ladder | 29/29初回成功 |
| C4 gradient vs CPU direct | RMSE worst 2.18e-5、max 6.51e-5 Eh/bohr |
| C4 performance | CPU direct比aggregate 20.96x |
| explicit-MINAO charge/spin parity | 12/12暫定gate通過 |
| charge/spin performance | CPU比aggregate 14.94x |

Gate 1図表は
[`gpu_c4_density_fit_minao_parity.png`](../../validation/orca_gpu4pyscf/analysis/c4/minao/gpu_c4_density_fit_minao_parity.png)、
数値表は同directoryのJSON/CSVに保存した。

## Conditional GOを選ぶ理由

無条件GOにはしない。density-fitting absolute energy shift、未freezeのthreshold、
charge/spin状態選択、8原子を超えるscalingが未確定だからである。一方、NO-GOにも該当しない。
初期guess由来の唯一の再現性問題は原因を分離してrunnerで解消でき、final C4は29/29、
charge/spinは12/12で再現し、速度も実用上十分だった。

## 採用後の影響

- Part II P2.3 label pipeline MVPと、1,000–5,000構造pilotの準備を開始できる。
- production configには上記protocol、scope、fallback、QC ruleを同時に実装する。
- dataset release前にcomposition baselineとspin-state registryを科学reviewする。
- pilotでmemory/atom-count scaling、failure rate、offset分布を再計測し、Gate 2へ渡す。

## 採択後も残るrelease条件

1. composition-dependent offsetの取扱い方針とbaseline実装
2. production用energy/gradient/spin thresholdの科学review
3. charge/spin state registryの採否とreview担当

これらが完了するまでproduction configの`release_allowed`はfalse、QC configの
`release_status`は`engineering_only_pending_scientific_freeze`とする。
