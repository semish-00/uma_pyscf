# 2026-08-31 scientific release conditions review

## 実装と検証

commit `ed2445f`で次を実装した。

- `uma-pyscf fit-baseline`: accepted recordとsplit manifestからtrain-only atomic baselineをfit
- `uma-pyscf label/qc --state-registry`: registry実体とchecksumを含む厳格なstate承認
- `uma-pyscf-composition-baseline-v1`と`uma-pyscf-state-registry-v1` schema
- 12件のSiH3/GeH3 candidate stateを記録したpending registry

local verificationはruff、mypy、`646 passed, 485 subtests passed`。SoftBank checkoutを同commitへ
fast-forwardし、job 1797134のQC accepted 50 recordを再利用した。GPU再計算は行っていない。

## baseline run

```bash
python3 -m uma_pyscf.cli.main split \
  --config configs/datasets/engineering_50_baseline_split_v1.yaml \
  --candidates /lustre/user140002/runs/label/engineering_50_v1/1797134/input/engineering_50_v1_candidates.json \
  --output-dir /lustre/user140002/runs/label/engineering_50_v1/1797134/baseline/splits

python3 -m uma_pyscf.cli.main fit-baseline \
  --config configs/datasets/engineering_50_atomic_baseline_v1.yaml \
  --split /lustre/user140002/runs/label/engineering_50_v1/1797134/baseline/splits/engineering_50_baseline_split_v1.json \
  --records /lustre/user140002/runs/label/engineering_50_v1/1797134/qc/records \
  --output-dir /lustre/user140002/runs/label/engineering_50_v1/1797134/baseline/artifacts
```

splitはtrain 40 record / 4 parent、holdout 10 record / 1 parent。fit元素はCl、Ge、H、Si、
design rankは4だった。artifact schemaをread-backし、50 record checksumが保存されていることを
確認した。

| Composition | Partition | centered RMSE (Ha) | centered max abs (Ha) | gradient max component (Ha/bohr) |
|---|---|---:|---:|---:|
| Cl3GeH3Si | train | 0.0455056 | 0.124721 | 0.298421 |
| Cl4Ge | train | 0.0431965 | 0.112419 | 0.230460 |
| GeH4 | train | 0.0391661 | 0.108930 | 0.256003 |
| H4Si | train | 0.00536240 | 0.0107894 | 0.0948659 |
| Cl4Si | holdout | 0.0178263 | 0.0436178 | 0.154063 |

全50件のgradient norm最大は0.539927 Ha/bohr、`s2_deviation`は全件0（neutral singlet）だった。
holdout Cl4Siのbaseline mean errorは-0.0374416 Ha。trainに含めていないholdoutの値であり、
fit leakageはない。

## state registry review

`configs/states/h_si_ge_cl_states_v1.yaml`はPart Iで数値比較した12 stateを列挙するが、approved
entryは0件である。12/12 parityはengine採否の証拠で、電子状態選択の科学承認ではないため、
exact provenanceを付けてもstatusがpendingならlabel/QCはblockする。approvedへ変更するには
evidence、reviewer、decisionの三つがschema上必須である。

## 結論

baselineとstate registryのrelease control機構は実装・実機データ検証済み。科学閾値と
non-default stateの承認証拠は未充足なのでreleaseは閉じたままにする。判断の詳細は
[decision 0004](../decisions/0004-release-controls-mechanized-release-remains-closed.md)に記録した。
