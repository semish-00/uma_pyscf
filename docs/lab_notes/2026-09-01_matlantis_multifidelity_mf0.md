# Matlantis/PFP multi-fidelity MF0・selection dry-run

- 実施日: 2026-09-01
- 状態: selection plumbing、10,000-step UMA評価、78件IRC poolの
  UMA/PFP推論・selection dry-run完了
- 科学的効力: engineering evidenceのみ。Gate MF-B/Cの判定には使用しない
- 計画:
  [Matlantis/PFP多フィデリティ教師データ選抜pilot](../plans/04_matlantis_multifidelity_screening_pilot.md)

## 1. 10,000-step UMA評価

200-step評価と結果を混同しないよう、次をversion分離した。

- evaluation config:
  `configs/evaluation/engineering_50_finetuned_10000step_v1.yaml`
- Slurm script:
  `scripts/slurm/run_finetuned_eval_engineering_50_v2_softbank_slurm.sh`
- 出力root:
  `/lustre/user140002/runs/uma/finetuned_eval_engineering_50_v2/<job-id>/`

SoftBank job `1799945`は`num_steps_completed = 10000`、
`Training Completed 10000 steps`、final checkpoint保存まで正常完了していた。
checkpoint SHA-256は
`2bac054b512d153a150cc5d1d11f0402c526ef7fa223f9fca4e5708435852e81`。
job `1801661`で同じ40-train/10-holdoutへ評価し、終了状態は`COMPLETED (0:0)`だった。

| Model | Partition | Centered energy MAE (eV) | Force MAE (eV/Angstrom) |
|---|---|---:|---:|
| base UMA-S-1.2 | train | 0.007352 | 0.025464 |
| 200-step | train | 0.646060 | 0.938110 |
| 10,000-step | train | 0.000663 | 0.002528 |
| base UMA-S-1.2 | holdout | 0.000591 | 0.004042 |
| 200-step | holdout | 0.363400 | 0.805977 |
| 10,000-step | holdout | 0.044772 | 0.239682 |

10,000-step modelはtrainへは強くoverfitしたが、holdoutではbase UMAよりcentered
energy MAEが約75.8倍、force MAEが約59.3倍悪い。200-step failureからは改善したが、
screening modelやscientific baselineへ昇格させない。新規78件poolはbase UMAで推論し、
job `1801662`で78/78 predictionをschema検証済みで回収した。

## 2. selection recordとCLI

P2.9の既存設計に従い、新しいpackageを作らず次を追加した。

- `uma-pyscf-acquisition-scores-v1`
- `uma-pyscf-selection-manifest-v1`
- `uma-pyscf select`
- parentごとの選抜上限
- policy unionのHF予算上限
- random/score policyの決定論的順位付け
- score欠落、policy budget不足、union超過時のfail-closed

初回scoreはPFPとbase UMAの予測だけから作り、GPU4PySCF reference fieldを
score生成には使わない。

- force component RMS disagreement
- 最大原子force-vector disagreement
- force方向差 `1 - cosine`
- 組成内offsetを除いたenergy disagreement
- 上記4指標のpercentile-rank平均

PFP recordの`model_version`、`calc_mode`、`pfp_api_client` versionと、
UMA evaluation/model IDを各score recordのprovenanceへ保存する。

## 3. engineering-50 dry-run

すでに観測済みの`engineering_50_v1`をplumbing fixtureとして再利用した。

| Stage | Result |
|---|---:|
| deterministic sampling | 50 accepted / 0 rejected |
| acquisition score records | 50 |
| policies | random / PFP-force-RMS / PFP-combined-rank |
| budget per policy | 10 |
| max per parent | 2 |
| union | 20（上限30） |

生成物はGit非追跡の次の場所に置いた。

```text
validation/matlantis_pfp/runs/engineering_50_acquisition_dry_run/
  scores.json
  selection.json
```

selectionを確定した後、既存GPU4PySCF値を用いて参考診断した。これは新しいHF計算を
行わず、selection実装が期待どおり高score側を選ぶかを見るためだけの結果である。

| Score/policy | Engineering diagnostic |
|---|---:|
| PFP–UMA force RMS score vs base UMA HF force RMSE Spearman rho | 0.2263 |
| PFP–UMA max-atom force score vs base UMA HF force RMSE Spearman rho | 0.3402 |
| PFP–UMA centered-energy score vs base UMA HF force RMSE Spearman rho | 0.6150 |
| combined-rank score vs base UMA HF force RMSE Spearman rho | 0.4175 |
| random: high-error top 10 capture | 2/10 |
| PFP force-RMS: high-error top 10 capture | 4/10 |
| combined-rank: high-error top 10 capture | 3/10 |

PFP force-RMS policyの選抜10件のbase UMA force RMSE平均は
0.04692 eV/angstrom、randomは0.02486 eV/angstromだった。PFP disagreementを
screening signalとして新しいpoolで検証する価値はある。ただし、この50件は既知の
5 parent、Cartesian displacementだけであり、bootstrapも独立testもないため、
Gate MF-Bを通過したとは扱わない。

## 4. fixed test候補源の確認

参照先`neb_arrhenius`には、neutral singletの次のPFP trajectoryが実在する。

- `SiH4 -> SiH2 + H2`: CI-NEB 9 images、IRC forward 30、reverse 60
- `GeHCl3 -> GeCl2 + HCl`: IRC forward 92、reverse 156

これらはreaction-path候補のengineering poolには使える。一方、独立な反応familyは
2つだけで、scientific fixed testを親構造単位で分けるには不足する。またSiH4は
`engineering_50_v1`にも含まれる。

fixed testは次の条件を満たすまでfreezeしない。

1. `engineering_50_v1`の5 parentを除外する。
2. selection poolとtestをreaction/parent family単位で分ける。
3. testは最低8 parent、各5構造を目安に40構造をscoreを見る前に固定する。
4. 同じtrajectoryのframeをpoolとtestへ分割しない。

したがって、既存2 reactionはtrajectory importerとunlabeled inferenceの
engineering fixtureとして先に利用し、科学pilotのHF label投入前にparent/reaction
familyを追加する。

## 5. unlabeled UMA prediction

新しいcandidate poolをGPU4PySCFでlabelする前にUMAでscoreできるよう、
`uma-pyscf predict-uma`を実装した。

- input: `uma-pyscf-candidate-manifest-v1`
- output: `uma-pyscf-model-predictions-v1`
- outputに含むもの: geometry、charge/multiplicity、UMA energy/forces、model/runtime provenance
- outputに含まないもの: GPU4PySCF reference energy/forces、error metric、QC verdict
- Slurm script: `scripts/slurm/run_predict_uma_candidates_softbank_slurm.sh`
- config: `configs/evaluation/mf_pfp_candidate_pool_base_uma_s_1p2_v1.yaml`

PFP acquisition score builderは、従来のengineering evaluation artifactに加え、
このunlabeled prediction manifestを直接読める。したがって新規poolでは
HF labelを見る前に`UMA -> PFP -> score -> selection`を完結できる。

## 6. PFP利用条件

Matlantisの公開案内では、利用規約Article 7.1にcustomer output dataの帰属を
明記したとされる。ただし、公開案内だけでは現在のtenant契約と、PFP出力を
内部ML選抜特徴量として継続保存する用途まで確定できない。

- 公開情報:
  https://matlantis.com/en/news/update-service-specification-20240401/
- 状態: tenant契約またはMatlantis窓口での限定確認待ち
- 影響: engineering comparisonは継続可能。新規HF budget確定とproduction policy採択は
  確認が済むまで行わない

## 7. 次の作業

1. 科学pilot用に独立parent/reaction familyを追加し、fixed testをfreezeする。
2. diversity descriptorを実装し、400件poolのgeneration mixをconfigへ固定する。
3. tenant契約上のPFP output内部利用・保存条件を確認する。
4. review後にだけ最大100件のGPU4PySCF label予算を確定する。

## 8. reaction trajectory importer実績

`uma-pyscf import-trajectory`を追加し、既存ASE trajectoryから参照labelを含まない
candidate manifestを生成できるようにした。source pathは論理相対pathとしてconfigへ
保存し、実行時に各trajectoryのSHA-256、全frame数、採択した元frame indexをmanifestへ
固定する。選抜側には`max_per_trajectory`も追加し、設定時にtrajectory provenanceが
欠けていればfail-closedとした。

2026-08-13の4本のIRCへ実行した結果は次の通り。

| trajectory | source frames | proposed | accepted |
|---|---:|---:|---:|
| SiH4 forward | 30 | 20 | 20 |
| SiH4 reverse | 60 | 20 | 19 |
| GeHCl3 forward | 92 | 20 | 20 |
| GeHCl3 reverse | 156 | 20 | 19 |
| **total** | **338** | **80** | **78** |

reverse側で棄却された2件はいずれもframe 0で、forward側frame 0と同じTS構造だった。
距離衝突による棄却はない。実行物はGit非追跡の
`validation/matlantis_pfp/runs/mf_neb_arrhenius_trajectory_pool_v1/`へ置いた。

SoftBank job `1801662`でbase UMA predictionを78/78件生成し、
`uma-pyscf-model-predictions-v1`としてschema検証した。MatlantisではPython 3.11を
有効化後、PFP v9.0.0 / `R2SCAN_PLUS_D3`を78/78件成功し、同一出力へのresumeで
0 completed / 78 skippedを確認した。

## 9. 78件poolのmulti-fidelity selection dry-run

PFP API wall timeは合計3.736秒、平均0.0479秒、median 0.0510秒/構造だった。
runtimeはPython 3.11.11、ASE 3.25.0、pfp-api-client 1.21.3。失敗は0件だった。

base UMAとPFPの78件予測だけからscoreを生成し、GPU4PySCF reference fieldを
一切読まずにselectionを確定した。

| Quantity | Result |
|---|---:|
| PFP-UMA force RMS disagreement median | 0.10770 eV/Angstrom |
| PFP-UMA force RMS disagreement max | 0.28638 eV/Angstrom |
| composition-centered energy disagreement median | 0.07911 eV |
| composition-centered energy disagreement max | 0.20025 eV |
| random selection | 10 |
| force-RMS selection | 10 |
| combined-rank selection | 10 |
| policy union | 22 |

各policyは2 parentから5件ずつ選び、各trajectoryは最大3件だった。
force-RMSとcombined-rankは7/10件が重なり、randomとの重複は0件または1件だった。
combined上位にはGeHCl3のTS近傍（forward frame 0/5、reverse frame 8/16）と、
SiH4のTS近傍（forward frame 0、reverse frame 3/6）が多く含まれた。

これはreaction coordinate上でPFP-UMA disagreementが非一様であることを示す
engineering evidenceである。ただしparentは2 reactionだけで、GPU4PySCF正解を
新規取得していないため、high-error捕捉率やGate MF-Bは判定しない。

主要artifact SHA-256:

| Artifact | SHA-256 |
|---|---|
| 10,000-step UMA evaluation | `0dd7fc58fddeee4f0a099785cb3257c0ff4a85999c18f08edef2fc2262c520b9` |
| 78-record base UMA predictions | `7f08d53e8c8156807bea846add8d71ef3de1540469ba00d406efaa330fd79c59` |
| acquisition scores | `a05ae1cce353a4f46dbdc6b68b20021e5562669717806ea7dfc3494f0fecff0c` |
| selection manifest | `35891773df0ad71dd5b3ff6f9984dee31cb6b2639d492d454fdb53b46d354db9` |
| sorted 78 PFP record digest lines | `6c4ace3213ed3d406a68a93492b40ee8e8975c14e379fac28ad522836861a028` |
