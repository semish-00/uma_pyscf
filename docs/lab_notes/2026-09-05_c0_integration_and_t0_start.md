# C0 integration audit and independent T0 start

## C0 training eligibility

2026-09-05にC0本体と追加のblind shell/hard-sourceを、record ID、QC、state-qualified
geometry fingerprint、parent provenanceで横断監査した。

- C0 source quota labels: 180
- C0 training-eligible: 179
- extra blind shell/hard-source QC-accepted: 45
- maximum integrated training-eligible: 224
- cross-source duplicate geometry groups: 0
- cross-source duplicate record-ID groups: 0

従来の「180 / 最大225」は、QC acceptedだった既知の
`c0_independent_reaction_paths_36_v1_sihcl3_to_sicl2_hcl_f00003`をそのまま数えた値だった。
この構造はdetached-Hを伴う`electronic_ambiguous` artifactなので、raw labelを保全したまま
training eligibilityから除外した。Si2H4Cl2 strong-distortion x0.70は、record固有のdirect parity
evidenceに基づき`valid_high_energy`として含めた。Si2H5Cl x0.70はx0.72 replacementに置換済みで、
元labelは削除せず隔離した。

監査artifact:

- report: `/lustre/user140002/runs/calibration/c0_integrated_teacher_pool_v1/20260905/c0_plus_shell_pool_audit.json`
- report SHA-256: `d5cbbf7f1de595bd1e05863afe4e4f61b1b5813cd3f59889f169717a5ed09868`

## C0-S direct sentinel

Slurm array job `1826977`でneutral doublet/quartet 24件をGPU direct SCFで再計算した。24/24が
収束し、density-fit基準に対して次を得た。

- maximum absolute energy difference: 2.60555 meV
- maximum gradient RMSE: 0.00793242 eV/angstrom
- maximum gradient component difference: 0.0214197 eV/angstrom
- maximum absolute S2 difference: 8.05e-6
- doublet/quartet ordering: 12/12 preserved

比較artifact:

- report: `/lustre/user140002/runs/calibration/si_ge_dimer_state_direct_audit_v1/20260905/direct_vs_density_fit_comparison.json`
- report SHA-256: `00350003aa7d9e964bdfe4768f3ac19b89a31940440cab9d109048bf6e3a0268`

これは計算protocolの安定性を支持するが、新しい電子状態のproduction承認ではない。C0-Sは
nonreleaseのまま保持し、occupation/root continuityを含むstate gateを別途通す。

## T0 start

C0、engineering-50、P0とparent IDを共有しないneutral-singletの8原子dimer親6件を構築し、
ASE XYZ write/read round trip、`pbc=False`、組成、原子数、最小原子間距離を検証した。parent registryと
XYZはcommit `712c1af`で固定した。blind base-UMA Langevin candidate generationをSlurm job
`1826995`として開始した。T0 parentはtrainingとacquisition-guided resamplingへ使用しない。

## D-optimal arm

random/source-stratified baselineと同一quotaで比較できるよう、組成と元素対距離RBFだけを使う
model-independent greedy D-optimal orderingを追加した。teacher label、T0 error、UMA residualは
入力しない。既存のsource quota、parent cap、trajectory cap、state-aware duplicate除去は維持する。
