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

T0の非MD sourceは次の固定quotaで生成した。

- local Cartesian displacement: 40
- controlled M-X scan/dissociation: 48
- connected high-energy tail: 18
- HCl-elimination linear interpolation: 18
- blind finite-temperature MD: 76（job `1826995`完了後）

deterministic source job `1827105`は106/106、HCl interpolation job `1827104`は18/18を
生成QC通過した。MD完了後、dependency job `1827106`が合計200件をfreezeし、job
`1827107_[0-7]`が8-way GPU4PySCF label/QCを開始する。

## D-optimal arm

random/source-stratified baselineと同一quotaで比較できるよう、組成と元素対距離RBFだけを使う
model-independent greedy D-optimal orderingを追加した。teacher label、T0 error、UMA residualは
入力しない。既存のsource quota、parent cap、trajectory cap、state-aware duplicate除去は維持する。

12 training-side parentから192件を提案し、189件がgeometry QCを通過した。random 24件と
D-optimal 24件の重複4件を一度だけ計算する44-record unionをfreezeし、Slurm array job
`1827008`で44/44 label、44/44 QC acceptedとなった。T0 parentとの共有はない。

- union manifest SHA-256: `d9d6b20347654d892f68c935cfdb6dc2bc478d0165539b25d3b60e17efca073b`
- arm-membership report SHA-256: `63ee98f1e58d3a1a5b424dd981c4ee659f1b915778674331c1a7afbc1bf65d1e`
- label/QC audit SHA-256: `0802a1386e8013591c0e2e8ee02ae8cbea8b31945f4ea68dca48859e407c4a7d`
- QC-record checksum ledger SHA-256: `0f9736e1dda1e04c074f820e20d7ec4baa54ead596d9dc935ac154c04f464ed5`
