# Si/Ge/H/Clクロスコード検証ラダー投入記録

- 日付: 2026-08-13
- 計算機: Ujilab1/Ujilab3（各64 CPU core）
- suite: `si_ge_h_cl_ladder_v1`
- 状態: ORCA 6.0.0 / CPU PySCF計算をPBSへ投入済み

## 目的と構成

既存のH2、SiH4、GeH4、SiCl4、GeCl4 seed比較を拡張し、次の29構造を固定した。
各構造についてORCAとCPU PySCFを同一geometry、charge、multiplicity、
ωB97M-V/def2-TZVPDで計算するため、計58 engine jobとなる。

| Category | 構造数 | 内容 |
|---|---:|---|
| bond scan | 12 | SiH4、GeH4、SiCl4、GeCl4の第1結合をseedの0.85、1.15、1.30倍 |
| radical | 4 | SiH3、GeH3、SiCl3、GeCl3の中性二重項・平面seed |
| mixed | 3 | H3Si-GeH3、H3Si-GeCl3、Cl3Si-GeH3のstaggered seed |
| random displacement | 10 | 5種の親構造へσ=0.04/0.12 Åの決定論的Gaussian変位 |

random displacementは全原子に変位を与えた後、剛体並進を除いた。random seed、変位幅、
親構造をXYZ commentとsuite manifestへ保存し、原子間距離が共有結合半径和の0.65倍を
下回る候補を拒否する。

## 計算資源

- H系中心の12構造: 8 CPU、32 GB、walltime 6 h
- Cl含有または8原子の17構造: 16 CPU、64 GB、walltime 12 h
- PySCF: thread数をPBS `ncpus`と一致
- ORCA: `%pal nprocs`をPBS `mpiprocs`と一致、`%maxcore 3000` MB/process
- ORCAはnode-local `/tmp` scratchを使用

16 CPU jobでもORCAの最大メモリ予算は48 GBであり、64 GB割当に収まる。投入直後は
Ujilab1/Ujilab3とも56/64 coreを使用し、各nodeに8 coreを残した。

## PBS投入

- 最初のORCA job: `1362.Ujilab`（SiH4 bond ×0.85、正常終了）
- 残りのjob: `1365.Ujilab`–`1421.Ujilab`
- 投入直後: running 13、queued 44、完了1
- submission receipt: `validation/orca_gpu4pyscf/runs/submissions/`（Git非追跡）

初回投入ではPySCFテンプレートの旧`nodes=1:ppn=4` directiveが残っていたため、
command-lineの`select`指定と競合して拒否された。ORCA 1362のみ受理・正常終了した時点で
停止し、テンプレートを`select=1:ncpus=4:mpiprocs=1:mem=16gb`へ統一した。
再投入scriptは既存normalized resultを検出して1362をskipし、重複なしで残りを投入した。

## 進捗確認

PBS状態はUjilabで次により確認する。

```bash
/usr/openpbs/bin/qstat -u seki
/usr/openpbs/bin/pbsnodes -aSj
```

結果が揃い始めたら、次で利用可能なpairを比較・集計する。

```bash
cd /home/seki/uma_pyscf
/home/seki/miniconda3/envs/qcthermo/bin/python \
  validation/orca_gpu4pyscf/summarize_suite.py \
  validation/orca_gpu4pyscf/suites/si_ge_h_cl_ladder_v1.json \
  --write-comparisons \
  --output validation/orca_gpu4pyscf/runs/si_ge_h_cl_ladder_v1_summary.json
```

特にradicalではSCF収束だけでなく、PySCFの`<S^2>`とtargetからの偏差を確認する。
scanでは絶対energy差に加えて、各code内のseed相対energyと勾配差の距離依存性を評価する。
