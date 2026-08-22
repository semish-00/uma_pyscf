# H2 ORCA 6.0.0–CPU PySCFクロスコード比較

- 日付: 2026-08-13
- 計算機: Ujilab1
- 状態: energy/analytic gradient比較合格

## 共通条件

- geometry: H–H = 0.74 Å
- charge: 0
- multiplicity: 1
- functional: ωB97M-V
- basis: def2-TZVPD
- SCF convergence: `1e-10 Eh`
- process/thread: 4

ORCA側はORCA 6.0.0、`VeryTightSCF DEFGRID3 SCNL NORI NOCOSX
NoAutoStart`を使用した。これはRI/COSX近似を外したクロスコード診断条件であり、
OMol25本番設定の完全な再現を主張するものではない。

PySCF側はPySCF 2.14.0、LibXC 7.0.0、RKS、grid level 5、VV10用NLC grid
level 5、`grid_response=True`、density fittingなしを使用した。

## 結果

| 項目 | CPU PySCF | ORCA 6.0.0 | 絶対差 |
|---|---:|---:|---:|
| Energy [Eh] | -1.160767969654182 | -1.160767976664 | 7.0098e-9 |
| H1 gradient z [Eh/bohr] | 0.001072777391 | 0.001072511417 | 2.6597e-7 |
| H2 gradient z [Eh/bohr] | -0.001072777391 | -0.001072511417 | 2.6597e-7 |

- gradient RMS difference: `1.5356e-7 Eh/bohr`
- gradient maximum absolute difference: `2.6597e-7 Eh/bohr`
- provisional comparison gate: PASS
- PySCF wall time (SCF + gradient): 23.38 s
- ORCA total run time: about 4 s

H2一例の速度差は起動、grid実装、コード固有最適化の影響が大きく、一般的な
ORCA/PySCF性能比としては扱わない。

## 実装上の知見

実際のORCA 6.0.0 `.engrad`は、座標blockの後に`The end of the file` markerを
出さずEOFで終了した。fixture由来のparserがmarkerを必須にしていたため、EOF終端に
対応し回帰testを追加した。Python 3.11で全11 testが合格した。

## 次のSi/Ge/H/Cl検証ラダー

比較構造は、目的と分布を区別して生成する。

1. **コード一致用の小分子**
   - SiH4, GeH4, SiCl4, GeCl4
   - SiH3/GeH3およびSiCl3/GeCl3 radical
   - H3Si–GeH3、H3Si–GeCl3などの混合分子
2. **制御された非平衡構造**
   - 各代表結合を平衡近傍の0.8–1.3倍へ伸縮
   - 角度変形、反転、解離方向
   - 反応物・生成物・遷移状態候補を結ぶ補間構造
3. **局所ランダム変位**
   - seed geometryへ複数振幅のCartesian displacement
   - 原子衝突、極端な孤立、重複構造をfilter
   - random seedと生成条件をmanifestへ保存
4. **多分子・高温由来構造**
   - MLIP/xTB等で高温MDを行い候補構造を生成
   - 距離・組成・局所環境で多様性選択してDFT labelを付与
   - 最終的な検証setは生成MDと独立に固定

最初から高温MDだけに依存すると、衝突構造が多すぎる、既存potentialのbiasを受ける、
どの自由度でコード差が出たか分かりにくい、という問題がある。そのため結合scanと
小振幅変位を先行し、高温MDは反応的・多分子領域を追加する第三段階とする。

## 品質管理

- 同一geometry、charge、multiplicityを両codeへ渡す。
- energyだけでなくanalytic gradientを必須にする。
- open-shellでは`<S^2>`、SCF収束、波動関数安定性を記録する。
- 各元素についてdef2-TZVPD availabilityを事前検査済み（H/Si/Ge/Cl）。
- ORCA–PySCF差と、同一code内のgrid/近似差を分離する。
- 学習用、モデル選択用、最終クロスコード検証用の構造を分離する。

## SiH4/GeH4パイロット結果

H2に続き、正四面体seedと1本のE–H結合を1.25倍へ伸ばした非平衡構造を比較した。
seedはコード一致検証用の近似構造であり、DFT最適化済み平衡構造ではない。

| Case | ORCA PBS | PySCF PBS | ORCA wall | PySCF wall | |ΔE| [Eh] | Gradient RMS [Eh/bohr] | Gradient max [Eh/bohr] | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SiH4 seed | 1350 | 1351 | 2:59 | 4:07 | 6.4144e-7 | 5.1409e-6 | 5.7477e-6 | PASS |
| SiH4 1 bond ×1.25 | 1352 | 1353 | 4:23 | 4:17 | 1.0907e-6 | 4.5051e-6 | 5.2709e-6 | PASS |
| GeH4 seed | 1354 | 1355 | 4:26 | 4:30 | 5.2291e-7 | 1.9202e-6 | 2.1470e-6 | PASS |
| GeH4 1 bond ×1.25 | 1356 | 1357 | 4:28 | 4:46 | 4.7511e-7 | 1.6588e-6 | 1.9567e-6 | PASS |

4 caseすべてORCA/PySCFともexit status 0で、暫定energy/gradient gateを通過した。
少なくとも閉殻Si–H/Ge–H系では、非平衡勾配を含めてクロスコード実装が整合している。

## SiCl4/GeCl4パイロット結果

正四面体の近似seed（Si–Cl = 2.020 Å、Ge–Cl = 2.110 Å）を同じ非RI診断条件で
比較した。どちらもDFT最適化済み平衡構造ではない。

| Case | ORCA PBS | PySCF PBS | ORCA PBS wall | PySCF wall (SCF + gradient) | CPU PySCF energy [Eh] | ORCA energy [Eh] | |ΔE| [Eh] | Gradient RMS [Eh/bohr] | Gradient max [Eh/bohr] | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SiCl4 seed | 1358 | 1359 | 5:04 | 876.29 s | -2130.478388520608 | -2130.478406032829 | 1.7512e-5 | 2.5729e-5 | 2.8766e-5 | PASS |
| GeCl4 seed | 1360 | 1361 | 5:06 | 921.36 s | -3917.844873137417 | -3917.844889571384 | 1.6434e-5 | 1.1954e-5 | 1.3365e-5 | PASS |

両caseともORCA/PySCFは正常終了し、暫定energy/gradient gateを通過した。Clを含む系では
SiH4/GeH4よりコード差が1–2桁大きくなったものの、energy差は約0.016–0.018 mEh、
gradient RMS差は約1.2–2.6e-5 Eh/bohrである。今後のscanでは、この差が結合距離や
grid設定に依存して系統的に増減しないか確認する。

非RIのCPU PySCFはSCF + gradientだけで約14.6–15.4分を要した。したがってCl系の
結合scanを多数点で行う前に、少数点でgrid収束とRI/密度fit近似の影響を分離して評価し、
本番データ生成で許容できる設定を決める。
