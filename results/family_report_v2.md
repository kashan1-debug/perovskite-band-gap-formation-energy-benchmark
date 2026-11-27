# Perovskite v2 — Stage 4 leave-one-family-out report

Magpie only (132 columns), e_form_per_atom, seed 0.
Models: Ridge and tuned XGBoost only — no Random Forest (118 retrains already; RF would make the run far too long).
A_family 'other' was NOT left out (14,560 rows, too coarse).

## Pooled headline (lead with this, not mean per-family R2)

family_type   model  n_test  r2_pooled  rmse_pooled  mae_pooled
          B   Ridge   18928   0.205610     0.132353    0.101563
          B XGBoost   18928   0.085721     0.141989    0.107224
          X   Ridge   18928  -9.958471     0.491576    0.147789
          X XGBoost   18928   0.375082     0.117389    0.089854

## Per-family R2 summary (secondary — context only)

family_type   model    mean_r2  median_r2     min_r2   max_r2 worst_family best_family
          B   Ridge  -0.774352  -0.469321  -5.103253 0.307712            W          Mg
          B XGBoost  -0.941065  -0.606987  -4.641594 0.431287           Re          Mg
          X   Ridge -12.519784  -0.168764 -84.580527 0.377401          FNO         N2O
          X XGBoost   0.283080   0.246427   0.181519 0.475931          FO2         N2O

## Per-family results (all families)

family_type family   model  n_test         r2     rmse      mae   experiment
          B     Ag   Ridge     364  -1.088167 0.126684 0.102658 reproduction
          B     Ag XGBoost     364  -0.873369 0.119992 0.096183 reproduction
          B     Al   Ridge     364  -0.558082 0.120384 0.100624 reproduction
          B     Al XGBoost     364  -0.127556 0.102410 0.082933 reproduction
          B     As   Ridge     364  -0.075895 0.090642 0.071835 reproduction
          B     As XGBoost     364   0.082153 0.083720 0.066245 reproduction
          B     Au   Ridge     364  -0.321208 0.090180 0.069817 reproduction
          B     Au XGBoost     364  -0.443637 0.094266 0.073380 reproduction
          B      B   Ridge     364  -3.171043 0.160669 0.150920 reproduction
          B      B XGBoost     364  -4.094393 0.177564 0.156656 reproduction
          B     Ba   Ridge     364  -1.477371 0.224438 0.191171 reproduction
          B     Ba XGBoost     364  -2.648416 0.272366 0.239452 reproduction
          B     Be   Ridge     364  -0.081624 0.096282 0.079586 reproduction
          B     Be XGBoost     364  -0.495429 0.113211 0.089159 reproduction
          B     Bi   Ridge     364  -0.082104 0.099789 0.079198 reproduction
          B     Bi XGBoost     364  -0.038363 0.097751 0.076811 reproduction
          B     Ca   Ridge     364  -0.161594 0.127232 0.096609 reproduction
          B     Ca XGBoost     364  -0.340042 0.136656 0.105109 reproduction
          B     Cd   Ridge     364  -0.195790 0.113842 0.089860 reproduction
          B     Cd XGBoost     364  -0.191174 0.113622 0.089881 reproduction
          B     Co   Ridge     364  -0.498119 0.085685 0.068529 reproduction
          B     Co XGBoost     364  -0.598965 0.088522 0.070617 reproduction
          B     Cr   Ridge     364  -1.227089 0.123068 0.101186 reproduction
          B     Cr XGBoost     364  -0.919702 0.114259 0.092031 reproduction
          B     Cs   Ridge     364  -1.367040 0.212703 0.178062 reproduction
          B     Cs XGBoost     364  -3.254649 0.285170 0.250718 reproduction
          B     Cu   Ridge     364   0.237128 0.073401 0.057189 reproduction
          B     Cu XGBoost     364   0.236810 0.073416 0.058138 reproduction
          B     Fe   Ridge     364  -0.752587 0.098623 0.078323 reproduction
          B     Fe XGBoost     364  -1.127113 0.108651 0.084372 reproduction
          B     Ga   Ridge     364   0.225581 0.090641 0.068821 reproduction
          B     Ga XGBoost     364   0.338783 0.083755 0.064253 reproduction
          B     Ge   Ridge     364  -0.279855 0.096385 0.076990 reproduction
          B     Ge XGBoost     364  -0.355392 0.099189 0.081078 reproduction
          B     Hf   Ridge     364  -0.392214 0.111953 0.089302 reproduction
          B     Hf XGBoost     364  -0.615010 0.120578 0.099598 reproduction
          B     Hg   Ridge     364  -0.633729 0.143481 0.117582 reproduction
          B     Hg XGBoost     364  -0.854832 0.152882 0.128269 reproduction
          B     In   Ridge     364   0.205316 0.096319 0.074911 reproduction
          B     In XGBoost     364   0.321067 0.089028 0.067828 reproduction
          B     Ir   Ridge     364  -0.860163 0.096046 0.077269 reproduction
          B     Ir XGBoost     364  -1.325256 0.107384 0.084471 reproduction
          B      K   Ridge     364  -1.773778 0.272671 0.233291 reproduction
          B      K XGBoost     364  -2.363970 0.300282 0.264738 reproduction
          B     La   Ridge     364  -0.536541 0.154388 0.128041 reproduction
          B     La XGBoost     364  -0.815603 0.167823 0.138040 reproduction
          B     Li   Ridge     364   0.099222 0.106997 0.085597 reproduction
          B     Li XGBoost     364   0.093795 0.107319 0.087452 reproduction
          B     Mg   Ridge     364   0.307712 0.108676 0.085399 reproduction
          B     Mg XGBoost     364   0.431287 0.098500 0.078187 reproduction
          B     Mn   Ridge     364  -0.234685 0.091981 0.071781 reproduction
          B     Mn XGBoost     364  -0.808591 0.111324 0.089119 reproduction
          B     Mo   Ridge     364  -2.352972 0.140784 0.114530 reproduction
          B     Mo XGBoost     364  -1.574542 0.123364 0.098541 reproduction
          B     Na   Ridge     364  -0.753943 0.154068 0.122666 reproduction
          B     Na XGBoost     364  -0.847672 0.158131 0.125463 reproduction
          B     Nb   Ridge     364  -2.070023 0.145628 0.123116 reproduction
          B     Nb XGBoost     364  -1.863304 0.140639 0.117765 reproduction
          B     Ni   Ridge     364   0.082627 0.071389 0.055740 reproduction
          B     Ni XGBoost     364  -0.022534 0.075370 0.060352 reproduction
          B     Os   Ridge     364  -1.342032 0.109911 0.087495 reproduction
          B     Os XGBoost     364  -1.719359 0.118434 0.094546 reproduction
          B     Pb   Ridge     364  -0.209336 0.107899 0.086266 reproduction
          B     Pb XGBoost     364  -0.172695 0.106252 0.085587 reproduction
          B     Pd   Ridge     364  -0.389351 0.089645 0.069493 reproduction
          B     Pd XGBoost     364  -0.180020 0.082616 0.064357 reproduction
          B     Pt   Ridge     364  -0.588444 0.087502 0.069603 reproduction
          B     Pt XGBoost     364  -0.471190 0.084211 0.066045 reproduction
          B     Rb   Ridge     364  -1.848175 0.267673 0.232509 reproduction
          B     Rb XGBoost     364  -3.063392 0.319716 0.289346 reproduction
          B     Re   Ridge     364  -2.893465 0.149668 0.120721 reproduction
          B     Re XGBoost     364  -4.641594 0.180161 0.147708 reproduction
          B     Rh   Ridge     364  -0.288624 0.083769 0.065788 reproduction
          B     Rh XGBoost     364  -0.493328 0.090177 0.072239 reproduction
          B     Ru   Ridge     364  -0.512557 0.094407 0.078030 reproduction
          B     Ru XGBoost     364  -0.664481 0.099034 0.081303 reproduction
          B     Sb   Ridge     364  -0.153429 0.103688 0.083426 reproduction
          B     Sb XGBoost     364  -0.028659 0.097919 0.078745 reproduction
          B     Sc   Ridge     364   0.268518 0.091182 0.070404 reproduction
          B     Sc XGBoost     364   0.263977 0.091465 0.069674 reproduction
          B     Si   Ridge     364  -1.746407 0.129125 0.109847 reproduction
          B     Si XGBoost     364  -1.416958 0.121133 0.100089 reproduction
          B     Sn   Ridge     364   0.057821 0.092340 0.069772 reproduction
          B     Sn XGBoost     364   0.122822 0.089098 0.068615 reproduction
          B     Sr   Ridge     364  -0.479083 0.184813 0.148241 reproduction
          B     Sr XGBoost     364  -0.845058 0.206415 0.170827 reproduction
          B     Ta   Ridge     364  -3.260993 0.171031 0.146784 reproduction
          B     Ta XGBoost     364  -2.949168 0.164654 0.143023 reproduction
          B     Te   Ridge     364  -0.280646 0.107379 0.086223 reproduction
          B     Te XGBoost     364  -0.109068 0.099927 0.079443 reproduction
          B     Ti   Ridge     364  -0.459559 0.112521 0.091762 reproduction
          B     Ti XGBoost     364  -0.994017 0.131519 0.112753 reproduction
          B     Tl   Ridge     364  -0.762639 0.166483 0.140976 reproduction
          B     Tl XGBoost     364  -0.473038 0.152194 0.123913 reproduction
          B      V   Ridge     364  -0.775126 0.115044 0.094050 reproduction
          B      V XGBoost     364  -1.077794 0.124466 0.101976 reproduction
          B      W   Ridge     364  -5.103253 0.193960 0.165397 reproduction
          B      W XGBoost     364  -4.474837 0.183703 0.153982 reproduction
          B      Y   Ridge     364   0.126223 0.115303 0.091326 reproduction
          B      Y XGBoost     364  -0.106705 0.129765 0.102924 reproduction
          B     Zn   Ridge     364   0.286195 0.081744 0.064987 reproduction
          B     Zn XGBoost     364   0.348871 0.078073 0.062143 reproduction
          B     Zr   Ridge     364  -0.123912 0.088873 0.067567 reproduction
          B     Zr XGBoost     364  -0.694062 0.109111 0.089572 reproduction
          X    FNO   Ridge    2704 -84.580527 1.235205 0.241389 reproduction
          X    FNO XGBoost    2704   0.215083 0.118294 0.095447 reproduction
          X    FO2   Ridge    2704  -0.168764 0.142486 0.114416 reproduction
          X    FO2 XGBoost    2704   0.181519 0.119237 0.095565 reproduction
          X    N2O   Ridge    2704   0.377401 0.124186 0.094989 reproduction
          X    N2O XGBoost    2704   0.475931 0.113937 0.088973 reproduction
          X     N3   Ridge    2704   0.205617 0.140960 0.111903 reproduction
          X     N3 XGBoost    2704   0.353620 0.127152 0.092331 reproduction
          X    NO2   Ridge    2704  -0.405396 0.157148 0.128876 reproduction
          X    NO2 XGBoost    2704   0.201000 0.118490 0.091653 reproduction
          X    O2S   Ridge    2704  -3.088714 0.260572 0.235789 reproduction
          X    O2S XGBoost    2704   0.307981 0.107200 0.075620 reproduction
          X     O3   Ridge    2704   0.021893 0.132712 0.107163 reproduction
          X     O3 XGBoost    2704   0.246427 0.116488 0.089389 reproduction

## GATE 4 — reproduction vs v1 Cell 11 pooled

  [PASS] B-out Ridge: pooled v2 0.206 vs v1 0.206 (diff -0.000)
  [PASS] B-out XGBoost: pooled v2 0.086 vs v1 0.086 (diff -0.000)
  [FAIL] X-out Ridge: pooled v2 -9.958 vs v1 -9.630 (diff -0.328)
  [PASS] X-out XGBoost: pooled v2 0.375 vs v1 0.375 (diff +0.000)

## Notes

- experiment column: 'reproduction' for the full run, 'smoke' for test runs.
- Pooled R2 is computed once on all pooled test predictions per (family_type, model); it is not the mean of per-family R2s.
- X-out Ridge pooled R2 ~ -9.6 is expected: linear extrapolation to a new anion fails catastrophically. Trees do not extrapolate, so XGBoost survives X-out while Ridge does not.
- If GATE 4 failed, STOP — do not proceed to Stage 5. Find why v2 differs from v1 first.

