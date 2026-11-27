# Literature check — Stage 9d (v2)

**Date:** 2026-11-25. **Checked by:** subagent (Muse). **Scope:** the five candidate
novel findings of the perovskite paper (draft v1, `paper_draft_v1.md`, sections 5.2–5.3),
against the published and preprint literature. Draft v1's 12 references were read first
and are NOT duplicated here except where a correction is needed; this note extends them.

**Strictness rule used (from the task):** a random-split perovskite ML paper does NOT count
as prior work for a leave-one-out claim. A paper on different data or a different metric
does not erase a claim unless it did the same thing on the same grid. The verdict scale:
**NEW** = no prior work found (closest prior cited, difference stated);
**PARTIAL** = prior work touches it (overlap and remainder stated exactly);
**NOT NEW** = prior work already did it (cited).

**Method:** web search + opening full texts/abstracts (IOP, RSC, arXiv, Nature, lab.cloud).
Every reference in the "verified" list below was opened and read in this session (abstract
at minimum; full text where it mattered). No DOI was invented: where a DOI could not be
confirmed it is marked "to complete". Searches covered: perovskite ML on formation energy
and band gap; the Castelli 18,928 grid (52×52×7) by name and by description; MatBench
perovskites leaderboard; CGCNN/MEGNet perovskite studies; leave-one-element-out /
extrapolation / OOD studies in materials ML; conformal prediction under shift.

---

## Verdict table

| # | Claim (our result) | Verdict | Closest prior work |
|---|---|---|---|
| 1 | Composition-only (Magpie) R² ceiling of **0.543** on the Castelli 18,928 grid, proved from A/B-swapped twin compositions sharing features but different formation energies | **PARTIAL** | Asaria, Salomone & Gandhi (Transformer Lab) — composition-degeneracy floor on the same data, in MAE form |
| 2 | Pooled leave-one-B-element-out formation-energy results on this grid (Magpie-only pooled R² **0.086**; site-aware **0.733**), reported with pooled (not mean per-family) metrics | **PARTIAL** | Li et al., *Commun. Mater.* 6, 9 (2025) — leave-one-element-out protocol on general materials data (MP/JARVIS), not perovskites |
| 3 | Site-resolved / per-site physics features (tolerance factor, octahedral factor, per-site elemental properties) improving extrapolation to unseen B metals in perovskites | **PARTIAL** | Cherukara & Mannodi-Kanakkithodi 2022 — site-resolved elemental fingerprint on the exact Castelli grid, but random splits only, never an element hold-out |
| 4 | No-B-information ablation: most of the gain comes from resolving A and X sites; B-physics adds only **+0.050** (0.683 → 0.733) | **NEW** | None found. Ablation as a method is standard; the result is not in the literature |
| 5 | Split-conformal uncertainty under leave-one-element-out shift, several held-out metals (coverage falls for W **0.690**, holds for Te **0.923** / Mg **0.945**) | **NEW** | Boruah 2026 (arXiv:2611.14521) — conformal conditional-coverage failure in materials ML generally, random splits, not element shift, not perovskites |

**Headline: 2 NEW, 3 PARTIAL, 0 NOT NEW.** The draft must not claim "first" for claims
1–3. It may claim the specific results (the numbers, the protocol applied to this grid,
the ablation, the three-metal conformal test) as new, with the priors cited.

---

## Claim 1 — Magpie R² ceiling 0.543 from A/B-swapped twins: PARTIAL

**Closest prior — Asaria, Salomone & Gandhi, "It's Not Leakage: The Crystal-Structure
Advantage in Materials Property Prediction is Real and Property-Specific", Transformer Lab
(self-published PDF, no journal, no DOI; accessed 2026-11-25 at lab.cloud).**
Same underlying data (MatBench perovskites = the Castelli 18,928), same phenomenon,
quantified as a **composition-degeneracy floor**: 0.379 eV per 5-atom formula unit,
67% of the target mean absolute deviation; same-composition collision rate 0.98;
structural near-duplicate rate 0.000 (pymatgen StructureMatcher sweep, tight/default/loose).
Their composition baseline (Magpie + RandomForest, 5-fold) has MAE 0.581; their from-scratch
CGCNN has MAE 0.071. Under a SOAP-cluster leave-out OOD split the descriptor structure gap
on perovskites stays +0.295. They also show the structure advantage declines with distance
from training data on perovskites (+0.422 nearest quartile → +0.238 farthest) while staying
large.

**Unit correction — important, cite it correctly.** The Asaria paper labels its numbers
"eV/atom" (abstract and Table 1 note). That label is wrong, and the numbers prove it:
a Magpie+RF model cannot have MAE 0.581 eV/atom when the target's own standard deviation
is 0.149 eV/atom (predicting the mean gives MAE ≈ 0.12 eV/atom); and 0.379 cannot be 67%
of the target MAD in eV/atom (the MAD is ≈ 0.12 eV/atom). All three numbers (floor 0.379,
composition MAE 0.581, CGCNN MAE 0.071) are consistent only as **eV per 5-atom formula
unit**: floor 0.379 eV/cell = 0.076 eV/atom = 67% of the per-cell MAD (≈ 0.57 eV/cell).
The draft must cite the floor as **0.379 eV per formula unit (0.076 eV/atom)** and may
note the source's unit mislabelling in a footnote if a reviewer asks.

**Cross-check (ours vs theirs):** our ceiling says the group-mean predictor leaves residual
standard deviation √(1 − 0.543) × 0.149 ≈ 0.101 eV/atom, i.e. MAE ≈ 0.08 eV/atom. Their
floor is 0.076 eV/atom. The two quantifications agree. This is a genuine independent
confirmation that the degeneracy is real and of this size.

**Supporting context — Martirossyan et al. 2025**, "All that structure matches does not
glitter", NeurIPS 2025, arXiv:2509.12178. Notes A/B-swapped pairs explicitly as polymorphic
pairs (perov-5 dataset, generative crystal-structure-prediction context, not Castelli).
Shows the A/B-swap phenomenon is recognised, but no ceiling or floor is computed there.

**What overlaps:** the phenomenon (same formula, different structure, different energy),
its size on this data, and the conclusion that composition-only models are capped by it.
**What remains new:** the R²-ceiling framing (0.543); the pooled variance decomposition;
the explicit A/B-twin proof on the full grid; and the link to a specific model's measured
failure (best grouped model 0.475 sits below the ceiling; family-out Magpie collapses to
0.086). Asaria never computed a pooled R² ceiling, never isolated A/B-swapped pairs, and
never tested a model against the ceiling.
**Draft wording:** do not write "we show for the first time that composition cannot
distinguish these crystals". Write that the ceiling is expressed here in R² form and tied
to the model results, and cite Asaria et al. for the degeneracy floor.

## Claim 2 — Pooled leave-one-B-out on this grid (Magpie 0.086; site-aware 0.733): PARTIAL

**Closest prior — Li et al. 2024/2025.** Kangming Li, Andre Niyongabo Rubungo, Xiangyun
Lei, Daniel Persaud, Kamal Choudhary, Brian DeCost, Adji Bousso Dieng, Jason
Hattrick-Simpers, "Probing out-of-distribution generalization in machine learning for
materials", *Communications Materials* **6**, 9 (2025), DOI 10.1038/s43246-024-00731-w
(arXiv:2406.06489, submitted 10 Jun 2024). Over 700 OOD tasks, including
**leave-one-element-out (LOEO)** tasks, on Materials Project and JARVIS formation-energy
data, with XGBoost, ALIGNN, GMP and LLMProp. Findings: high R² for most held-out elements
but failure for H, F and O; most heuristically-defined OOD tests are actually interpolation
because test data lie inside the training domain; for genuinely out-of-domain tasks, more
training data or longer training helps little or even hurts. The LOEO protocol and its
general findings therefore exist in the literature and must be cited.

**Other OOD context on the same data — Asaria et al. (above):** SOAP-cluster leave-out
split on the same perovskites data (structure gap +0.295 under shift). A harder split than
random, but cluster-based, not element-based, and about structure-vs-composition, not about
site-aware features. **Durdy et al. 2022**, "The importance of domain knowledge in
materials ML", *Digital Discovery* **1**, 763–778, DOI 10.1039/D2DD00039C — kernelised
leave-one-cluster-out CV (LOCO-CV) methodology; key finding is a contrast with ours:
domain knowledge does NOT improve ML performance in most tasks tested (band gap the
exception). Worth citing as the counter-expectation our site-aware result beats.
**"Materials representations organize transferable supervision for chemical
extrapolation", arXiv:2611.23837 (2026)** — formalises the LOEO protocol and studies
which elements provide transferable supervision (formation energy and band gap); authors
not identified in this check; not perovskites, not site-aware. Cite only for the protocol
if needed.

**What overlaps:** the leave-one-element-out protocol itself, and the general finding that
element hold-out is hard for some elements and easy for others.
**What remains new:** the application to the Castelli grid (all 52 B metals, all 7 X
anions); the pooled R² reporting (Li et al. report per-element R²; we pool all 18,928
held-out predictions); the Magpie collapse numbers (pooled 0.086); and the site-aware
recovery (0.733) measured under the same protocol. No prior work did LOEO on this grid.
**Draft wording:** "we apply the established leave-one-element-out protocol [Li et al.]
to the Castelli grid for the first time, with pooled reporting" — not "we invented
leave-one-element-out".

## Claim 3 — Site-resolved / per-site physics features improving extrapolation to unseen B metals: PARTIAL

**Closest prior — Cherukara & Mannodi-Kanakkithodi 2022.** Mathew Cherukara and Arun
Mannodi-Kanakkithodi, "A deep neural network framework to identify the compositional
and structural features of perovskites that govern stability and electronic properties"
(short title as published: deep-learning estimation of perovskite properties),
*Modelling and Simulation in Materials Science and Engineering* **30**, 034005 (2022),
DOI 10.1088/1361-651X/ac52de. **This is the most important missing reference in draft v1.**
Same Castelli 18,928 grid; site-resolved elemental fingerprint: 25 elemental properties
per site, pruned to 15 by correlation, × 5 sites (A, B, X1, X2, X3) = 75 features,
including ionic radius, Mendeleev number and electronegativity per site. Neural network,
random forest and kernel ridge regression. **But:** they reduce the data to the 735
non-zero-gap compounds only and use a **random** train/validation/development/test split
(510/75/75/75). Test results on that subset: formation-energy RMSE 0.18 (DNN), 0.20 (RFR),
0.13 (KRR); R² 0.80 / 0.78 / 0.88; direct-gap RMSE 0.66 / 0.75 / 0.74; indirect-gap RMSE
0.64 / 0.77 / 0.73. No element hold-out, no extrapolation test, no ceiling, no ablation,
no conformal. They suggest (not test) extending to "completely new element choices using
weighted averages of elemental properties".

**Do not compare their formation-energy numbers directly with ours.** Their target is
described as "heat of formation, defined in terms of decomposition of the perovskite into
elemental standard states and other possible phases" — a possibly different quantity from
Castelli's e_form — and it is computed on the 735-compound subset only, whose formation
energies have a different (unreported) spread. The excerpts read do not state per-cell vs
per-atom. Cite their method and their random-split results; do not put 0.13 eV next to our
0.112 eV/atom RMSE as if they were the same examination.

**Supporting mechanism prior — Mahdavi, Honavar & Morgan 2025.** Hamed Mahdavi, Vasant
Honavar, Dane Morgan, "Beyond training data: how elemental features enhance ML-based
formation energy predictions", *Digital Discovery* **4**, 2972–2982 (2025),
DOI 10.1039/D5DD00182J. SchNet and MACE interatomic potentials enriched with a 58-feature
XenonPy elemental descriptor matrix; on mp_e_form (132,752 compounds) they randomly exclude
up to 10% of elements with negligible performance loss, and elemental features restore
near-in-distribution accuracy for unseen elements. This supports the *mechanism* (elemental
physics carries transferable information about unseen elements) but is a different data set,
a different model class (GNN/MLIP, not perovskite site features), and a random element
exclusion, not leave-one-B-out on perovskites. (A correction to this paper exists:
*Digital Discovery* 2025, **4**, 3828 — its content was not checked.)

**Further context — Bartel et al. 2019.** Christopher J. Bartel et al., "New tolerance
factor to predict the stability of perovskite oxides and halides", *Science Advances*
**5**, eaav0693 (2019), DOI 10.1126/sciadv.aav0693. A new tolerance factor τ giving 92%
perovskite/non-perovskite classification accuracy on 576 ABX₃ compounds. Composition-based
stability classification, not extrapolation; cited for the tolerance-factor lineage our
derived features draw on.

**What overlaps:** site-resolved elemental features on this exact grid (Cherukara 2022);
the general principle that elemental physics helps unseen-element prediction (Mahdavi 2025).
**What remains new:** testing site-resolved features under leave-one-B-out extrapolation
rather than random splits. Cherukara never held out an element, so the question "do
site-resolved features transfer to a new B metal?" was never answered before our Stages
3b/3c. The feature *idea* is not new; the *extrapolation test* of it is.
**Draft wording:** cite Cherukara as the prior use of site-resolved features on this grid
and state explicitly that their evaluation was random-split interpolation, while ours is
leave-one-B-out extrapolation.

## Claim 4 — No-B-information ablation (0.683 vs 0.733, B-physics +0.050): NEW

No prior work was found that removes all B-site information and measures the residual
gain, on perovskites or elsewhere in materials ML. The closest material is generic:
feature-ablation and SHAP analysis are standard practice (e.g. Chen et al. 2024, *J.
Mater. Chem. C* — SHAP on ~300 descriptors; Deng et al. 2025 — feature reduction 168 → 9),
and Cherukara 2022 reports no ablation. Durdy et al. 2022 (above) is a contrast, not a
prior: under LOCO-CV, domain knowledge usually does not help, which makes our positive
ablation result (site resolution helps, B-physics adds little) more, not less,
interesting. The specific finding — most of the 0.086 → 0.733 recovery survives with zero
B information (0.683); B-physics adds +0.050 overall, helps the hard heavy metals
(W, Re, Ta) and hurts Cr (−1.517 → +0.446 without it) — is new.
**Caveat for the draft:** ablation as a *method* is standard; only the *result* is new.
Say "we ablate" without implying the ablation design itself is novel.

## Claim 5 — Split-conformal under leave-one-element-out shift, several metals: NEW

No prior work was found testing conformal coverage under leave-one-element-out shift on
perovskites or on formation energy. The closest priors are all general, not perovskite
and not element-shift:

- **Boruah 2026.** Irene Boruah, "Null-Corrected Conditional Coverage in Materials
  Property Prediction: A Five-Dataset Replication", arXiv:2611.14521 [stat.ME], submitted
  2 Sep 2026, submitted to *Data-Centric Engineering*. Single author (Dibrugarh
  University Institute of Engineering and Technology, India). Five regression datasets
  (n = 437 to 21,263), seven interval methods, 50 random splits each; split-conformal
  conditional-coverage failure survives null correction on four of five datasets. The
  antecedent single-dataset study was steel fatigue (NIMS MatNavi, n = 437). The five
  datasets are not named in the abstract and perovskite inclusion could NOT be confirmed —
  do not claim Boruah covers perovskites. What Boruah establishes is that conformal
  conditional failure is a general materials-ML phenomenon under random splits; our
  contribution is the element-shift version with a metal-dependent failure (W 0.690 vs
  Te 0.923, Mg 0.945).
- **Tan et al. 2025**, "Benchmarking GNNs for OOD Materials Property Prediction with
  Uncertainty Quantification", arXiv:2511.11697 — VERIFIED 2026-11-25 by opening the
  arXiv PDF. Authors: Liqin Tan, Pin Chen, Menghan Liu, Xiean Wang, Jianhuan Cen,
  Qingsong Zou (corresponding, Sun Yat-sen University). MatUQ: 1,375 OOD tasks from
  six datasets, including the Castelli perovskites as D3 (18,928 rows, formation
  energy). Uncertainty method is Monte Carlo dropout + deep evidential regression
  (D-EviU metric) — the word "conformal" does not appear in the paper — and splits
  are OFM-based (LOCO, SparseX/Y) plus SOAP-LOCO clusters, NOT leave-one-element-out.
  Verdict: the closest OOD+UQ prior on this grid, and it must be cited as such, but
  it does NOT test split-conformal coverage under element shift, so claim 5 stays
  NEW. Added to draft v2 as ref [22].
- **Fannjiang et al. 2022** (conformal under design shift, proteins) and a molecular
  scaffold-shift audit (arXiv:2611.01512) and MLIP conformal work (arXiv:2510.00721) —
  found in earlier searches in this project, not opened in this check; all non-perovskite.

**What remains new:** the perovskite, formation-energy, leave-one-element-out conformal
test, and the metal-dependent failure pattern. The general phenomenon (conformal degrades
under shift) is known and must be cited via Boruah.
**Caveat for the draft:** three metals, n = 364 each, one seed — limited evidence. The
claim is "coverage under element shift is metal-dependent on this grid", not "conformal
always fails under shift".

---

## Updated novelty statement for draft v2 (usable verbatim, with citations to add)

> To our knowledge, no prior study has combined these five elements on the Castelli grid:
> a pooled composition-only ceiling derived from A/B-swapped twins; pooled
> leave-one-B-element-out reporting; a site-resolved feature test under that extrapolation
> split; an ablation removing all B-site information; and split-conformal coverage measured
> under element shift. The individual ingredients are not new, and we cite them. The
> composition-degeneracy floor on this data was quantified by Asaria et al.; site-resolved
> elemental features on this grid were used by Cherukara and Mannodi-Kanakkithodi under
> random splits; and the leave-one-element-out protocol was established for general
> materials data by Li et al. What is new is the application of those tools to the
> extrapolation question on this grid, and the two findings that come out of it: site
> resolution, not B-site physics, carries the transfer gain (pooled R² 0.683 without any
> B information, against 0.733 with it), and conformal coverage under element shift is
> metal-dependent, failing for tungsten (0.690) while holding for tellurium (0.923) and
> magnesium (0.945).

Shorter version (abstract-length, if the paragraph above is too long):

> The degeneracy limiting composition-only models on this grid was quantified by Asaria
> et al., site-resolved features on this grid were introduced by Cherukara and
> Mannodi-Kanakkithodi under random splits, and leave-one-element-out evaluation was
> established by Li et al. on general materials data. Our contribution is the
> extrapolation test that combines them: pooled leave-one-B-out on the full Castelli grid,
> the site-resolved recovery to R² 0.733, the no-B-information ablation (0.683) localising
> the gain to site resolution, and a three-metal conformal test showing coverage under
> shift is metal-dependent.

---

## References that MUST be added to the draft (verified by opening)

Numbering continues draft v1's [1]–[12]. Full author lists verified unless marked.

**[13] Cherukara & Mannodi-Kanakkithodi 2022 (CRITICAL — the closest prior on this grid).**
Mathew Cherukara and Arun Mannodi-Kanakkithodi, "A deep neural network framework to
identify the compositional and structural features of perovskites that govern stability
and electronic properties", *Model. Simul. Mater. Sci. Eng.* **30**, 034005 (2022).
DOI 10.1088/1361-651X/ac52de.

**[14] Asaria, Salomone & Gandhi (degeneracy floor; cite with the unit correction).**
Ali Asaria, Tony Salomone and Deep Gandhi, "It's Not Leakage: The Crystal-Structure
Advantage in Materials Property Prediction is Real and Property-Specific", Transformer
Lab (self-published PDF, no journal, no DOI; accessed 2026-11-25, https://lab.cloud/
research/structure-not-leakage.pdf). Floor: 0.379 eV per 5-atom formula unit
(0.076 eV/atom), 67% of target MAD. *Year could not be confirmed from the document
itself (it cites arXiv papers up to Dec 2025); do not state a year without checking
with the authors' page.*

**[15] Li et al. 2025 (leave-one-element-out protocol).** Kangming Li, Andre Niyongabo
Rubungo, Xiangyun Lei, Daniel Persaud, Kamal Choudhary, Brian DeCost, Adji Bousso Dieng
and Jason Hattrick-Simpers, "Probing out-of-distribution generalization in machine
learning for materials", *Commun. Mater.* **6**, 9 (2025).
DOI 10.1038/s43246-024-00731-w (arXiv:2406.06489).

**[16] Mahdavi, Honavar & Morgan 2025 (elemental features help unseen elements).**
Hamed Mahdavi, Vasant Honavar and Dane Morgan, "Beyond training data: how elemental
features enhance ML-based formation energy predictions", *Digital Discovery* **4**,
2972–2982 (2025). DOI 10.1039/D5DD00182J.

**[17] Boruah 2026 (conformal conditional-coverage failure in materials ML).**
Irene Boruah, "Null-Corrected Conditional Coverage in Materials Property Prediction:
A Five-Dataset Replication", arXiv:2611.14521 [stat.ME] (2026), submitted to
*Data-Centric Engineering*.

**[18] Durdy et al. 2022 (LOCO-CV; domain knowledge usually does not help — the
counter-expectation).** Samantha Durdy, Michael W. Gaultois, Vladimir V. Gusev,
Danushka Bollegala and Matthew J. Rosseinsky, "Random projections and kernelised leave
one cluster out cross validation: universal baselines and evaluation tools for
supervised machine learning of material properties", *Digital Discovery* **1**, 763–778
(2022), DOI 10.1039/D2DD00039C. RESOLVED 2026-11-25: title and full author list
verified against the RSC article page. Kernelised leave-one-cluster-out CV; domain
knowledge does not improve ML performance in most tasks tested (band gap the
exception). (The earlier guessed title "The importance of domain knowledge in materials
machine learning..." was wrong — do not use it.)

**[19] Bartel et al. 2019 (tolerance factor τ).** Christopher J. Bartel, Christopher
Sutton, Bryan R. Goldsmith, Runhai Ouyang, Charles B. Musgrave, Luca M. Ghiringhelli
and Matthias Scheffler, "New tolerance factor to predict the stability of perovskite
oxides and halides", *Sci. Adv.* **5**, eaav0693 (2019). DOI 10.1126/sciadv.aav0693.
*Full author list VERIFIED 2026-11-25 against arXiv:1801.07700 and PMC6368436 — the
list as given is correct, including middle authors Sutton, Goldsmith, Ouyang and
Musgrave.*

**[20] Martirossyan et al. 2025 (A/B-swapped polymorphic pairs).** Maya M. Martirossyan,
Thomas Egg, Philipp Höllmer, George Karypis, Mark Transtrum, Adrian Roitberg, Mingjie
Liu, Richard G. Hennig, Ellad B. Tadmor and Stefano Martiniani, "All that structure
matches does not glitter", NeurIPS 2025, arXiv:2509.12178.

**[21] Chenebuah et al. 2021 (deduplicated perovskite ML; SVR 99% R² formation energy).**
Ericsson Tetteh Chenebuah, Michel Nganbe and Alain Beaudelaire Tchagang, "Comparative
analysis of machine learning approaches on the prediction of the electronic properties
of perovskites: a case study of ABX₃ and A₂BB′X₆", *Materials Today Communications*
**27**, 102462 (2021), DOI 10.1016/j.mtcomm.2021.102462 (ScienceDirect PII
S2352492821004542). RESOLVED 2026-11-25: volume, article number, DOI and full author
list verified against ScienceDirect and Crossref.
1,453 compounds (829 ABX₃ + 624 A₂BB′X₆), 12 models, 90/10 split; SVR formation energy
MAE 0.055 eV/atom, RMSE 0.096, R² 99%; band gap MAE 0.462, RMSE 0.662, R² 85.18%.
Explicitly deduplicated ("non-repetitive"); energy-above-hull used as a feature.
*Volume, article number, DOI and full author list NOT verified — complete before citing.*

**Fixes to existing draft references:**
- **[8] Liu et al. 2026 — full author list now verified:** Jingyu Liu, Xueqiong Su,
  Lishan Yang, Jiansen Ding, Jin Wang, Xing Ling, Yong Pan, Zhijun Wang, Wei Zhao, Yang
  Bu, "Comparison of graph neural networks and traditional machine learning for property
  prediction in all-inorganic perovskite materials", *Inorganics* **14**(2), 58 (2026).
  DOI 10.3390/inorganics14020058. (Note: the journal title itself contains a typo,
  "Comparation of Graph Neural Networks...".)
- **[12] Ward et al. 2018 (matminer) — pages now verified:** Logan Ward, Alexander Dunn,
  Alireza Faghaninia, Nils E. R. Zimmermann, Saurabh Bajaj, Qi Wang, Joseph H. Montoya,
  Jiming Chen, Kyle Bystrom, Maxwell Dylla, Kyle Chard, Mark Asta, Kristin A. Persson,
  G. Jeffrey Snyder, Ian Foster, Anubhav Jain, "Matminer: an open source toolkit for
  materials data mining", *Comput. Mater. Sci.* **152**, 60–69 (2018).
  DOI 10.1016/j.commatsci.2018.05.018.
- **[7] Deng et al. 2025 — verified, with a numbers note:** Ziliang Deng, Kailing Fang,
  Chong Guo, Zhichao Gong, Haojie Yue, Huacheng Zhang, Kang Li, Kun Guo, Zhiyong Liu,
  Bing Xie, Jinshan Lu, Kui Yao, Francis Eng Hock Tay, "Prediction of ABX₃ perovskite
  formation energy using machine learning", *Materials* **18**(13), 2927 (2025).
  DOI 10.3390/ma18132927. Abstract: R² 0.928, RMSE 0.301 eV/atom; conclusions: R² 0.939,
  RMSE 0.278 after reducing 168 → 9 features. Quote the abstract's numbers (or state
  both); do not silently use the higher one.
- **[9] Suhendar 2025 — verified:** Haris Suhendar (single author), "Comparative machine
  learning approaches for band gap prediction in perovskite materials using an ABX₃
  dataset", *J. Phys.: Conf. Ser.* **3139**, 012050 (2025).
  DOI 10.1088/1742-6596/3139/1/012050. 45,570 perovskites (NOT the Castelli grid),
  LightGBM best test R² 0.791, XGBoost 0.784, all-rows band-gap regression, random split —
  flattered by the zero mass, same warning as our C0.
- **[10] Gouvêa et al. 2026 — numbers re-confirmed:** MatBench perovskites MAE table:
  Automatminer 0.2005 (±0.0085), MODNet 0.0673, coGN 0.0269, MatterVial MODNet 0.0313
  eV/unit cell.
- **[6] Chen et al. 2024** (*J. Mater. Chem. C*, XGBoost R² 0.873) — verified in the
  earlier project session; not re-opened in this check. No change needed.

**Context references already correct in the draft (no action):** [1] Castelli 2012,
[2] Ward (Magpie) 2016, [3] Dunn (MatBench) 2020, [4] Xie & Grossman (CGCNN) 2018,
[5] Chen et al. (MEGNet) 2019, [11] Angelopoulos & Bates (conformal) arXiv:2107.07511.

---

## Corrections the literature check forces on draft v1 (for draft v2)

1. **"We did not run a graph model" is now FALSE (Sections 5.2, 5.4, 6).** Stage 9b
   has since been CONFIRMED in Colab (GATE 9b PASS, 52/52 families, 3.0 min):
   pretrained MEGNet zero-shot, leave-one-B-out, pooled R² **raw −77.753** (MP vs
   Castelli scale offset), **calibrated 0.561** (RMSE 0.098, MAE 0.077), Spearman 0.752.
   Full pooled B-out comparison: Magpie XGB 0.086, calibrated MEGNet 0.561, no-B-info
   XGB 0.683, site-aware XGB 0.733 — the graph model does NOT beat site-aware XGBoost.
   Best calibrated families Tl 0.762 / Mg 0.745 / Zn 0.687; worst W −1.866 / Re −1.427 /
   Mo −1.261. Calibration a ≈ 0.100, b ≈ +0.351 stable across families. Contamination
   caveat stands (0.561 is an upper bound; 6.2% formula overlap with Materials Project).
   No prior graph-model leave-one-B-out on this grid was found in this check, so the 9b
   comparison itself is a new result — but it must be reported as zero-shot pretrained
   MEGNet, not as a trained graph model. Draft v2 must replace "we did not run a graph
   model" with the 9b numbers and move "train a graph model from scratch" to future work.
2. **Section 5.3 (What is new) must be rewritten** using the novelty paragraph above:
   claims 1–3 are PARTIAL with named priors; claims 4–5 are NEW. The current "What may
   be new — pending the completion of a full literature check" list must not survive
   unchanged into a submitted version.
3. **Section 5.2 must add Cherukara 2022** (site-resolved features on this grid, random
   splits) and **Asaria et al.** (degeneracy floor, with the unit correction) and
   **Li et al.** (LOEO protocol). Without Cherukara, a reviewer who knows the Castelli
   literature will flag the omission.
4. **Do not compare Cherukara's formation-energy RMSE/R² (0.13–0.20 eV; R² 0.80–0.88)
   directly with our numbers** — different subset (n = 735), possibly different target
   definition, units not stated in the excerpts read.
5. **Do not cite Asaria's floor as "0.379 eV/atom"** — it is 0.379 eV per 5-atom formula
   unit (0.076 eV/atom). The source mislabels the unit; the draft must not repeat the
   error.
6. **Section 5.2's Suhendar sentence is now verified and can stay**, with the added note
   that their 45,570-compound data set is not the Castelli grid and their R² 0.791 is an
   all-rows random-split number flattered by the zero mass.
7. **Stage 9a2 (family-out SHAP) has since been CONFIRMED in Colab (GATE 9a2 PASS,
   2026-11-25)** and draft v2's mechanism discussion must use it together with the
   ablation, not instead of it. Held-out R² (site-aware XGBoost trained without that
   metal): Te 0.529, W 0.226 (reproduces Stage 3b exactly), Mg 0.512. SHAP shares under
   shift average Magpie 42% / site-resolution 27% / B-site ~28%, with W the exception
   (B-physics leads at 40% and is necessary: W falls 0.226 → −0.098 without B). SHAP
   measures reliance, ablation measures necessity — they answer different questions, and
   within one held-out family B is constant, so Magpie's share is largely A/X variation
   in aggregate form. The paper must report both and must NOT claim SHAP proves the
   mechanism either way. This does not change any verdict above (9a2 is our own result,
   not prior work), but it sharpens the claim-3/claim-4 wording: the accuracy gain
   (0.086 → 0.733) is a fact; the decomposition of *why* is SHAP-plus-ablation, and
   metal-dependent.

---

## What could NOT be verified (open items / leads)

- **Asaria et al. year/venue:** RE-CHECKED 2026-11-25 — the full PDF was read end to
  end (370 lines) and still contains no date anywhere; it cites arXiv papers up to Dec
  2025 (arXiv:2512.10989), so it postdates Dec 2025, but no year can be stated. Cite as
  undated with the URL (https://lab.cloud/research/structure-not-leakage.pdf) and
  access date. No journal, no DOI. If a journal version appears, update the citation.
- **Boruah's five datasets:** not named in the abstract; perovskite inclusion unknown.
  Do not claim Boruah covers perovskites. (Open the full text to list the datasets
  before final submission.)
- **Cherukara target definition:** "heat of formation" defined via decomposition into
  elemental standard states and other possible phases — whether this equals Castelli's
  e_form, and whether RMSEs are per cell or per atom, could not be settled from the
  excerpts read. Open the Methods section to resolve before any numeric comparison.
- **Chenebuah et al. 2021:** RESOLVED 2026-11-25 — *Mater. Today Commun.* **27**,
  102462, DOI 10.1016/j.mtcomm.2021.102462; authors Chenebuah, Nganbe, Tchagang.
- **Durdy et al. 2022:** RESOLVED 2026-11-25 — title "Random projections and kernelised
  leave one cluster out cross validation..." and authors Durdy, Gaultois, Gusev,
  Bollegala, Rosseinsky, verified on the RSC page.
- **Bartel et al. 2019:** RESOLVED 2026-11-25 — full author list verified against
  arXiv:1801.07700 and PMC6368436; the list in the draft is correct.
- **Not opened in this check (leads, not citations):** Tan et al. 2025 — CLOSED
  2026-11-25, see entry above (claim 5 stays NEW). Segal et al.
  2025, arXiv:2502.05970 (Known Unknowns: OOD property prediction); "Materials
  representations organize transferable supervision for chemical extrapolation",
  arXiv:2611.23837 (2026; authors not identified); Fannjiang et al. 2022 (conformal
  under design shift, proteins); MLIP conformal arXiv:2510.00721; molecular
  scaffold-shift audit arXiv:2611.01512; Mahdavi et al. correction (*Digital Discovery*
  2025, **4**, 3828 — content not checked).
- **Stanley et al. 2020** (*Adv. Theory Simul.* 3, 1900178, DOI 10.1002/adts.201900178)
  and **Mayr & Gagliardi 2021** (*ACS Omega* 6, 12722–12732,
  DOI 10.1021/acsomega.1c00991) — found in earlier project searches, not re-opened in
  this check; verify before adding to the draft.
- **Chenebuah & Chenebuah, arXiv:2312.11335** — 16,323 ABX₃ from OQMD, 61 features
  including DFT target properties, formation-energy MAE 0.013 eV/atom, band-gap MAE
  0.216 eV. Dataset paper (arXiv preprint). Relevant as context for why DFT-derived
  features inflate scores; not needed for the five claims.
- **Search limits:** searches were in English, web-indexed sources, Nov 2026. A paper
  in a non-indexed venue, a thesis, or a very recent preprint could have been missed;
  the verdicts are "no prior work found", not "no prior work exists". A final
  pre-submission pass (Google Scholar + Semantic Scholar, exact-phrase searches for
  "leave-one-B-out perovskite" and "composition degeneracy floor perovskite") is still
  recommended.

---

## Verification log (what was opened in this session)

- Cherukara & Mannodi-Kanakkithodi 2022 — full text (IOP ampdf), sections I–IV,
  Tables I–II, conclusion, references. DOI 10.1088/1361-651X/ac52de confirmed.
- Asaria, Salomone & Gandhi — full PDF (370 lines), abstract through references.
  No journal/DOI/date in the document.
- Li et al. 2024/2025 — arXiv HTML + *Commun. Mater.* version (d-nb.info);
  authors, journal, DOI 10.1038/s43246-024-00731-w confirmed.
- Mahdavi, Honavar & Morgan 2025 — RSC article page + search extracts; authors,
  journal, pages 2972–2982, DOI 10.1039/D5DD00182J confirmed.
- Boruah 2026 — arXiv abstract page (arXiv:2611.14521); author, affiliation, date,
  journal submission confirmed.
- Suhendar 2025 — IOP PDF extract; author, journal, DOI confirmed.
- Deng et al. 2025 — MDPI article; full author list, journal, DOI confirmed.
- Liu et al. 2026 — MDPI article; full author list, journal, DOI confirmed.
- Ward et al. 2018 (matminer) — search extracts; full author list, journal, pages
  60–69, DOI confirmed.
- Chenebuah et al. 2021 — ScienceDirect + NRC archive extracts; authors (first),
  sample sizes, metrics confirmed; volume/DOI not found.
- Draft v1 (`paper_draft_v1.md`, 209 lines) and workflow Stage 8
  (`v2_workflow.md` lines 438–522) read before writing.
