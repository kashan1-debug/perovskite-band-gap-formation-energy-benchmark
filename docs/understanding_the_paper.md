# Understand the paper in 1–2 hours — the six numbers

Written 2026-11-25 for Muhammad's own-words gate before submission.
No code in this file. The rule agreed with Muse: you do NOT need to write code,
know the maths inside XGBoost, memorise syntax, or fix errors. "Just enough" for
each result is four things: (1) what goes in, (2) what the test does in one plain
sentence, (3) what comes out and what it means chemically, (4) why this test.

Your chemistry is already the hard part. A perovskite is BX6 octahedra sharing
corners, with A sitting in the holes — that is coordination chemistry you already
know. The tolerance factor and octahedral factor are ionic-radius rules from your
own field. A MOF is a metal node + linker framework; a perovskite is the same idea,
simpler. The ML is only a tool on top of that.

How to use this file: read one number (about 10 minutes), then close the file and
say it back in your own words, as if explaining it to a classmate. If you can do
all six, you are ready — that is the gate, not the code.

## The one question the whole paper asks

Can a model predict a perovskite that contains a B-site metal it has never seen?
A random test only checks familiar crystals. Discovery is about new ones, so the
paper hides one whole metal at a time (52 tests, 364 crystals each) and asks the
model to predict it. Everything below is an answer to that question.

## 1. The ceiling — 0.543

- **What goes in:** Magpie numbers for 18,928 crystals, and their formation energies.
  Magpie only sees the ingredients (the composition), not which seat each atom sits in.
- **What the test does:** two crystals can have the same ingredients but the A and B
  atoms swapped between seats. The model sees them as identical; reality gives them
  different energies.
- **What comes out:** even a perfect model that only sees ingredients can score at
  most R² 0.543 on this data. R² is just "how much of the ups and downs the model
  explains" — 1.0 is perfect, 0 means guessing the average.
- **Why this test:** so nobody reads a low score as "our model is bad". The limit is
  built into the data itself, not into our model. Another group (Asaria) found the
  same limit independently, in different units — that makes it stronger, not weaker.
- **Say it back:** why can no ingredient-only model ever score above 0.543?

## 2. The failure — Magpie 0.086 on a new metal

- **What goes in:** the same Magpie numbers, nothing else.
- **What the test does:** hide every crystal with one B metal (say, magnesium),
  train on the rest, then ask for magnesium.
- **What comes out:** pooled R² 0.086 over all 52 hidden metals. That is barely
  better than guessing the average. On familiar crystals the same model scored
  about 0.43 — so it interpolates, but it does not transfer.
- **Why this test:** this is the discovery question, asked honestly. A random split
  would have hidden the failure.
- **Say it back:** what is the difference between "works on familiar crystals" and
  "works on a new metal" — and which one matters for finding new materials?

## 3. The fix — site-aware 0.733

- **What goes in:** Magpie, plus each site's own physics — A, B and X each get their
  own ionic radius, electronegativity, and so on — plus the two textbook rules you
  already know, tolerance factor t and octahedral factor mu.
- **What the test does:** the same hide-one-metal test as in (2), nothing else changed.
- **What comes out:** pooled R² 0.733, up from 0.086. Per metal: 46 of 52 families
  positive, best Ni 0.747, worst Cr −1.517 (chromium still fails — say so honestly).
- **Why this test:** it shows the fix is the *representation*, not a fancier model.
  Give the model the seats, not just the ingredients, and the same simple model
  transfers. We checked four ways that this is not a leak (the alarm fired at 0.60
  and we investigated before reporting it).
- **Say it back:** what changed between 0.086 and 0.733 — the model, or what the
  model was allowed to see?

## 4. The surprise — no-B-info 0.683

- **What goes in:** the site-aware features with ALL B information deleted — no B
  one-hot, no B physics, no t/mu (they need the B radius). Only A and X are given
  explicitly; Magpie still carries B in aggregate form.
- **What the test does:** the same hide-one-metal test again.
- **What comes out:** pooled R² 0.683. B physics only adds +0.050 on top of that.
  But it is metal-dependent: without B, chromium jumps from −1.517 to +0.446,
  while tungsten falls from 0.226 to −0.098 and rhenium from 0.300 to −0.497.
- **Why this test:** it locates the gain. Most of the fix comes from naming A and X
  explicitly (site resolution), not from B physics. B physics helps the hard heavy
  metals and hurts chromium.
- **Say it back:** if B numbers barely matter, what is actually doing the work —
  and why does tungsten still need them?

## 5. The graph model — MEGNet 0.561

- **What goes in:** the crystal structures themselves, fed to a pretrained graph
  network (MEGNet) trained on 133,420 Materials Project structures. It was never
  trained on our data.
- **What the test does:** ask it to predict our crystals, metal by metal, same
  hide-one-metal test. Its raw scores look catastrophic (R² −77.8) only because its
  energies use a different oxygen reference — a scale difference, not a failure.
  After one straight-line rescaling, the honest number is R² 0.561, ranking
  (Spearman) 0.752.
- **What comes out:** 0.561 beats Magpie-only (0.086) by a lot, but it does NOT beat
  our site-aware model (0.733). Structure alone does not close the gap that site
  resolution closes. The number is also an upper bound: 6.2% of our crystals share
  a formula with Materials Project, so MEGNet may have seen some of them.
- **Why this test:** reviewers always ask "did you compare with a graph model?"
  Now the answer is in the paper, on the same hard test, with the scale warning
  stated first so nobody panics at −77.8.
- **Say it back:** why is the raw −77.8 not the real result, and what does 0.561
  vs 0.733 actually prove?

## 6. The uncertainty — tungsten 0.690

- **What goes in:** the tuned XGBoost model plus conformal intervals — instead of
  one number, the model gives a range ("0.30, plus or minus 0.19") chosen so the
  true value should fall inside it 90 times out of 100.
- **What the test does:** check that 90% promise twice — on random crystals, and
  on a metal the model never saw.
- **What comes out:** on random crystals coverage is 0.899 — the promise holds.
  Under shift it is metal-dependent: Te 0.923 and Mg 0.945 hold, but tungsten
  falls to 0.690. The intervals are honest and wide; the point predictions for
  tungsten are still bad (point R² −4.211 in the same test). Coverage is not
  accuracy — say both numbers together.
- **Why this test:** a screening tool that says "I'm 90% sure" must be checked on
  the new metals it will actually be used for, not only on familiar ones.
- **Say it back:** the Te intervals cover 92% of the time — does that mean the
  model predicts Te well? Why not?

## The rest of the paper in three lines

- Band gap: 96.1% of gaps are zero, so classify metal vs non-metal first (it ranks
  well but finds only a small share of non-metals at the usual threshold), then
  regress only the 735 real gaps.
- SHAP (which features the model leans on) vs ablation (what happens when you
  remove them) are different questions. SHAP says Magpie still carries 42% of the
  credit on unseen metals; the ablation says Magpie alone scores 0.086. Both are
  true: the same Magpie signal becomes usable only once A and X are resolved.
- Nothing here claims a new solar material or a stable material: no
  energy-above-hull data, DFT gaps only, one synthetic grid, three seeds.

## Your 1–2 hour plan (in this order)

1. The question (5 min) — section above.
2. The ceiling 0.543 (10 min) — number 1.
3. The failure 0.086 and the fix 0.733 (15 min) — numbers 2–3.
4. The no-B result 0.683 (10 min) — number 4.
5. MEGNet 0.561 and tungsten 0.690 (15 min) — numbers 5–6.
6. Read the plain-English summary at the top of `paper_draft_v2.md` and explain
   each number back in your own words (rest of the time).

## Where to learn the ML words (your order: Python, then libraries, then ML)

- **ML basics, plain English:** StatQuest with Josh Starmer (YouTube) — *Random
  Forest*, *XGBoost / gradient boosting*, *train/test split*, *R-squared*,
  *overfitting*. 10–15 minutes each.
- **ML for materials:** Taylor Sparks, "Materials Informatics" playlist
  (https://www.youtube.com/playlist?list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) —
  start with lectures 1, 7, 8, 9, 11, 17 and 20. Lecture 20 is literally
  "Can we extrapolate to extraordinary materials?" — this paper's question.
- **Just enough Python:** Data School (Kevin Markham, YouTube) — pandas,
  train/test split, scikit-learn basics.
- **Python foundation course:** the Google Data Analytics certificate — it was
  upgraded from R to Python (Course 7 is now "Introduction to Data Analysis
  Using Python": Python, pandas, NumPy, matplotlib), so it now fits the
  Python + pandas foundation the Colab cells use. Taylor Sparks + this project
  supply the materials-ML part no generic course teaches.
- **Later, for depth:** Andrew Ng's Machine Learning Specialization (Coursera),
  Coursera's Materials Data Sciences and Informatics, 3Blue1Brown's neural
  network series.

## Still open before submission (not for this study session)

- Read draft v2 (`workspace/perovskite/paper_draft_v2.md`) and mark what to change.
- Upload Figure 7 (`shap_familyout.png`) from Drive `results_v2/figures/` the same
  way you uploaded Figures 1–6, so the PDF can be built with all seven figures.
  The draft embeds it at `figures/shap_familyout.png`, which is not yet in
  `workspace/perovskite/figures/`.
- Author names / affiliations, preprint-or-journal target, repo link, funding.
- Similarity check, and the final pre-submission literature pass on Google Scholar
  / Semantic Scholar (recommended in the draft, not yet done).
