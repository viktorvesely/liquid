# Supplementary analysis

## Capacity curves across all factors

![Full capacity curves](figures/02_capacity_scaling.png)

Each row varies predictor width, predictor count, or delegator width; the other factors are averaged equally. All curves use the logged loss and L=1 baseline. L=0 is included.

## Quantitative comparison of the capacity factors

For each task and each nontrivial positive scaling level L∈{2,4,8,16,32}, let g(M,b_M,b_L,L) be the seed-mean direct-loss reduction against L=1. We subtract its capacity-grid mean at that L and decompose the resulting 5×3×3 balanced grid into orthogonal main effects, pairwise interactions, and the three-way interaction. We sum each term's squared contribution over all five scaling levels and divide by the total squared capacity variation. This is an exact **descriptive partition of the observed gain variation**, including finite-seed noise—not a causal importance measure or a significance test. L=1 has identically zero gain; L=0 is shown in the curves but excluded from this ranking because it compares fixed routing with learned routing rather than scaling an existing delegator ensemble.

![Factor importance across the scaling trajectory](figures/06_factor_importance.png)

*Figure S4. Contribution to observed capacity variation in scaling gains, integrated across all five L>1 levels. The interaction bar combines all pairwise and three-way terms; bars sum to 100% within a task. Direct CE/MSE is used. [Vector PDF](figures/06_factor_importance.pdf).*

| Task | Predictor width alone | Predictor count alone | Delegator width alone | Capacity interactions |
|---|---:|---:|---:|---:|
| CIFAR-10 | 0.7% | 26.8% | 27.4% | 45.1% |
| SVHN | **38.0%** | 8.5% | 1.4% | 52.1% |
| Bikes | 11.0% | 20.1% | 15.1% | 53.9% |
| Energy | 13.1% | 12.2% | **33.6%** | 41.1% |

**Predictor width is the dominant single moderator on SVHN, not a universal moderator.** On CIFAR-10, count and delegator width matter jointly; their pairwise interaction alone accounts for 21.3% of capacity variation. Bikes cannot be summarized well by a single capacity factor: interactions account for more than half its variation. Energy is most strongly moderated by delegator width. Repeating the calculation with the logged objective preserves these task-specific rankings. For example, SVHN's predictor-width main effect is 39.1% under logged loss and 38.0% under direct CE.

The substantial interaction shares also explain why a global “pwidth is most impactful” or “dwidth is most impactful” conclusion would be misleading. Predictor width can matter through an interaction even when its averaged main effect is small. The full partition is retained in `factorial_importance.csv`, and each L-specific partition in `factorial_variation_by_L.csv`.


## Detailed task-error attribution

| Task / transition | Predictor/reference contribution | Delegator contribution | Total loss increase | Attribution of the deterioration or improvement |
|---|---:|---:|---:|---|
| CIFAR-10, 1→4 | +0.03% | +1.29% | +1.32% | Early deterioration is almost entirely delegation |
| CIFAR-10, 1→32 | −1.05% | −3.28% | −4.32% | Both improve; delegation contributes most of the gain |
| SVHN, 1→2 | −2.73% | −0.01% | −2.74% | Early improvement is predominantly predictor/reference |
| SVHN, 8→32 | +2.57% | −1.04% | +1.53% | Late deterioration is predictor/reference-side; delegation offsets it |
| Bikes, 1→32 | −11.17% | +24.81% | +13.64% | Delegation worsens more than predictor/reference performance improves |
| Energy, 1→32 | −6.61% | +10.93% | +4.32% | Same delegation-side failure as Bikes |

All percentages in this table use L=1 as denominator, including the 8→32 transition. Thus the SVHN late-stage figure is 1.53% of L=1 loss, not a relative change using L=8 as a new baseline. Exact unrounded components and seed-bootstrap intervals are in `failure_contrasts.csv` and `fault_uncertainty.csv`.

**Bikes and Energy: delegators are responsible for the scaling-induced deterioration relative to the fitted reference.** The predictors become better as an ensemble under reference routing, so worsening predictor/reference capability cannot explain the total loss increase. The learned delegators fail to realize those gains, and their excess loss is larger than the gain available from the predictor ensemble. This conclusion also holds if the reference is replaced by the better of learned and oracle routing: the corresponding delegation penalties are +23.84% and +7.52%, against predictor/reference improvements of 10.19% and 3.21%.

**SVHN: the source of the limitation changes with L.** At the first doubling, almost all the improvement is predictor/reference-side. At large L, delegation improves but cannot offset worsening predictor/reference performance. Calling the L=8→32 downturn a delegator failure would give the wrong attribution. This distinction also survives the better-of-learned-and-oracle reference.

**CIFAR-10: the initial failure and eventual recovery are primarily delegation-side.** At four delegators, predictor/reference loss barely changes but delegation is worse. At 32, both improve and the delegation contribution is larger. This matches the need for sufficient delegator width in the capacity curves.

### How does this explain the capacity dependence?

The same task-loss decomposition localizes the width effects, rather than simply listing which slice wins:

| Task / width varied, L=1→32 | Width | Predictor/reference change | Delegation change | Total change |
|---|---:|---:|---:|---:|
| SVHN / predictor width | 4 | +0.73% | −9.26% | −8.53% |
| SVHN / predictor width | 16 | −1.46% | −0.33% | −1.80% |
| Bikes / predictor width | 4 | −11.78% | +30.51% | +18.73% |
| Bikes / predictor width | 16 | −10.50% | +19.87% | +9.36% |
| CIFAR-10 / delegator width | 4 | −0.09% | −0.42% | −0.51% |
| CIFAR-10 / delegator width | 16 | −1.72% | −5.86% | −7.58% |
| Energy / delegator width | 4 | −4.42% | +5.93% | +1.51% |
| Energy / delegator width | 16 | −8.74% | +16.59% | +7.85% |

**Predictor width's contrasting effects are largely expressed through delegation.** On SVHN, the large narrow-predictor benefit is associated with a much larger reduction in the delegation gap. On Bikes, wider predictors help primarily by reducing the delegation penalty. Greater delegator width has the opposite effect between CIFAR-10 and Energy: it reduces the gap in the former and amplifies it in the latter. The data therefore support a task-specific interaction between predictor capacity and the generalization of learned routing, rather than a universal preferred predictor or delegator width. All three factors and every positive L are retained in `conditional_fault_attribution.csv`; the table highlights the contrasts that explain the observed curve separation.

### Additional error is not the same as all remaining error

The attribution above concerns **why increasing L changes error**. It does not mean the delegators account for all the residual error. At L=32, mean Bikes MSE is 0.06103 = 0.04668 under the fitted reference + 0.01435 delegation gap; Energy MSE is 0.66949 = 0.62007 + 0.04942. Most remaining loss lies in the predictor/reference term even though the *increase caused by scaling* lies in delegation. For SVHN, the fitted reference is worse than learned routing on average at L=32, giving a negative signed gap; that denotes superiority to this reference, not negative physical error.

The component attribution is conditional on the fitted oracle and jointly trained predictors. It identifies where the measured deterioration appears, not a controlled causal intervention on one component. This qualification does not prevent the concrete attribution above.

## 5. Does the pattern persist beyond the final epoch?

![Native performance and endpoint sensitivity including uniform routing](figures/05_endpoint_sensitivity.png)

*Figure S1. Top: accuracy or R² across every L including uniform routing. Bottom: logged-loss reductions under final epoch, final-10%-of-training average, and per-seed minimum-validation endpoints, all normalized to the corresponding L=1 endpoint. The minimum is an optimistic sensitivity analysis, not test performance. [Vector PDF](figures/05_endpoint_sensitivity.pdf).*

The last-10% average preserves the L=32 direction on all tasks: +3.44% CIFAR-10, +4.68% SVHN, −13.77% Bikes, −4.08% Energy in logged-loss reduction. Even optimistic minimum-validation selection preserves the directions, although it reduces the regression penalties. The failure is not a single noisy final checkpoint.

![Training and validation including uniform routing](figures/07_learning_curves.png)

*Figure S2. Training (dashed) and validation (solid) logged loss for L∈{0,1,4,32}, averaged over all capacity configurations and seeds. The first 10% of epochs are omitted; a trailing 1%-of-training-window mean is used only for display. [Vector PDF](figures/07_learning_curves.pdf).*

The regression training/validation separation supports the delegation-generalization interpretation: between L=1 and 32, Bikes training MSE falls from 0.01635 to 0.00327 while validation MSE rises from 0.05375 to 0.06103; Energy training MSE falls from 0.55057 to 0.46494 while validation MSE rises from 0.64220 to 0.66949. Combined with improving predictor/reference loss, this points to increasingly poor generalization of the learned routing relative to the fitted reference, rather than failure to fit the training objective.

![Complete capacity grid at L=32](figures/08_full_grid.png)

*Figure S3. The complete 45-configuration direct-loss comparison of L=32 against L=1, retained as a supplement to the all-L, all-factor analysis. Each cell averages five seeds first; positive values favor L=32. Rows encode predictor count and width, columns delegator width. This endpoint map is not the basis for choosing the dominant moderator. [Vector PDF](figures/08_full_grid.pdf).*

## Appendix A. Definitions and measurement details

**Data and averaging.** Only the four September 7 launches named in the notebook are used: CIFAR-10 `742fde19a2`, SVHN `1b88e205dd`, Bikes `4f767e649d`, and Energy `2f7815bfc8`. All 1,260 conditions and 6,300 seed records are present. The grid contains 45 capacity configurations per task and seven L values. At L=0, the three nominal delegator-width runs have identical seedwise losses, so these are repeated baselines rather than independent replications. They are matched to each learned configuration when calculating reductions. Five-seed means are computed before ratios and capacity averages. At no point are individual seeds or model members counted as independent capacity configurations.

**Uncertainty.** Pointwise 95% bootstrap intervals enumerate all 5⁵=3,125 ordered resamples of the five seed indices. The same indices are resampled jointly across configurations and L, preserving shared-key dependence; means and ratios are recomputed. The intervals are conditional on the fixed grid and validation split and have only five underlying seed blocks. They are not simultaneous significance intervals. The factorial partition is a descriptive analysis of the observed grid and retains sampling noise; it does not justify population-wide causal rankings.

**Endpoints and L=0 availability.** JSON trajectories exist for every L. Direct evaluation and per-model decomposition NPZ files are empty at L=0. Consequently, all-L loss curves use JSON consistently, while direct task attribution and member decompositions start at L=1. Regression logged loss agrees with direct MSE to small numerical error; classification logged loss differs because its KL ambiguity clips probabilities at 10⁻⁶ while individual CE is unclipped. Mean logged-minus-direct discrepancies are 0.0381 nats on CIFAR-10 and 0.0184 on SVHN. The primary positive-L conclusions and factor rankings agree across these endpoints.

**Two different decompositions.** The saved task-loss decomposition E=O+H uses the fitted oracle directly and is numerically exact. The saved predictor P/A arrays instead use the better of learned and oracle weights. Therefore P−A corresponds to that selected reference, with a clipping residual in classification; it is not always O. The report uses direct O/H to attribute task loss and P/A to characterize predictor individual-error/ambiguity compensation. The previous selected-reference attribution is retained as a sensitivity comparison in `failure_contrasts.csv`.

For arithmetic-mean delegators, the untruncated theoretical identity is CE(q,mean p_l)=mean CE(q,p_l)−mean Σq log[(mean p_l)/p_l]. The recorded C is unclipped CE but D clips probabilities; C−D is therefore not exact aggregate router CE. This is why responsibility for wrong task predictions is determined from **E=O+H**, not inferred from the routing-proxy residual. Raw delegator probabilities and oracle target distributions needed to remove that discrepancy are not saved. The fact that the same C/D pattern accompanies opposite task-level outcomes is itself evidence against using the proxy alone for task attribution.

**Reference and protocol.** The fitted oracle uses frozen predictors, fresh restarts, and validation-selected training; its capacity varies with L. It can be worse than learned routing, hence signed gaps. Validation is the first 2,560 samples from the loaded training split, shared across model seeds; these are not independent test results. The supplied draft's per-seed 85/15 split description and image batch/epoch table need reconciliation with the source and recorded trajectories (100 image epochs, batch size 128; 2,000 regression epochs, batch size 256). The current oracle trains on task loss with frozen predictors, not supervised samplewise optimal weights. The implemented routing is dense and soft, and increasing L increases parameters and compute; this experiment is not a fixed-budget or sparse-inference comparison.


## Reproduction

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python liquid_jax/scaling_analysis/analyze.py --rebuild
.venv/bin/python liquid_jax/scaling_analysis/render_report.py
```

Without `--rebuild`, the existing seed/trajectory CSV cache is used. With it, all 2,520 selected JSON/NPZ files are reread and hashed. Five-seed means, complete factorial coverage, component-sum identities and the orthogonal variance partition are checked. No experiments are rerun. `capacity_scaling.csv`, `factorial_importance.csv`, `both_ambiguity_decompositions.csv`, `fault_attribution.csv`, `failure_contrasts.csv`, `fault_uncertainty.csv`, and `conditional_fault_attribution.csv` contain the corresponding numerical results. `input_manifest.json` and `provenance.json` record source hashes and runtime versions.

## Exact normalization of the member-decomposition curves

For a fixed task and capacity configuration, P_L and A_L are the five-seed means of weighted predictor individual loss and ambiguity; C_L and D_L are the corresponding delegator individual routing loss and ambiguity. With ΔX=X_L−X₁, the predictor curves are 100ΔP/P₁, 100ΔA/P₁, and 100Δ(P−A)/P₁. The delegator curves are 100ΔC/C₁, 100ΔD/C₁, and 100Δ(C−D)/C₁. All three curves within a family share one denominator. Their percentages are averaged over the 45 capacity configurations. A negative residual change does not imply a negative raw residual. `raw_decomposition_means.csv` contains the unnormalized condition means for direct inspection.
