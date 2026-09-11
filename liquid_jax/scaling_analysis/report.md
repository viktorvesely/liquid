# When does delegator scaling improve an ensemble?

Increasing the number of delegators generates more diversity, but its benefit depends on how predictor capacity and routing generalization interact. The regression tasks fail because the learned delegation loses more than the predictor ensemble gains. SVHN eventually encounters the opposite limitation: delegation improves, but predictor/reference performance deteriorates. CIFAR-10 benefits primarily from improved delegation at sufficiently large routing capacity.

All comparisons first average the five seeds within each fixed configuration of predictor count and both widths. Each plotted percentage is computed against that same configuration at **L=1**, then averaged over configurations. The denominator is specified in each figure caption; it is not always the same kind of loss. Performance curves include **L=0, fixed uniform routing**.

## Scaling has three distinct regimes

![Scaling performance](figures/01_scaling.png)

*Figure 1. For each configuration, the plotted reduction is 100 × (E₁ − E_L)/E₁, where E is its seed-mean recorded validation loss. These percentages are then averaged over configurations. Thus +5% means a mean 5% reduction relative to each matched one-delegator loss. Positive values indicate improvement. Shading shows pointwise 95% seed-block bootstrap intervals. L=0 denotes uniform routing. [PDF](figures/01_scaling.pdf).*

CIFAR-10 exhibits a **delayed benefit**: two or four delegators initially hurt, and larger ensembles recover. SVHN has an **intermediate optimum**, around 8–16 delegators. Bikes and Energy show **early saturation followed by deterioration**: their best observed averages occur at two and four delegators, respectively, with Bikes' small gain indistinguishable from zero under the seed interval.

Uniform routing matters to this interpretation. On CIFAR-10, scaling learned routing does not establish superiority to the static ensemble: mean accuracy at L=32 remains 0.31 percentage points below L=0. SVHN exceeds uniform accuracy by 3.60 points. Bikes eventually falls below uniform routing, whereas Energy remains slightly above it. Improvements over one learned delegator are therefore not automatically improvements over a static ensemble.

## Capacity changes the scaling regime

We compared **all three capacity factors across the entire scaling curve**, rather than selecting an L=32 slice. For each L>1, an orthogonal factorial decomposition separates variation in matched scaling gains into predictor-width, predictor-count, delegator-width, and interaction terms. The table summarizes their contributions across L∈{2,4,8,16,32}.

| Task | Predictor width | Predictor count | Delegator width | Interactions |
|---|---:|---:|---:|---:|
| CIFAR-10 | 0.7% | 26.8% | 27.4% | 45.1% |
| SVHN | **38.0%** | 8.5% | 1.4% | 52.1% |
| Bikes | 11.0% | 20.1% | 15.1% | 53.9% |
| Energy | 13.1% | 12.2% | **33.6%** | 41.1% |

*Shares of observed capacity variation in direct-loss scaling gains; the four columns sum to 100% up to rounding. These describe this factorial grid, including finite-seed noise, rather than causal importance. The complete curves and partition are in the supplement.*

**Predictor width is the strongest individual moderator on SVHN, but its direction does not generalize across tasks.** Narrow SVHN predictors gain 11.96% at L=8 and 12.19% at L=16, versus only 1.09% and 2.38% for wide predictors. On Bikes, narrow predictors instead suffer more: the L=32 penalty is 18.73% at width four and 9.36% at width 16. On Energy, wider predictors reach their useful scaling limit earlier. CIFAR-10's predictor-width curves largely converge at large L; predictor count and delegator width explain more of its scaling response.

The broader pattern is therefore **a balance between the prediction and routing components, not a universally favorable small or large width**. CIFAR-10 needs sufficiently capable individual delegators to benefit from a large delegator ensemble. SVHN's narrow predictors benefit strongly from improved routing. On the regression tasks, adding delegators ultimately increases the routing penalty, with predictor capacity modulating how severe that penalty becomes. The large interaction shares explain why averaging everything into a single “most important width” obscures the result.

## Both components diversify while individual performance worsens

![Predictor and delegator performance–ambiguity decompositions](figures/03_loss_accounting.png)

*Figure 2. **Changes from L=1, not absolute loss or ambiguity.** Let P and A be the seed-mean weighted individual predictor loss and ambiguity. The top-row curves are 100ΔP/P₁, 100ΔA/P₁, and 100Δ(P−A)/P₁, where ΔX=X_L−X₁. All three share the same denominator P₁. For delegators (bottom), replace P,A with individual routing loss C and ambiguity D, and use C₁ as the common denominator. Percentages are computed per configuration, then averaged. [PDF](figures/03_loss_accounting.pdf).*

A negative green value means **loss minus ambiguity has decreased from its L=1 value**, not that loss minus ambiguity is negative. For example, Bikes' mean raw P−A decreases from 0.05222 at L=1 to 0.04668 at L=32; both are positive. Every saved seed/configuration has positive raw P−A (minimum 0.04204). The plotted percentage uses each configuration's P₁ as denominator—not its (P₁−A₁)—so it is a contribution measured relative to baseline individual loss, not a relative percentage change in the residual itself. The shared denominator preserves the subtraction identity between the three curves.

Predictor ambiguity grows together with individual predictor loss. Their near cancellation is particularly pronounced in regression: on Bikes, mean reference-weighted individual MSE rises from 16.65 to 97.00 between L=1 and L=32, while ambiguity rises from 16.60 to 96.95. The ensemble increasingly relies on compensation between individually poor predictions.

Delegators also become more diverse, while their individual losses against the fitted routing reference increase. In the stored decomposition, the individual-loss increase exceeds the ambiguity increase. This happens on successful classification tasks as well as unsuccessful regression tasks, so it does not by itself identify the source of task-level failure.

The shared conclusion is **compensation rather than improved individual correctness**. Greater diversity is useful only when it offsets individual error in the resulting task prediction—the trade-off emphasized by [Wood et al. (2023)](https://jmlr.org/papers/v24/23-0041.html). Whether learned delegation actually realizes that compensation is answered by the task-loss decomposition below.

## Which component limits scaling?

The saved task-loss decomposition is **E = O + H**: O is the predictor ensemble's task loss under the fitted reference delegators, and H is the learned delegation's excess task loss relative to that reference. It separates predictor/reference capability from the additional error associated with learned routing.

![Attribution of scaling gains and losses](figures/04_fault_attribution.png)

*Figure 3. For each configuration, the predictor/reference contribution is 100(O₁−O_L)/E₁ and the delegation contribution is 100(H₁−H_L)/E₁, where E₁ is the seed-mean direct task loss at L=1. Their sum is 100(E₁−E_L)/E₁. The plotted values average these percentages across configurations. Both components use total baseline task loss E₁, not their own baseline values. Positive contributions improve performance; negative contributions worsen it. [PDF](figures/04_fault_attribution.pdf).*

| Transition | Predictor/reference contribution | Delegation contribution | Net reduction | Interpretation |
|---|---:|---:|---:|---|
| CIFAR-10, 1→4 | −0.03% | −1.29% | −1.32% | Early failure is delegation-side |
| CIFAR-10, 1→32 | +1.05% | +3.28% | +4.32% | Delegation provides most of the gain |
| SVHN, 8→32 | −2.57% | +1.04% | −1.53% | Late failure is predictor/reference-side |
| Bikes, 1→32 | +11.17% | −24.81% | −13.64% | Delegation loses more than predictors gain |
| Energy, 1→32 | +6.61% | −10.93% | −4.32% | Same delegation-side failure |

*All contributions are percentages of the matched L=1 loss, including the 8→32 transition. Unrounded values and seed intervals are supplied with the analysis.*

**On Bikes and Energy, the additional error comes from delegation relative to the fitted reference.** Predictor/reference loss improves as L grows, but learned routing fails to realize that improvement. The training/validation trajectories support a generalization failure: training MSE falls while validation MSE rises. More recorded ambiguity is therefore not solving the problem; the learned routing increasingly fails to exploit the compensating predictor ensemble.

**On SVHN, the limiting component changes.** The first doubling primarily improves predictor/reference performance. After the intermediate optimum, predictor/reference loss worsens while delegation continues to improve. Blaming the late downturn on delegators would reverse the measured attribution.

**On CIFAR-10, early deterioration and later recovery are primarily delegation-side.** This also explains the width interaction. Increasing delegator width from four to 16 changes the L=1→32 delegation contribution from a 0.42% improvement to a 5.86% improvement. Energy shows the opposite pattern: the corresponding delegation penalty grows from 5.93% to 16.59%.

Predictor width's contrasting effects are likewise expressed through routing. At L=32, narrow SVHN predictors obtain a 9.26% delegation improvement, versus 0.33% for wide predictors. On Bikes, widening predictors reduces the delegation penalty from 30.51% to 19.87%. Thus the capacity curves and decomposition support the same narrative: the benefit depends on whether the learned delegation can generalize well enough to exploit the available predictor ensemble.

These statements concern **the change in error caused by scaling**, not all remaining error. Most residual regression loss still lies in the predictor/reference term. Attribution is conditional on the fitted reference, whose capacity also changes with L; it is not an intervention that holds one component fixed. The identified regression failure and SVHN late-stage limitation persist when the reference is replaced by the better of learned and fitted-oracle routing.

## Data and scope

The analysis uses only the four notebook-selected scaling launches: 1,260 conditions and five seeds each. Intervals jointly resample the five seed indices across the fixed grid. Figure 1 uses the recorded objective available for every L; attribution uses direct evaluation CE/MSE. Positive-L conclusions agree across the two endpoints. The member decompositions use recorded ambiguity terms, including their probability-clipping residuals; task-error attribution uses the separate exact E=O+H identity.

The results concern the recorded validation split, not an independent test set, and compare increasing capacity rather than a fixed parameter budget. Full capacity curves, sensitivity analyses, numerical definitions, and reproduction instructions are provided below. The manuscript-ready text is in `paper_results.tex`.
