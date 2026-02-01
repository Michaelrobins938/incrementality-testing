# Incrementality & Geo-Testing: Recovering True Causal Lift
## Integrating Synthetic Control, Difference-in-Differences, and Placebo-Validated Inference

**Technical Whitepaper v1.0.0**

| **Attribute** | **Value** |
|---|---|
| **Version** | 1.0.0 |
| **Status** | Production-Ready |
| **Date** | January 31, 2026 |
| **Classification** | Causal Inference / Econometrics |
| **Document Type** | Technical Whitepaper |

---

## **Abstract**

Marketing attribution often confuses correlation with causation. This whitepaper details a **Geographical Incrementality Testing Framework** designed to measure the true "causal lift" of marketing interventions. By leveraging **Synthetic Control (SCM)** and **Difference-in-Differences (DiD)**, we construct counterfactual "what-if" scenarios that allow marketers to see what would have happened if they HAD NOT spent money in a specific region. 

---

## **Glossary & Notation**

| **Term** | **Definition** |
|---|---|
| **Counterfactual** | The scenario that would have occurred without the intervention. |
| **Synthetic Control** | A weighted combination of control markets that best mimics the treated market's pre-period history. |
| **DiD** | Difference-in-Differences; comparing the change over time in a treated group vs. a control group. |
| **RMPE** | Root Mean Squared Prediction Error; a measure of fit in the pre-treatment period. |
| **Placebo Test** | Re-running the analysis on non-treated units or time periods to verify that estimated effects are significant. |
| **iROAS** | Incremental Return on Ad Spend; revenue lift divided by experimental spend. |

---

## **1. The Flaw in Observation: Why We Need Experiments**

Observational attribution (like last-touch) rewards channels that reach users who were already going to buy. To find **Incremental Revenue**, we must run randomized experiments. However, since user-level randomization is often impossible (due to "walled gardens" or privacy), we use **Geographical Randomization** (Geo-Testing).

---

## **2. Methodology: Constructing the Counterfactual**

### **2.1 Geo-Matching**
We identify a pool of potential control markets (DMAs) and use a **Mahalanobis Distance** or **Optimal Hungarian Assignment** to find the pair that most closely tracks the treatment market during the pre-period ($R^2 > 0.85$).

### **2.2 Difference-in-Differences (DiD)**
The simplest causal estimator. We measure the "gap of gaps":
$$\hat{\tau} = (\bar{Y}_{T,post} - \bar{Y}_{T,pre}) - (\bar{Y}_{C,post} - \bar{Y}_{C,pre})$$
This removes time-invariant biases between markets and global time trends.

### **2.3 Synthetic Control Method (SCM)**
For high-stakes decisions, we use SCM. Rather than picking one control market, we create a **Weighted Synthetic Market**:
$$\hat{Y}_{T,counterfactual} = \sum_{j \in DonorPool} w_j \cdot Y_j$$
Weights are optimized using constrained regression to minimize pre-period error.

---

## **3. Statistical Inference & Validity**

Standard p-values often fail in time-series contexts. We use **Placebo-Based Inference**:
1. **In-Space Placebo:** We "pretend" a control market was treated and measure its "effect." If the real treatment effect is larger than 95% of placebo effects, it's significant.
2. **In-Time Placebo:** We "pretend" the treatment started earlier. The effect should be zero.
3. **Block Bootstrap:** We resample time blocks to estimate the variance of the lift, preserving temporal correlation.

---

## **4. Power Analysis: Preventing "Unclear" Results**

Before launching a test, we run a **Power Simulation** to determine the probability of detection. We vary the potential lift (e.g., 5%, 10%, 15%) and the test duration to ensure that stakeholders aren't wasting money on an "underpowered" experiment where the result is likely to be "Inconclusive."

---

## **5. Business Integration: The iROAS Metric**

The output of every incrementality test is the **Incremental ROAS (iROAS)**:
$$iROAS = \frac{\text{Incremental Sales}}{\text{Incremental Spend}}$$
This is the only metric that should be used for budget scaling. If $iROAS > \text{Target}$, the channel is truly driving growth; if not, the channel is merely "harvesting" existing demand.

---

## **6. Technical Implementation Specification**

- **Inference Engine:** Python (SCM, DiD, BSTS wrappers).
- **Validation Suite:** Automated Type I/II error tracking and placebo generation.
- **Reporting:** 95% Confidence Intervals provided for every lift estimate.

---

## **7. Causal Interpretation & Limitations**

- **Spillover Effects:** Marketing in one geo might leak into neighboring geos (TV bleed-over).
- **Parallel Trends Assumption:** DiD assumes the treated and control geos would have followed the same trend—often hard to prove.
- **Duration Bias:** Short-term tests may miss "delayed" effects that occur weeks later.

---

## **8. Conclusion**

Incrementality testing is the "Gold Standard" of attribution. By moving away from observational counting and towards experimental causality, businesses can eliminate waste and double down on the channels that actually move the needle.
