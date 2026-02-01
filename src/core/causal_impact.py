"""
Causal Impact Analyzer
======================

Implements difference-in-differences and Bayesian structural time series
for measuring causal impact of geo-based interventions.

Key Features:
- DiD with parallel trends testing
- Bayesian structural time series (BSTS)
- Posterior predictive counterfactuals
- 95% credible intervals on lift estimates
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Literal
from scipy import stats
from scipy.optimize import minimize
import warnings


@dataclass
class CausalImpactResult:
    """Container for causal impact analysis results."""
    # Core estimates
    average_effect: float
    cumulative_effect: float
    relative_effect: float  # Percentage lift
    
    # Uncertainty
    effect_lower: float
    effect_upper: float
    credible_interval: float
    
    # Statistical tests
    p_value: float
    significant: bool
    
    # Time series
    actual: np.ndarray
    predicted: np.ndarray
    counterfactual: np.ndarray
    effect_series: np.ndarray
    time_index: np.ndarray
    
    # Pre-period diagnostics
    pre_period_mape: float
    parallel_trends_pvalue: float
    
    # Metadata
    treatment_start: int
    treatment_end: int
    method: str
    diagnostics: Dict = field(default_factory=dict)


class CausalImpactAnalyzer:
    """
    Causal Impact Analysis for Geo-Based Experiments
    
    Implements multiple methods for measuring causal effects:
    - Difference-in-Differences (DiD)
    - Synthetic DiD
    - Bayesian Structural Time Series (simplified)
    
    Parameters
    ----------
    method : str
        Analysis method: 'did', 'synthetic_did', 'bsts'
    credible_interval : float
        Credible interval width (default: 0.95)
    n_bootstrap : int
        Bootstrap samples for uncertainty (default: 1000)
    seasonality : bool
        Model seasonality in BSTS (default: True)
    """
    
    def __init__(
        self,
        method: Literal['did', 'synthetic_did', 'bsts'] = 'did',
        credible_interval: float = 0.95,
        n_bootstrap: int = 1000,
        seasonality: bool = True,
        random_state: int = 42
    ):
        self.method = method
        self.credible_interval = credible_interval
        self.n_bootstrap = n_bootstrap
        self.seasonality = seasonality
        self.random_state = random_state
        np.random.seed(random_state)
    
    def analyze(
        self,
        treatment_series: np.ndarray,
        control_series: np.ndarray,
        treatment_start: int,
        treatment_end: Optional[int] = None,
        time_index: Optional[np.ndarray] = None
    ) -> CausalImpactResult:
        """
        Analyze causal impact of intervention.
        
        Parameters
        ----------
        treatment_series : array
            Time series for treatment group (aggregated)
        control_series : array
            Time series for control group (aggregated)
        treatment_start : int
            Index where treatment begins
        treatment_end : int
            Index where treatment ends (default: end of series)
        time_index : array
            Time labels for x-axis
            
        Returns
        -------
        CausalImpactResult
            Comprehensive causal impact results
        """
        if treatment_end is None:
            treatment_end = len(treatment_series)
        
        if time_index is None:
            time_index = np.arange(len(treatment_series))
        
        # Validate inputs
        assert len(treatment_series) == len(control_series), \
            "Treatment and control series must have same length"
        assert treatment_start > 0, \
            "Need at least 1 pre-period"
        assert treatment_start < len(treatment_series), \
            "Treatment start must be within series"
        
        # Split pre/post periods
        pre_treatment = slice(0, treatment_start)
        post_treatment = slice(treatment_start, treatment_end)
        
        # Test parallel trends
        pt_pvalue = self._test_parallel_trends(
            treatment_series[pre_treatment],
            control_series[pre_treatment]
        )
        
        # Estimate counterfactual based on method
        if self.method == 'did':
            counterfactual = self._did_counterfactual(
                treatment_series, control_series, treatment_start
            )
        elif self.method == 'synthetic_did':
            counterfactual = self._synthetic_did_counterfactual(
                treatment_series, control_series, treatment_start
            )
        else:  # bsts
            counterfactual = self._bsts_counterfactual(
                treatment_series, control_series, treatment_start
            )
        
        # Calculate effects
        effect_series = treatment_series - counterfactual
        post_effects = effect_series[post_treatment]
        
        average_effect = np.mean(post_effects)
        cumulative_effect = np.sum(post_effects)
        
        # Relative effect (percentage lift)
        baseline = np.mean(counterfactual[post_treatment])
        relative_effect = average_effect / baseline if baseline != 0 else 0
        
        # Bootstrap confidence intervals
        bootstrap_effects = self._bootstrap_effects(
            treatment_series, control_series, treatment_start, treatment_end
        )
        
        alpha = 1 - self.credible_interval
        effect_lower = np.percentile(bootstrap_effects, alpha / 2 * 100)
        effect_upper = np.percentile(bootstrap_effects, (1 - alpha / 2) * 100)
        
        # P-value (proportion of bootstrap samples crossing zero)
        if average_effect >= 0:
            p_value = np.mean(bootstrap_effects <= 0)
        else:
            p_value = np.mean(bootstrap_effects >= 0)
        p_value = min(p_value * 2, 1.0)  # Two-sided
        
        significant = p_value < (1 - self.credible_interval)
        
        # Pre-period fit
        pre_mape = np.mean(np.abs(
            treatment_series[pre_treatment] - counterfactual[pre_treatment]
        ) / treatment_series[pre_treatment]) * 100
        
        # Diagnostics
        diagnostics = {
            'bootstrap_effects': bootstrap_effects,
            'n_bootstrap': self.n_bootstrap,
            'pre_period': pre_treatment,
            'post_period': post_treatment
        }
        
        return CausalImpactResult(
            average_effect=average_effect,
            cumulative_effect=cumulative_effect,
            relative_effect=relative_effect,
            effect_lower=effect_lower,
            effect_upper=effect_upper,
            credible_interval=self.credible_interval,
            p_value=p_value,
            significant=significant,
            actual=treatment_series,
            predicted=counterfactual,  # Would be model fit for BSTS
            counterfactual=counterfactual,
            effect_series=effect_series,
            time_index=time_index,
            pre_period_mape=pre_mape,
            parallel_trends_pvalue=pt_pvalue,
            treatment_start=treatment_start,
            treatment_end=treatment_end,
            method=self.method,
            diagnostics=diagnostics
        )
    
    def _test_parallel_trends(
        self,
        treatment_pre: np.ndarray,
        control_pre: np.ndarray
    ) -> float:
        """
        Test for parallel trends assumption.
        
        Uses regression of treatment-control gap on time.
        Null hypothesis: no diverging trend (slope = 0)
        """
        gap = treatment_pre - control_pre
        time = np.arange(len(gap))
        
        # Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(time, gap)
        
        return p_value
    
    def _did_counterfactual(
        self,
        treatment: np.ndarray,
        control: np.ndarray,
        treatment_start: int
    ) -> np.ndarray:
        """
        Standard difference-in-differences counterfactual.
        
        Assumes parallel trends and estimates treatment effect as:
        tau = (Y_T,post - Y_T,pre) - (Y_C,post - Y_C,pre)
        
        Counterfactual: Y_T,pre + (Y_C,post - Y_C,pre)
        """
        # Pre-period ratio
        pre_treatment_mean = np.mean(treatment[:treatment_start])
        pre_control_mean = np.mean(control[:treatment_start])
        
        # Level adjustment
        level_diff = pre_treatment_mean - pre_control_mean
        
        # Counterfactual is control + level difference
        counterfactual = control + level_diff
        
        return counterfactual
    
    def _synthetic_did_counterfactual(
        self,
        treatment: np.ndarray,
        control: np.ndarray,
        treatment_start: int
    ) -> np.ndarray:
        """
        Synthetic DiD counterfactual.
        
        Estimates optimal scaling factor in pre-period
        and applies to post-period.
        """
        # Find optimal scale factor in pre-period
        def objective(scale):
            pred = control[:treatment_start] * scale
            return np.sum((treatment[:treatment_start] - pred) ** 2)
        
        result = minimize(objective, x0=1.0, method='Nelder-Mead')
        scale = result.x[0]
        
        # Apply to full series
        counterfactual = control * scale
        
        return counterfactual
    
    def _bsts_counterfactual(
        self,
        treatment: np.ndarray,
        control: np.ndarray,
        treatment_start: int
    ) -> np.ndarray:
        """
        Simplified Bayesian Structural Time Series counterfactual.
        
        Uses linear regression with control as covariate.
        Full BSTS would use PyMC or Stan.
        """
        # Fit linear model on pre-period
        X_pre = control[:treatment_start].reshape(-1, 1)
        y_pre = treatment[:treatment_start]
        
        # Add time trend if seasonality
        if self.seasonality:
            time = np.arange(len(treatment)).reshape(-1, 1)
            time_pre = time[:treatment_start]
            X_pre = np.hstack([X_pre, time_pre])
        
        # OLS fit
        X_pre_aug = np.hstack([np.ones((len(X_pre), 1)), X_pre])
        beta = np.linalg.lstsq(X_pre_aug, y_pre, rcond=None)[0]
        
        # Predict full series
        X_full = control.reshape(-1, 1)
        if self.seasonality:
            X_full = np.hstack([X_full, time])
        X_full_aug = np.hstack([np.ones((len(X_full), 1)), X_full])
        
        counterfactual = X_full_aug @ beta
        
        return counterfactual
    
    def _bootstrap_effects(
        self,
        treatment: np.ndarray,
        control: np.ndarray,
        treatment_start: int,
        treatment_end: int
    ) -> np.ndarray:
        """
        Bootstrap uncertainty estimation.
        
        Resamples pre-period residuals to generate counterfactual distribution.
        """
        # Get counterfactual
        if self.method == 'did':
            cf = self._did_counterfactual(treatment, control, treatment_start)
        elif self.method == 'synthetic_did':
            cf = self._synthetic_did_counterfactual(treatment, control, treatment_start)
        else:
            cf = self._bsts_counterfactual(treatment, control, treatment_start)
        
        # Pre-period residuals
        residuals = treatment[:treatment_start] - cf[:treatment_start]
        res_std = np.std(residuals)
        n_post = treatment_end - treatment_start
        
        # Bootstrap
        bootstrap_effects = []
        block_size = min(8, treatment_start)  # Use blocks to capture temporal correlation
        
        for _ in range(self.n_bootstrap):
            # Block Bootstrap for residuals
            resampled_residuals = []
            while len(resampled_residuals) < n_post:
                start_idx = np.random.randint(0, treatment_start - block_size + 1)
                block = residuals[start_idx : start_idx + block_size]
                resampled_residuals.extend(block)
            
            resampled_residuals = np.array(resampled_residuals[:n_post])
            
            # Add Gaussian jitter with a "safety multiplier" to account for 
            # post-period variance shifts and potential selection bias.
            # 1.5x multiplier handles the "regression to mean" effect in matched units.
            jitter = np.random.normal(0, res_std * 1.5, size=n_post)
            
            # Add noise to counterfactual
            noisy_cf = cf[treatment_start:treatment_end] + resampled_residuals + jitter
            
            # Effect
            effect = np.mean(treatment[treatment_start:treatment_end] - noisy_cf)
            bootstrap_effects.append(effect)
        
        return np.array(bootstrap_effects)
    
    def analyze_from_geo_data(
        self,
        geo_data: pd.DataFrame,
        treatment_geos: List[str],
        control_geos: List[str],
        treatment_start: int,
        treatment_end: Optional[int] = None,
        geo_col: str = 'geo_id',
        time_col: str = 'period',
        metric_col: str = 'revenue'
    ) -> CausalImpactResult:
        """
        Analyze causal impact from geo-level panel data.
        
        Aggregates treatment and control geos before analysis.
        """
        # Aggregate by time
        treatment_data = geo_data[geo_data[geo_col].isin(treatment_geos)]
        control_data = geo_data[geo_data[geo_col].isin(control_geos)]
        
        treatment_series = treatment_data.groupby(time_col)[metric_col].sum().values
        control_series = control_data.groupby(time_col)[metric_col].sum().values
        
        time_index = treatment_data.groupby(time_col)[metric_col].sum().index.values
        
        return self.analyze(
            treatment_series,
            control_series,
            treatment_start,
            treatment_end,
            time_index
        )
    
    def run_robustness_checks(
        self,
        result: CausalImpactResult,
        treatment_series: np.ndarray,
        control_series: np.ndarray
    ) -> Dict:
        """
        Run robustness checks on causal impact estimate.
        
        Checks:
        1. Leave-one-out sensitivity
        2. Alternative treatment timing
        3. Placebo treatment in pre-period
        """
        checks = {}
        
        treatment_start = result.treatment_start
        
        # 1. Leave-one-out on pre-period
        loo_effects = []
        for i in range(treatment_start):
            t_loo = np.delete(treatment_series[:treatment_start], i)
            c_loo = np.delete(control_series[:treatment_start], i)
            
            # Re-estimate effect with reduced pre-period
            level_diff = np.mean(t_loo) - np.mean(c_loo)
            post_effect = np.mean(
                treatment_series[treatment_start:] - 
                (control_series[treatment_start:] + level_diff)
            )
            loo_effects.append(post_effect)
        
        loo_effects = np.array(loo_effects)
        checks['loo_mean'] = np.mean(loo_effects)
        checks['loo_std'] = np.std(loo_effects)
        checks['loo_stable'] = checks['loo_std'] < 0.1 * abs(result.average_effect)
        
        # 2. Placebo treatment at midpoint of pre-period
        placebo_start = treatment_start // 2
        if placebo_start > 2:
            placebo_result = self.analyze(
                treatment_series[:treatment_start],
                control_series[:treatment_start],
                placebo_start
            )
            checks['placebo_effect'] = placebo_result.average_effect
            checks['placebo_pvalue'] = placebo_result.p_value
            checks['placebo_passed'] = placebo_result.p_value > 0.10
        else:
            checks['placebo_passed'] = None
        
        # 3. Effect sensitivity to treatment timing (±1 period)
        timing_effects = []
        for offset in [-1, 0, 1]:
            alt_start = treatment_start + offset
            if 2 <= alt_start < len(treatment_series) - 2:
                alt_result = self.analyze(
                    treatment_series, control_series, alt_start
                )
                timing_effects.append(alt_result.average_effect)
        
        checks['timing_sensitivity'] = np.std(timing_effects) if timing_effects else 0
        checks['timing_stable'] = checks['timing_sensitivity'] < 0.2 * abs(result.average_effect)
        
        # Overall robustness score
        robustness_flags = [
            checks.get('loo_stable', False),
            checks.get('placebo_passed', True),  # None counts as pass
            checks.get('timing_stable', False)
        ]
        checks['robustness_score'] = sum(robustness_flags) / len(robustness_flags)
        
        return checks


def calculate_incremental_lift(
    result: CausalImpactResult,
    total_spend: float,
    baseline_metric: str = 'revenue'
) -> Dict:
    """
    Calculate business metrics from causal impact result.
    
    Parameters
    ----------
    result : CausalImpactResult
        Output from CausalImpactAnalyzer
    total_spend : float
        Total marketing spend during treatment period
        
    Returns
    -------
    dict
        Business metrics including iROAS, incremental revenue, etc.
    """
    incremental_revenue = result.cumulative_effect
    
    # Incremental ROAS
    iroas = incremental_revenue / total_spend if total_spend > 0 else 0
    
    # Confidence interval on iROAS
    iroas_lower = result.effect_lower * (result.treatment_end - result.treatment_start) / total_spend
    iroas_upper = result.effect_upper * (result.treatment_end - result.treatment_start) / total_spend
    
    return {
        'incremental_revenue': incremental_revenue,
        'incremental_roas': iroas,
        'iroas_lower': iroas_lower,
        'iroas_upper': iroas_upper,
        'relative_lift': result.relative_effect,
        'lift_lower': result.effect_lower / np.mean(result.counterfactual[result.treatment_start:]),
        'lift_upper': result.effect_upper / np.mean(result.counterfactual[result.treatment_start:]),
        'p_value': result.p_value,
        'significant': result.significant,
        'total_spend': total_spend,
        'cost_per_incremental_dollar': total_spend / incremental_revenue if incremental_revenue > 0 else np.inf
    }


if __name__ == '__main__':
    print("=" * 60)
    print("CAUSAL IMPACT ANALYZER DEMONSTRATION")
    print("=" * 60)
    
    # Generate synthetic data with known treatment effect
    np.random.seed(42)
    n_periods = 52
    treatment_start = 40
    true_lift = 0.12  # 12% lift
    
    # Base series with trend and seasonality
    trend = np.linspace(100, 120, n_periods)
    seasonality = 10 * np.sin(np.linspace(0, 4*np.pi, n_periods))
    noise_control = np.random.normal(0, 3, n_periods)
    noise_treatment = np.random.normal(0, 3, n_periods)
    
    control_series = trend + seasonality + noise_control
    treatment_series = trend + seasonality + noise_treatment
    
    # Add treatment effect
    treatment_series[treatment_start:] *= (1 + true_lift)
    
    print(f"\n1. Synthetic data generated:")
    print(f"   Periods: {n_periods}")
    print(f"   Treatment start: {treatment_start}")
    print(f"   True lift: {true_lift*100:.0f}%")
    
    # Analyze with DiD
    print("\n2. Running Difference-in-Differences analysis...")
    analyzer = CausalImpactAnalyzer(method='did', n_bootstrap=500)
    result = analyzer.analyze(treatment_series, control_series, treatment_start)
    
    print(f"\n3. Results:")
    print(f"   Method: {result.method}")
    print(f"   Average effect: {result.average_effect:.2f}")
    print(f"   Relative lift: {result.relative_effect*100:.1f}%")
    print(f"   95% CI: [{result.effect_lower:.2f}, {result.effect_upper:.2f}]")
    print(f"   P-value: {result.p_value:.4f}")
    print(f"   Significant: {result.significant}")
    
    # Compare to truth
    print(f"\n4. Accuracy:")
    print(f"   True lift: {true_lift*100:.0f}%")
    print(f"   Estimated lift: {result.relative_effect*100:.1f}%")
    print(f"   Error: {abs(result.relative_effect - true_lift)*100:.1f}pp")
    
    # Diagnostics
    print(f"\n5. Diagnostics:")
    print(f"   Pre-period MAPE: {result.pre_period_mape:.2f}%")
    print(f"   Parallel trends p-value: {result.parallel_trends_pvalue:.3f}")
    
    # Robustness
    print("\n6. Running robustness checks...")
    robustness = analyzer.run_robustness_checks(
        result, treatment_series, control_series
    )
    print(f"   Leave-one-out stable: {robustness['loo_stable']}")
    print(f"   Timing stable: {robustness['timing_stable']}")
    print(f"   Robustness score: {robustness['robustness_score']*100:.0f}%")
    
    # Business metrics
    print("\n7. Business impact (assuming $100K spend):")
    metrics = calculate_incremental_lift(result, total_spend=100000)
    print(f"   Incremental revenue: ${metrics['incremental_revenue']:,.0f}")
    print(f"   Incremental ROAS: ${metrics['incremental_roas']:.2f}")
    print(f"   Cost per incremental $: ${metrics['cost_per_incremental_dollar']:.2f}")
    
    print("\n" + "=" * 60)
    print("CAUSAL IMPACT ANALYSIS COMPLETE")
    print("=" * 60)
