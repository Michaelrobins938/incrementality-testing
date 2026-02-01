"""
Geo Power Analyzer
==================

Power analysis for geo-based incrementality tests.

Key Features:
- Minimum detectable effect (MDE) calculation
- Required sample size (geos and duration)
- Power curves and sensitivity analysis
- Test duration recommendations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.optimize import brentq
import warnings


@dataclass
class PowerResult:
    """Container for power analysis results."""
    # Core metrics
    power: float
    mde: float  # Minimum detectable effect (relative)
    required_geos: int
    required_duration: int  # Weeks
    
    # Inputs
    baseline_metric: float
    baseline_std: float
    alpha: float
    target_power: float
    
    # Diagnostics
    n_treatment: int
    n_control: int
    effect_size: float  # Cohen's d
    intra_geo_correlation: float
    design_effect: float
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class GeoPowerAnalyzer:
    """
    Power Analysis for Geo-Based Experiments
    
    Accounts for:
    - Clustered design (geos as units)
    - Repeated measures (time series)
    - Intra-class correlation
    - Design effect
    
    Parameters
    ----------
    alpha : float
        Significance level (default: 0.05)
    power : float
        Target statistical power (default: 0.80)
    one_sided : bool
        One-sided vs two-sided test (default: False)
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        power: float = 0.80,
        one_sided: bool = False,
        random_state: int = 42
    ):
        self.alpha = alpha
        self.target_power = power
        self.one_sided = one_sided
        self.random_state = random_state
        np.random.seed(random_state)
    
    def calculate_mde(
        self,
        n_geos: int,
        n_periods: int,
        baseline_mean: float,
        baseline_std: float,
        icc: float = 0.1,
        treatment_fraction: float = 0.5
    ) -> PowerResult:
        """
        Calculate minimum detectable effect for given sample size.
        
        Parameters
        ----------
        n_geos : int
            Total number of geos in experiment
        n_periods : int
            Number of time periods in test
        baseline_mean : float
            Mean of outcome metric
        baseline_std : float
            Standard deviation of outcome metric
        icc : float
            Intra-class correlation (within-geo correlation)
        treatment_fraction : float
            Fraction assigned to treatment
            
        Returns
        -------
        PowerResult
            Power analysis results including MDE
        """
        n_treatment = int(n_geos * treatment_fraction)
        n_control = n_geos - n_treatment
        
        # Design effect for clustered design
        design_effect = 1 + (n_periods - 1) * icc
        
        # Effective sample size
        n_eff = (n_treatment * n_control * n_periods) / (n_geos * design_effect)
        
        # Critical values
        if self.one_sided:
            z_alpha = stats.norm.ppf(1 - self.alpha)
        else:
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.target_power)
        
        # MDE (in standard deviations)
        se = baseline_std * np.sqrt(design_effect / n_eff)
        mde_absolute = (z_alpha + z_beta) * se * np.sqrt(1/n_treatment + 1/n_control)
        
        # Relative MDE
        mde_relative = mde_absolute / baseline_mean if baseline_mean > 0 else np.inf
        
        # Effect size (Cohen's d)
        effect_size = mde_absolute / baseline_std
        
        # Recommendations
        recommendations = self._generate_recommendations(
            n_geos, n_periods, mde_relative, icc, effect_size
        )
        
        return PowerResult(
            power=self.target_power,
            mde=mde_relative,
            required_geos=n_geos,
            required_duration=n_periods,
            baseline_metric=baseline_mean,
            baseline_std=baseline_std,
            alpha=self.alpha,
            target_power=self.target_power,
            n_treatment=n_treatment,
            n_control=n_control,
            effect_size=effect_size,
            intra_geo_correlation=icc,
            design_effect=design_effect,
            recommendations=recommendations
        )
    
    def calculate_required_geos(
        self,
        target_mde: float,
        n_periods: int,
        baseline_mean: float,
        baseline_std: float,
        icc: float = 0.1,
        treatment_fraction: float = 0.5
    ) -> PowerResult:
        """
        Calculate required number of geos for target MDE.
        
        Parameters
        ----------
        target_mde : float
            Target minimum detectable effect (relative, e.g., 0.05 = 5%)
        n_periods : int
            Number of time periods in test
        Other parameters same as calculate_mde
            
        Returns
        -------
        PowerResult
            Power analysis results including required geos
        """
        mde_absolute = target_mde * baseline_mean
        
        # Critical values
        if self.one_sided:
            z_alpha = stats.norm.ppf(1 - self.alpha)
        else:
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.target_power)
        
        # Design effect
        design_effect = 1 + (n_periods - 1) * icc
        
        # Solve for n_geos
        # MDE = (z_alpha + z_beta) * sigma * sqrt(DE) * sqrt(1/n_t + 1/n_c) / sqrt(n_periods)
        # For equal split: n_t = n_c = n/2
        # MDE = (z_alpha + z_beta) * sigma * sqrt(DE) * sqrt(4/n) / sqrt(n_periods)
        
        variance_factor = (z_alpha + z_beta) ** 2 * baseline_std ** 2 * design_effect
        
        # For equal allocation
        if treatment_fraction == 0.5:
            required_n = 4 * variance_factor / (mde_absolute ** 2 * n_periods)
        else:
            # General case
            k = treatment_fraction
            required_n = variance_factor * (1/(k*(1-k))) / (mde_absolute ** 2 * n_periods)
        
        required_geos = int(np.ceil(required_n))
        required_geos = max(required_geos, 4)  # Minimum 4 geos
        
        # Calculate actual power achieved
        actual_result = self.calculate_mde(
            required_geos, n_periods, baseline_mean, baseline_std, icc, treatment_fraction
        )
        
        return PowerResult(
            power=self.target_power,
            mde=actual_result.mde,
            required_geos=required_geos,
            required_duration=n_periods,
            baseline_metric=baseline_mean,
            baseline_std=baseline_std,
            alpha=self.alpha,
            target_power=self.target_power,
            n_treatment=int(required_geos * treatment_fraction),
            n_control=required_geos - int(required_geos * treatment_fraction),
            effect_size=actual_result.effect_size,
            intra_geo_correlation=icc,
            design_effect=design_effect,
            recommendations=actual_result.recommendations
        )
    
    def calculate_required_duration(
        self,
        target_mde: float,
        n_geos: int,
        baseline_mean: float,
        baseline_std: float,
        icc: float = 0.1,
        treatment_fraction: float = 0.5,
        max_weeks: int = 52
    ) -> PowerResult:
        """
        Calculate required test duration for target MDE.
        """
        mde_absolute = target_mde * baseline_mean
        
        # Critical values
        if self.one_sided:
            z_alpha = stats.norm.ppf(1 - self.alpha)
        else:
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.target_power)
        
        # Search for required duration
        for n_periods in range(1, max_weeks + 1):
            result = self.calculate_mde(
                n_geos, n_periods, baseline_mean, baseline_std, icc, treatment_fraction
            )
            if result.mde <= target_mde:
                return result
        
        # If max weeks not enough
        warnings.warn(
            f"Cannot achieve MDE of {target_mde:.2%} with {n_geos} geos "
            f"in {max_weeks} weeks. Consider more geos."
        )
        return self.calculate_mde(
            n_geos, max_weeks, baseline_mean, baseline_std, icc, treatment_fraction
        )
    
    def power_curve(
        self,
        n_geos: int,
        n_periods: int,
        baseline_mean: float,
        baseline_std: float,
        icc: float = 0.1,
        effect_range: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate power curve for range of effect sizes.
        
        Returns
        -------
        dict
            'effects': array of effect sizes
            'power': array of power values
        """
        if effect_range is None:
            effect_range = np.linspace(0.01, 0.20, 20)
        
        powers = []
        for effect in effect_range:
            power = self._calculate_power_for_effect(
                effect, n_geos, n_periods, baseline_mean, baseline_std, icc
            )
            powers.append(power)
        
        return {
            'effects': effect_range,
            'power': np.array(powers)
        }
    
    def _calculate_power_for_effect(
        self,
        effect: float,
        n_geos: int,
        n_periods: int,
        baseline_mean: float,
        baseline_std: float,
        icc: float
    ) -> float:
        """Calculate power for specific effect size."""
        n_treatment = n_geos // 2
        n_control = n_geos - n_treatment
        
        # Design effect
        design_effect = 1 + (n_periods - 1) * icc
        
        # Effect in absolute terms
        effect_absolute = effect * baseline_mean
        
        # Standard error
        n_eff = (n_treatment * n_control * n_periods) / (n_geos * design_effect)
        se = baseline_std * np.sqrt(design_effect / n_eff) * np.sqrt(1/n_treatment + 1/n_control)
        
        # Non-centrality parameter
        ncp = effect_absolute / se
        
        # Critical value
        if self.one_sided:
            z_crit = stats.norm.ppf(1 - self.alpha)
        else:
            z_crit = stats.norm.ppf(1 - self.alpha / 2)
        
        # Power
        power = 1 - stats.norm.cdf(z_crit - ncp)
        if not self.one_sided:
            power += stats.norm.cdf(-z_crit - ncp)
        
        return power
    
    def estimate_icc_from_data(
        self,
        geo_data: pd.DataFrame,
        geo_col: str = 'geo_id',
        time_col: str = 'period',
        metric_col: str = 'revenue'
    ) -> float:
        """
        Estimate intra-class correlation from historical data.
        
        Uses one-way random effects ANOVA.
        """
        # Group statistics
        geo_means = geo_data.groupby(geo_col)[metric_col].mean()
        overall_mean = geo_data[metric_col].mean()
        
        # Between-geo variance
        n_per_geo = geo_data.groupby(geo_col)[metric_col].count().mean()
        ssb = n_per_geo * sum((geo_means - overall_mean) ** 2)
        dfb = len(geo_means) - 1
        msb = ssb / dfb
        
        # Within-geo variance
        ssw = sum(
            sum((geo_data[geo_data[geo_col] == geo][metric_col] - geo_means[geo]) ** 2)
            for geo in geo_means.index
        )
        dfw = len(geo_data) - len(geo_means)
        msw = ssw / dfw
        
        # ICC (one-way random effects)
        icc = (msb - msw) / (msb + (n_per_geo - 1) * msw)
        
        return max(0, min(1, icc))  # Bound to [0, 1]
    
    def _generate_recommendations(
        self,
        n_geos: int,
        n_periods: int,
        mde: float,
        icc: float,
        effect_size: float
    ) -> List[str]:
        """Generate actionable recommendations based on power analysis."""
        recommendations = []
        
        if mde > 0.10:
            recommendations.append(
                f"MDE of {mde:.1%} is high. Consider: "
                f"(1) More geos (current: {n_geos}), "
                f"(2) Longer test (current: {n_periods} periods), "
                f"(3) Variance reduction (CUPED)"
            )
        
        if n_geos < 10:
            recommendations.append(
                f"Only {n_geos} geos may limit matching quality. "
                f"Consider at least 20 geos for robust results."
            )
        
        if n_periods < 4:
            recommendations.append(
                f"Test duration of {n_periods} periods is short. "
                f"Consider at least 4 weeks for stable estimates."
            )
        
        if n_periods > 26:
            recommendations.append(
                f"Test duration of {n_periods} periods is long. "
                f"Risk of external shocks or drift increases."
            )
        
        if icc > 0.3:
            recommendations.append(
                f"High ICC ({icc:.2f}) indicates strong geo clustering. "
                f"Consider stratified assignment or propensity matching."
            )
        
        if effect_size < 0.2:
            recommendations.append(
                f"Small effect size ({effect_size:.2f}). "
                f"Ensure business impact justifies test investment."
            )
        
        if not recommendations:
            recommendations.append(
                "Design looks reasonable. Proceed with caution on "
                "external validity and parallel trends assumption."
            )
        
        return recommendations
    
    def simulate_power(
        self,
        n_geos: int,
        n_periods: int,
        true_effect: float,
        baseline_mean: float,
        baseline_std: float,
        icc: float = 0.1,
        n_simulations: int = 1000
    ) -> Dict[str, float]:
        """
        Simulate power via Monte Carlo.
        
        More accurate than analytical formula for complex designs.
        """
        significant_count = 0
        effect_estimates = []
        
        for _ in range(n_simulations):
            # Generate data
            data = self._simulate_geo_experiment(
                n_geos, n_periods, true_effect, baseline_mean, baseline_std, icc
            )
            
            # Analyze
            treatment_series = data[data['treatment']].groupby('period')['metric'].mean()
            control_series = data[~data['treatment']].groupby('period')['metric'].mean()
            
            # Simple t-test on post-period differences
            post_mask = slice(n_periods // 2, n_periods)
            t_post = treatment_series.values[post_mask]
            c_post = control_series.values[post_mask]
            
            # Estimate effect
            effect_estimate = (np.mean(t_post) - np.mean(c_post)) / np.mean(c_post)
            effect_estimates.append(effect_estimate)
            
            # Test significance
            stat, pval = stats.ttest_ind(t_post, c_post)
            if pval < self.alpha:
                significant_count += 1
        
        simulated_power = significant_count / n_simulations
        effect_estimates = np.array(effect_estimates)
        
        return {
            'simulated_power': simulated_power,
            'mean_effect_estimate': np.mean(effect_estimates),
            'effect_bias': np.mean(effect_estimates) - true_effect,
            'effect_rmse': np.sqrt(np.mean((effect_estimates - true_effect) ** 2)),
            'n_simulations': n_simulations
        }
    
    def _simulate_geo_experiment(
        self,
        n_geos: int,
        n_periods: int,
        true_effect: float,
        baseline_mean: float,
        baseline_std: float,
        icc: float
    ) -> pd.DataFrame:
        """Generate one simulated geo experiment dataset."""
        records = []
        
        # Assign treatment (first half)
        treatment_geos = set(range(n_geos // 2))
        
        # Geo random effects
        geo_effects = np.random.normal(0, baseline_std * np.sqrt(icc), n_geos)
        
        # Within-geo std
        within_std = baseline_std * np.sqrt(1 - icc)
        
        for geo in range(n_geos):
            is_treatment = geo in treatment_geos
            
            for period in range(n_periods):
                # Base metric
                metric = baseline_mean + geo_effects[geo]
                metric += np.random.normal(0, within_std)
                
                # Add treatment effect (post-period only)
                if is_treatment and period >= n_periods // 2:
                    metric *= (1 + true_effect)
                
                records.append({
                    'geo_id': f'geo_{geo}',
                    'period': period,
                    'metric': metric,
                    'treatment': is_treatment
                })
        
        return pd.DataFrame(records)


def plan_geo_test(
    historical_data: pd.DataFrame,
    target_mde: float = 0.05,
    target_power: float = 0.80,
    alpha: float = 0.05,
    min_weeks: int = 4,
    max_weeks: int = 12,
    geo_col: str = 'geo_id',
    time_col: str = 'period',
    metric_col: str = 'revenue'
) -> Dict:
    """
    Comprehensive test planning from historical data.
    
    Returns test design with power analysis, timeline, and recommendations.
    """
    analyzer = GeoPowerAnalyzer(alpha=alpha, power=target_power)
    
    # Estimate parameters from data
    baseline_mean = historical_data[metric_col].mean()
    baseline_std = historical_data[metric_col].std()
    n_geos = historical_data[geo_col].nunique()
    icc = analyzer.estimate_icc_from_data(historical_data, geo_col, time_col, metric_col)
    
    # Calculate for different durations
    results_by_duration = {}
    for weeks in range(min_weeks, max_weeks + 1):
        result = analyzer.calculate_mde(
            n_geos, weeks, baseline_mean, baseline_std, icc
        )
        results_by_duration[weeks] = {
            'mde': result.mde,
            'power': result.power,
            'effect_size': result.effect_size
        }
    
    # Find minimum duration for target MDE
    optimal_result = analyzer.calculate_required_duration(
        target_mde, n_geos, baseline_mean, baseline_std, icc, max_weeks=max_weeks
    )
    
    # Power curve
    power_curve = analyzer.power_curve(
        n_geos, optimal_result.required_duration,
        baseline_mean, baseline_std, icc
    )
    
    return {
        'test_design': {
            'n_geos': n_geos,
            'n_treatment': n_geos // 2,
            'n_control': n_geos - n_geos // 2,
            'duration_weeks': optimal_result.required_duration,
            'target_mde': target_mde,
            'achievable_mde': optimal_result.mde
        },
        'parameters': {
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'icc': icc,
            'design_effect': optimal_result.design_effect
        },
        'results_by_duration': results_by_duration,
        'power_curve': power_curve,
        'recommendations': optimal_result.recommendations
    }


if __name__ == '__main__':
    print("=" * 60)
    print("GEO POWER ANALYZER DEMONSTRATION")
    print("=" * 60)
    
    analyzer = GeoPowerAnalyzer(alpha=0.05, power=0.80)
    
    # Example 1: Calculate MDE for given design
    print("\n1. MDE Calculation for 50 geos, 8 weeks:")
    result = analyzer.calculate_mde(
        n_geos=50,
        n_periods=8,
        baseline_mean=10000,
        baseline_std=2000,
        icc=0.15
    )
    print(f"   MDE: {result.mde:.1%}")
    print(f"   Effect size (Cohen's d): {result.effect_size:.3f}")
    print(f"   Design effect: {result.design_effect:.2f}")
    
    # Example 2: Required geos for 5% MDE
    print("\n2. Required geos for 5% MDE:")
    result = analyzer.calculate_required_geos(
        target_mde=0.05,
        n_periods=8,
        baseline_mean=10000,
        baseline_std=2000,
        icc=0.15
    )
    print(f"   Required geos: {result.required_geos}")
    print(f"   Achievable MDE: {result.mde:.1%}")
    
    # Example 3: Required duration for 5% MDE with 50 geos
    print("\n3. Required duration for 5% MDE with 50 geos:")
    result = analyzer.calculate_required_duration(
        target_mde=0.05,
        n_geos=50,
        baseline_mean=10000,
        baseline_std=2000,
        icc=0.15
    )
    print(f"   Required weeks: {result.required_duration}")
    print(f"   Achievable MDE: {result.mde:.1%}")
    
    # Example 4: Power curve
    print("\n4. Power curve (50 geos, 8 weeks):")
    curve = analyzer.power_curve(
        n_geos=50,
        n_periods=8,
        baseline_mean=10000,
        baseline_std=2000,
        icc=0.15
    )
    print("   Effect  | Power")
    print("   --------|------")
    for effect, power in zip(curve['effects'][::4], curve['power'][::4]):
        print(f"   {effect:6.1%} | {power:.0%}")
    
    # Example 5: Recommendations
    print("\n5. Recommendations:")
    for rec in result.recommendations:
        print(f"   - {rec}")
    
    print("\n" + "=" * 60)
    print("POWER ANALYSIS COMPLETE")
    print("=" * 60)
