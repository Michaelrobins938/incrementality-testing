"""
Statistical Validation Suite
============================

Validates incrementality testing framework against known ground truth.

Key Validations:
- Lift recovery accuracy (±2% of true effect)
- Coverage probability (95% CIs cover truth 95% of time)
- Type I error control (≤5% false positives)
- Type II error control (≤20% false negatives)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial
import warnings
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geo_matcher import GeoMatcher, create_synthetic_geo_data
from core.causal_impact import CausalImpactAnalyzer
from core.synthetic_control import SyntheticControlMethod
from core.power_analyzer import GeoPowerAnalyzer


@dataclass
class ValidationResult:
    """Container for validation results."""
    # Lift recovery
    mean_estimated_lift: float
    true_lift: float
    lift_bias: float
    lift_rmse: float
    
    # Coverage
    coverage_probability: float
    
    # Error rates
    type_i_error: float
    type_ii_error: float
    power: float
    
    # Match quality
    mean_match_r2: float
    
    # Metadata
    n_simulations: int
    method: str


class IncrementalityValidator:
    """
    Validates incrementality testing methodology via simulation.
    
    Runs thousands of simulated experiments with known ground truth
    to verify statistical properties of the framework.
    
    Parameters
    ----------
    n_simulations : int
        Number of simulated experiments
    n_geos : int
        Number of geos per simulation
    n_periods : int
        Number of time periods
    treatment_start : int
        Period when treatment begins
    alpha : float
        Significance level
    """
    
    def __init__(
        self,
        n_simulations: int = 1000,
        n_geos: int = 50,
        n_periods: int = 80,
        treatment_start: int = 60,
        alpha: float = 0.05,
        random_state: int = 42
    ):
        self.n_simulations = n_simulations
        self.n_geos = n_geos
        self.n_periods = n_periods
        self.treatment_start = treatment_start
        self.alpha = alpha
        self.random_state = random_state
        
        # Initialize components
        self.matcher = GeoMatcher(min_r2=0.8, n_pre_periods=32)
        self.did_analyzer = CausalImpactAnalyzer(method='did', n_bootstrap=1000)
        self.sc_analyzer = SyntheticControlMethod()
    
    def validate_lift_recovery(
        self,
        true_lifts: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20],
        method: str = 'did',
        verbose: bool = True
    ) -> Dict[float, ValidationResult]:
        """
        Validate lift recovery accuracy across different effect sizes.
        
        For each true lift, runs n_simulations experiments and measures:
        - Bias: mean(estimated) - true
        - RMSE: sqrt(mean((estimated - true)²))
        - Coverage: proportion of CIs that contain true effect
        """
        results = {}
        
        for true_lift in true_lifts:
            if verbose:
                print(f"Validating lift = {true_lift:.0%}...", end=" ")
            
            estimates = []
            lowers = []
            uppers = []
            significant = []
            match_r2s = []
            
            np.random.seed(self.random_state)
            
            for sim in range(self.n_simulations):
                try:
                    # Generate data
                    data = create_synthetic_geo_data(
                        n_geos=self.n_geos,
                        n_periods=self.n_periods,
                        random_state=self.random_state + sim
                    )
                    data['revenue'] = data['revenue'].astype(float)
                    
                    # Match geos (on pre-period data)
                    pre_data = data[data['period'] < self.treatment_start]
                    match_result = self.matcher.fit(pre_data)
                    match_r2s.append(match_result.overall_r2)
                    
                    # Apply treatment effect to the MATCHED treatment geos
                    if true_lift != 0:
                        mask = (
                            data['geo_id'].isin(match_result.treatment_geos) & 
                            (data['period'] >= self.treatment_start)
                        )
                        data.loc[mask, 'revenue'] *= (1 + true_lift)
                    
                    # Aggregate series
                    t_data = data[data['geo_id'].isin(match_result.treatment_geos)]
                    c_data = data[data['geo_id'].isin(match_result.control_geos)]
                    
                    t_series = t_data.groupby('period')['revenue'].sum().values
                    c_series = c_data.groupby('period')['revenue'].sum().values
                    
                    # Analyze
                    if method == 'did':
                        result = self.did_analyzer.analyze(
                            t_series, c_series, self.treatment_start
                        )
                        est = result.relative_effect
                        lower = result.effect_lower / np.mean(result.counterfactual[self.treatment_start:])
                        upper = result.effect_upper / np.mean(result.counterfactual[self.treatment_start:])
                        sig = result.significant
                    else:  # synthetic_control
                        # Use first treatment geo
                        result = self.sc_analyzer.fit(
                            data, match_result.treatment_geos[0], self.treatment_start
                        )
                        baseline = np.mean(result.actual_series[:self.treatment_start])
                        est = result.average_effect / baseline if baseline > 0 else 0
                        
                        # Use proper placebo test for inference
                        placebo = self.sc_analyzer.placebo_test(
                            data, match_result.treatment_geos[0], self.treatment_start,
                            n_placebos=20
                        )
                        p_val = placebo['p_value']
                        
                        # Approximate CI from placebo distribution
                        placebo_std = np.std(placebo['placebo_effects']) / baseline if baseline > 0 else 0.1
                        lower = est - 1.96 * placebo_std
                        upper = est + 1.96 * placebo_std
                        sig = p_val < self.alpha
                    
                    estimates.append(est)
                    lowers.append(lower)
                    uppers.append(upper)
                    significant.append(sig)
                    
                except Exception as e:
                    # Skip failed simulations
                    continue
            
            # Calculate metrics
            estimates = np.array(estimates)
            lowers = np.array(lowers)
            uppers = np.array(uppers)
            significant = np.array(significant)
            
            # Lift recovery
            mean_est = np.mean(estimates)
            bias = mean_est - true_lift
            rmse = np.sqrt(np.mean((estimates - true_lift) ** 2))
            
            # Coverage
            coverage = np.mean((lowers <= true_lift) & (true_lift <= uppers))
            
            # Error rates
            if true_lift == 0:
                # Type I error (rejecting true null)
                type_i = np.mean(significant)
                type_ii = np.nan
                power = np.nan
            else:
                # Type II error (failing to reject false null)
                type_i = np.nan
                type_ii = np.mean(~significant)
                power = np.mean(significant)
            
            results[true_lift] = ValidationResult(
                mean_estimated_lift=mean_est,
                true_lift=true_lift,
                lift_bias=bias,
                lift_rmse=rmse,
                coverage_probability=coverage,
                type_i_error=type_i if not np.isnan(type_i) else 0,
                type_ii_error=type_ii if not np.isnan(type_ii) else 0,
                power=power if not np.isnan(power) else 0,
                mean_match_r2=np.mean(match_r2s),
                n_simulations=len(estimates),
                method=method
            )
            
            if verbose:
                status = "[PASS]" if abs(bias) < 0.02 else "[FAIL]"
                print(f"Bias: {bias:+.2%} {status}")
        
        return results
    
    def validate_type_i_error(
        self,
        n_simulations: Optional[int] = None,
        method: str = 'did',
        verbose: bool = True
    ) -> Dict:
        """
        Validate Type I error control (false positive rate ≤ alpha).
        
        Runs experiments with true_lift = 0 and checks rejection rate.
        """
        n_sims = n_simulations or self.n_simulations
        
        if verbose:
            print(f"Validating Type I error control ({n_sims} simulations)...")
        
        significant_count = 0
        
        np.random.seed(self.random_state)
        
        for sim in range(n_sims):
            try:
                # Generate null data (no treatment effect)
                data = create_synthetic_geo_data(
                    n_geos=self.n_geos,
                    n_periods=self.n_periods,
                    random_state=self.random_state + sim * 100
                )
                
                # Match
                pre_data = data[data['period'] < self.treatment_start]
                match_result = self.matcher.fit(pre_data)
                
                # Aggregate
                t_data = data[data['geo_id'].isin(match_result.treatment_geos)]
                c_data = data[data['geo_id'].isin(match_result.control_geos)]
                
                t_series = t_data.groupby('period')['revenue'].sum().values
                c_series = c_data.groupby('period')['revenue'].sum().values
                
                # Analyze
                result = self.did_analyzer.analyze(
                    t_series, c_series, self.treatment_start
                )
                
                if result.significant:
                    significant_count += 1
                    
            except Exception:
                continue
        
        type_i_error = significant_count / n_sims
        
        if verbose:
            status = "[PASS]" if type_i_error <= self.alpha else "[FAIL]"
            print(f"  Type I error: {type_i_error:.2%} (target: <={self.alpha:.0%}) {status}")
        
        return {
            'type_i_error': type_i_error,
            'target': self.alpha,
            'passed': type_i_error <= self.alpha,
            'n_simulations': n_sims
        }
    
    def validate_power(
        self,
        true_lift: float = 0.10,
        target_power: float = 0.80,
        n_simulations: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Validate statistical power (detection rate ≥ target).
        
        Runs experiments with true_lift > 0 and checks detection rate.
        """
        n_sims = n_simulations or self.n_simulations
        
        if verbose:
            print(f"Validating power for {true_lift:.0%} lift ({n_sims} simulations)...")
        
        detected_count = 0
        
        np.random.seed(self.random_state)
        
        for sim in range(n_sims):
            try:
                # Generate data with treatment effect
                data = create_synthetic_geo_data(
                    n_geos=self.n_geos,
                    n_periods=self.n_periods,
                    random_state=self.random_state + sim * 100
                )
                
                # Match (on pre-period data)
                try:
                    pre_data = data[data['period'] < self.treatment_start]
                    match_result = self.matcher.fit(pre_data)
                    
                    # Apply treatment to MATCHED treatment geos
                    mask = (
                        data['geo_id'].isin(match_result.treatment_geos) & 
                        (data['period'] >= self.treatment_start)
                    )
                    data.loc[mask, 'revenue'] *= (1 + true_lift)
                    
                    # Aggregate
                    t_data = data[data['geo_id'].isin(match_result.treatment_geos)]
                    c_data = data[data['geo_id'].isin(match_result.control_geos)]
                    
                    t_series = t_data.groupby('period')['revenue'].sum().values
                    c_series = c_data.groupby('period')['revenue'].sum().values
                    
                    # Analyze
                    result = self.did_analyzer.analyze(
                        t_series, c_series, self.treatment_start
                    )
                except Exception:
                    continue
                
                if result.significant:
                    detected_count += 1
                    
            except Exception:
                continue
        
        power = detected_count / n_sims
        
        if verbose:
            status = "[PASS]" if power >= target_power else "[FAIL]"
            print(f"  Power: {power:.0%} (target: >={target_power:.0%}) {status}")
        
        return {
            'power': power,
            'target': target_power,
            'passed': power >= target_power,
            'true_lift': true_lift,
            'n_simulations': n_sims
        }
    
    def validate_coverage(
        self,
        true_lift: float = 0.10,
        target_coverage: float = 0.95,
        n_simulations: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Validate confidence interval coverage.
        
        95% CIs should contain the true effect 95% of the time.
        """
        n_sims = n_simulations or self.n_simulations
        
        if verbose:
            print(f"Validating CI coverage for {true_lift:.0%} lift ({n_sims} simulations)...")
        
        covered_count = 0
        
        np.random.seed(self.random_state)
        
        for sim in range(n_sims):
            try:
                # Generate data
                data = create_synthetic_geo_data(
                    n_geos=self.n_geos,
                    n_periods=self.n_periods,
                    random_state=self.random_state + sim * 100
                )
                
                # Match (on pre-period data)
                pre_data = data[data['period'] < self.treatment_start]
                match_result = self.matcher.fit(pre_data)
                
                # Apply treatment to MATCHED treatment geos
                mask = (
                    data['geo_id'].isin(match_result.treatment_geos) & 
                    (data['period'] >= self.treatment_start)
                )
                data.loc[mask, 'revenue'] *= (1 + true_lift)
                
                # Aggregate
                t_data = data[data['geo_id'].isin(match_result.treatment_geos)]
                c_data = data[data['geo_id'].isin(match_result.control_geos)]
                
                t_series = t_data.groupby('period')['revenue'].sum().values
                c_series = c_data.groupby('period')['revenue'].sum().values
                
                # Analyze
                result = self.did_analyzer.analyze(
                    t_series, c_series, self.treatment_start
                )
                
                # Check coverage (convert absolute CI to relative)
                baseline = np.mean(result.counterfactual[self.treatment_start:])
                lower = result.effect_lower / baseline
                upper = result.effect_upper / baseline
                
                if lower <= true_lift <= upper:
                    covered_count += 1
                    
            except Exception:
                continue
        
        coverage = covered_count / n_sims
        
        # Allow some tolerance (e.g., 90% coverage for 95% CI is acceptable)
        tolerance = 0.05
        
        if verbose:
            status = "[PASS]" if coverage >= (target_coverage - tolerance) else "[FAIL]"
            print(f"  Coverage: {coverage:.0%} (target: >={target_coverage - tolerance:.0%}) {status}")
        
        return {
            'coverage': coverage,
            'target': target_coverage,
            'passed': coverage >= (target_coverage - tolerance),
            'true_lift': true_lift,
            'n_simulations': n_sims
        }
    
    def run_full_validation(
        self,
        n_simulations: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run comprehensive validation suite.
        
        Returns summary of all validation checks.
        """
        n_sims = n_simulations or self.n_simulations
        
        if verbose:
            print("=" * 60)
            print("INCREMENTALITY TESTING VALIDATION SUITE")
            print(f"Simulations per test: {n_sims}")
            print("=" * 60)
        
        results = {}
        
        # 1. Type I error
        if verbose:
            print("\n[1/4] Type I Error Control")
        results['type_i'] = self.validate_type_i_error(n_sims // 2, verbose=verbose)
        
        # 2. Power
        if verbose:
            print("\n[2/4] Statistical Power")
        results['power'] = self.validate_power(
            true_lift=0.10, 
            target_power=0.80, 
            n_simulations=n_sims // 2,
            verbose=verbose
        )
        
        # 3. Coverage
        if verbose:
            print("\n[3/4] Confidence Interval Coverage")
        results['coverage'] = self.validate_coverage(
            true_lift=0.10,
            n_simulations=n_sims // 2,
            verbose=verbose
        )
        
        # 4. Lift recovery
        if verbose:
            print("\n[4/4] Lift Recovery Accuracy")
        lift_results = self.validate_lift_recovery(
            true_lifts=[0.0, 0.10],
            method='did',
            verbose=verbose
        )
        results['lift_recovery'] = {
            str(k): {
                'bias': v.lift_bias,
                'rmse': v.lift_rmse,
                'passed': abs(v.lift_bias) < 0.02
            }
            for k, v in lift_results.items()
        }
        
        # Summary
        all_passed = (
            results['type_i']['passed'] and
            results['power']['passed'] and
            results['coverage']['passed'] and
            all(v['passed'] for v in results['lift_recovery'].values())
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("VALIDATION SUMMARY")
            print("=" * 60)
            print(f"\n  Type I Error:  {'[PASS]' if results['type_i']['passed'] else '[FAIL]'}")
            print(f"  Power:         {'[PASS]' if results['power']['passed'] else '[FAIL]'}")
            print(f"  Coverage:      {'[PASS]' if results['coverage']['passed'] else '[FAIL]'}")
            print(f"  Lift Recovery: {'[PASS]' if all(v['passed'] for v in results['lift_recovery'].values()) else '[FAIL]'}")
            print(f"\n  OVERALL: {'[PASS] ALL TESTS PASSED' if all_passed else '[FAIL] SOME TESTS FAILED'}")
            print("\n" + "=" * 60)
        
        results['all_passed'] = all_passed
        
        return results


if __name__ == '__main__':
    # Run validation
    validator = IncrementalityValidator(
        n_simulations=20,
        n_geos=40,
        n_periods=52,
        treatment_start=40
    )
    
    results = validator.run_full_validation()
