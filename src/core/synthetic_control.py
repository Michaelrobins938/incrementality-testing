"""
Synthetic Control Method
========================

Implements Abadie et al. (2010) synthetic control method for 
constructing counterfactual control groups from donor pool.

Key Features:
- Optimal weight selection for donor units
- Pre-period fit optimization
- Placebo tests for inference
- RMSPE-based significance testing
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize, Bounds, LinearConstraint
from sklearn.metrics import mean_squared_error
import warnings


@dataclass
class SyntheticControlResult:
    """Container for synthetic control results."""
    treatment_unit: str
    donor_weights: Dict[str, float]
    synthetic_series: np.ndarray
    actual_series: np.ndarray
    time_index: np.ndarray
    treatment_period: int
    pre_period_rmspe: float
    post_period_rmspe: float
    treatment_effect: np.ndarray
    cumulative_effect: float
    average_effect: float
    placebo_results: Optional[Dict] = None


class SyntheticControlMethod:
    """
    Synthetic Control Method for Causal Inference
    
    Creates an optimal weighted combination of control units to 
    construct a synthetic counterfactual for the treated unit.
    
    Parameters
    ----------
    n_pre_periods : int
        Number of pre-treatment periods for fitting
    optimization_method : str
        Optimization algorithm: 'SLSQP', 'trust-constr', 'L-BFGS-B'
    max_donors : int
        Maximum number of donor units with non-zero weights
    regularization : float
        L2 regularization to prevent overfitting (default: 0.01)
    """
    
    def __init__(
        self,
        n_pre_periods: Optional[int] = None,
        optimization_method: str = 'SLSQP',
        max_donors: Optional[int] = None,
        regularization: float = 0.01,
        random_state: int = 42
    ):
        self.n_pre_periods = n_pre_periods
        self.optimization_method = optimization_method
        self.max_donors = max_donors
        self.regularization = regularization
        self.random_state = random_state
        np.random.seed(random_state)
    
    def fit(
        self,
        panel_data: pd.DataFrame,
        treatment_unit: str,
        treatment_period: int,
        unit_col: str = 'geo_id',
        time_col: str = 'period',
        outcome_col: str = 'revenue',
        covariates: Optional[List[str]] = None
    ) -> SyntheticControlResult:
        """
        Fit synthetic control model.
        
        Parameters
        ----------
        panel_data : pd.DataFrame
            Panel data with unit, time, and outcome columns
        treatment_unit : str
            Identifier of the treated unit
        treatment_period : int
            Time period when treatment begins
        unit_col : str
            Column name for unit identifier
        time_col : str
            Column name for time period
        outcome_col : str
            Column name for outcome variable
        covariates : list
            Additional covariates for matching
            
        Returns
        -------
        SyntheticControlResult
            Results including weights, synthetic series, and effects
        """
        # Pivot data to wide format
        pivot = panel_data.pivot(
            index=unit_col,
            columns=time_col,
            values=outcome_col
        )
        
        # Separate treatment and donor units
        treatment_series = pivot.loc[treatment_unit].values
        donor_units = [u for u in pivot.index if u != treatment_unit]
        donor_matrix = pivot.loc[donor_units].values
        
        time_index = pivot.columns.values
        
        # Determine pre-period
        pre_mask = time_index < treatment_period
        if self.n_pre_periods is not None:
            n_pre = min(self.n_pre_periods, pre_mask.sum())
            pre_indices = np.where(pre_mask)[0][-n_pre:]
            pre_mask = np.zeros(len(time_index), dtype=bool)
            pre_mask[pre_indices] = True
        
        # Extract pre-period data
        y_pre = treatment_series[pre_mask]
        X_pre = donor_matrix[:, pre_mask].T  # (n_pre, n_donors)
        
        # Optimize weights
        n_donors = len(donor_units)
        weights = self._optimize_weights(y_pre, X_pre, n_donors)
        
        # Create synthetic control series
        synthetic_series = donor_matrix.T @ weights
        
        # Calculate RMSPE
        pre_period_rmspe = np.sqrt(mean_squared_error(
            treatment_series[pre_mask],
            synthetic_series[pre_mask]
        ))
        
        post_mask = time_index >= treatment_period
        if post_mask.sum() > 0:
            post_period_rmspe = np.sqrt(mean_squared_error(
                treatment_series[post_mask],
                synthetic_series[post_mask]
            ))
        else:
            post_period_rmspe = np.nan
        
        # Calculate treatment effects
        treatment_effect = treatment_series - synthetic_series
        post_effects = treatment_effect[post_mask]
        cumulative_effect = post_effects.sum() if len(post_effects) > 0 else 0
        average_effect = post_effects.mean() if len(post_effects) > 0 else 0
        
        # Create weight dictionary
        donor_weights = {
            unit: float(w) for unit, w in zip(donor_units, weights)
            if w > 0.01  # Only include non-trivial weights
        }
        
        return SyntheticControlResult(
            treatment_unit=treatment_unit,
            donor_weights=donor_weights,
            synthetic_series=synthetic_series,
            actual_series=treatment_series,
            time_index=time_index,
            treatment_period=treatment_period,
            pre_period_rmspe=pre_period_rmspe,
            post_period_rmspe=post_period_rmspe,
            treatment_effect=treatment_effect,
            cumulative_effect=cumulative_effect,
            average_effect=average_effect
        )
    
    def _optimize_weights(
        self,
        y: np.ndarray,
        X: np.ndarray,
        n_donors: int
    ) -> np.ndarray:
        """
        Optimize donor weights to minimize pre-period MSE.
        
        Constraints:
        - Weights sum to 1
        - Weights are non-negative
        - Optional L2 regularization
        """
        
        def objective(w):
            synthetic = X @ w
            mse = np.mean((y - synthetic) ** 2)
            reg = self.regularization * np.sum(w ** 2)
            return mse + reg
        
        # Initial weights (uniform)
        w0 = np.ones(n_donors) / n_donors
        
        # Constraints
        bounds = Bounds(lb=0, ub=1)
        sum_constraint = LinearConstraint(
            np.ones(n_donors),
            lb=1.0,
            ub=1.0
        )
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method=self.optimization_method,
            bounds=bounds,
            constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        weights = result.x
        
        # Apply sparsity if max_donors specified
        if self.max_donors is not None and self.max_donors < n_donors:
            # Keep only top weights
            top_indices = np.argsort(weights)[-self.max_donors:]
            sparse_weights = np.zeros(n_donors)
            sparse_weights[top_indices] = weights[top_indices]
            sparse_weights /= sparse_weights.sum()  # Renormalize
            weights = sparse_weights
        
        return weights
    
    def placebo_test(
        self,
        panel_data: pd.DataFrame,
        treatment_unit: str,
        treatment_period: int,
        unit_col: str = 'geo_id',
        time_col: str = 'period',
        outcome_col: str = 'revenue',
        n_placebos: Optional[int] = None
    ) -> Dict:
        """
        Run placebo tests for inference.
        
        Applies the synthetic control method to each donor unit
        to generate a distribution of placebo effects.
        
        Parameters
        ----------
        panel_data, treatment_unit, treatment_period, etc.
            Same as fit()
        n_placebos : int
            Number of placebo tests (default: all donors)
            
        Returns
        -------
        dict
            Placebo test results including p-value
        """
        # First, get the actual treatment effect
        actual_result = self.fit(
            panel_data, treatment_unit, treatment_period,
            unit_col, time_col, outcome_col
        )
        
        # Get all units
        all_units = panel_data[unit_col].unique()
        donor_units = [u for u in all_units if u != treatment_unit]
        
        if n_placebos is not None:
            donor_units = np.random.choice(
                donor_units, 
                size=min(n_placebos, len(donor_units)),
                replace=False
            ).tolist()
        
        # Run placebo for each donor
        placebo_effects = []
        placebo_rmspe_ratios = []
        
        for placebo_unit in donor_units:
            try:
                placebo_result = self.fit(
                    panel_data, placebo_unit, treatment_period,
                    unit_col, time_col, outcome_col
                )
                
                # RMSPE ratio (post/pre)
                if placebo_result.pre_period_rmspe > 0:
                    rmspe_ratio = (
                        placebo_result.post_period_rmspe / 
                        placebo_result.pre_period_rmspe
                    )
                else:
                    rmspe_ratio = np.inf
                
                placebo_effects.append(placebo_result.average_effect)
                placebo_rmspe_ratios.append(rmspe_ratio)
                
            except Exception as e:
                # Skip failed placebos
                continue
        
        # Calculate p-value
        actual_effect = actual_result.average_effect
        placebo_effects = np.array(placebo_effects)
        
        # Two-sided p-value
        if len(placebo_effects) > 0:
            p_value = np.mean(np.abs(placebo_effects) >= np.abs(actual_effect))
        else:
            p_value = np.nan
        
        # RMSPE-based p-value
        actual_rmspe_ratio = (
            actual_result.post_period_rmspe / actual_result.pre_period_rmspe
            if actual_result.pre_period_rmspe > 0 else np.inf
        )
        
        if len(placebo_rmspe_ratios) > 0:
            rmspe_p_value = np.mean(
                np.array(placebo_rmspe_ratios) >= actual_rmspe_ratio
            )
        else:
            rmspe_p_value = np.nan
        
        return {
            'actual_effect': actual_effect,
            'placebo_effects': placebo_effects.tolist(),
            'p_value': p_value,
            'rmspe_ratio': actual_rmspe_ratio,
            'placebo_rmspe_ratios': placebo_rmspe_ratios,
            'rmspe_p_value': rmspe_p_value,
            'n_placebos': len(placebo_effects),
            'significant_at_10': p_value < 0.10 if not np.isnan(p_value) else False,
            'significant_at_05': p_value < 0.05 if not np.isnan(p_value) else False
        }
    
    def in_time_placebo(
        self,
        panel_data: pd.DataFrame,
        treatment_unit: str,
        actual_treatment_period: int,
        placebo_period: int,
        unit_col: str = 'geo_id',
        time_col: str = 'period',
        outcome_col: str = 'revenue'
    ) -> Dict:
        """
        Run in-time placebo test.
        
        Tests for pre-treatment effects by applying the method
        to an earlier time period.
        
        Parameters
        ----------
        placebo_period : int
            Earlier time period to test (should be before actual treatment)
            
        Returns
        -------
        dict
            In-time placebo results
        """
        # Filter to pre-treatment period only
        pre_data = panel_data[
            panel_data[time_col] < actual_treatment_period
        ].copy()
        
        # Run synthetic control on placebo period
        placebo_result = self.fit(
            pre_data, treatment_unit, placebo_period,
            unit_col, time_col, outcome_col
        )
        
        # Effect should be ~0 for valid design
        return {
            'placebo_period': placebo_period,
            'actual_treatment_period': actual_treatment_period,
            'placebo_effect': placebo_result.average_effect,
            'pre_rmspe': placebo_result.pre_period_rmspe,
            'post_rmspe': placebo_result.post_period_rmspe,
            'valid_design': abs(placebo_result.average_effect) < 0.05 * np.mean(placebo_result.actual_series)
        }


if __name__ == '__main__':
    from geo_matcher import create_synthetic_geo_data
    
    print("=" * 60)
    print("SYNTHETIC CONTROL METHOD DEMONSTRATION")
    print("=" * 60)
    
    # Generate data
    print("\n1. Generating synthetic panel data...")
    data = create_synthetic_geo_data(n_geos=20, n_periods=30)
    
    # Add treatment effect to one geo
    treatment_unit = 'DMA_005'
    treatment_period = 20
    lift = 0.15  # 15% lift
    
    mask = (data['geo_id'] == treatment_unit) & (data['period'] >= treatment_period)
    data.loc[mask, 'revenue'] *= (1 + lift)
    
    print(f"   Treatment unit: {treatment_unit}")
    print(f"   Treatment period: {treatment_period}")
    print(f"   True lift: {lift*100:.0f}%")
    
    # Fit synthetic control
    print("\n2. Fitting synthetic control model...")
    sc = SyntheticControlMethod(regularization=0.001)
    result = sc.fit(data, treatment_unit, treatment_period)
    
    print(f"\n3. Results:")
    print(f"   Pre-period RMSPE: {result.pre_period_rmspe:.2f}")
    print(f"   Post-period RMSPE: {result.post_period_rmspe:.2f}")
    print(f"   Average treatment effect: {result.average_effect:.2f}")
    print(f"   Cumulative effect: {result.cumulative_effect:.2f}")
    
    # Calculate lift
    baseline = np.mean(result.actual_series[:treatment_period])
    estimated_lift = result.average_effect / baseline
    print(f"   Estimated lift: {estimated_lift*100:.1f}%")
    print(f"   True lift: {lift*100:.0f}%")
    print(f"   Recovery error: {abs(estimated_lift - lift)*100:.1f}pp")
    
    # Top donors
    print(f"\n4. Top donor weights:")
    sorted_donors = sorted(
        result.donor_weights.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    for unit, weight in sorted_donors:
        print(f"   {unit}: {weight:.3f}")
    
    # Placebo test
    print("\n5. Running placebo tests (10 placebos)...")
    placebo = sc.placebo_test(
        data, treatment_unit, treatment_period,
        n_placebos=10
    )
    print(f"   P-value: {placebo['p_value']:.3f}")
    print(f"   RMSPE p-value: {placebo['rmspe_p_value']:.3f}")
    print(f"   Significant at 5%: {placebo['significant_at_05']}")
    
    print("\n" + "=" * 60)
    print("SYNTHETIC CONTROL COMPLETE")
    print("=" * 60)
