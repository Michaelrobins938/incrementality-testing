"""
Geo Matching Module
===================

Implements synthetic control matching for geo-based incrementality testing.

Key Features:
- Pre-period matching with R² > 0.8 target
- Multiple matching algorithms (correlation, DTW, Mahalanobis)
- Treatment/control assignment with balance constraints
- Match quality diagnostics and visualization
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Literal
from scipy import stats
from scipy.spatial.distance import cdist, mahalanobis
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings


@dataclass
class MatchingResult:
    """Container for geo matching results."""
    treatment_geos: List[str]
    control_geos: List[str]
    match_pairs: Dict[str, str]  # treatment -> control mapping
    match_quality: Dict[str, float]  # per-pair R² scores
    overall_r2: float
    pre_period_correlation: float
    balance_metrics: Dict[str, float]
    diagnostics: Dict[str, any] = field(default_factory=dict)


class GeoMatcher:
    """
    Synthetic Control Matching for Geo-Based Experiments
    
    Implements multiple matching strategies to find optimal treatment/control
    geo assignments that maximize pre-period similarity.
    
    Parameters
    ----------
    min_r2 : float
        Minimum R² threshold for acceptable matches (default: 0.8)
    matching_method : str
        Matching algorithm: 'correlation', 'dtw', 'mahalanobis', 'optimal'
    n_pre_periods : int
        Number of pre-treatment periods to use for matching
    balance_features : list
        Additional features to balance (e.g., population, spend)
    """
    
    def __init__(
        self,
        min_r2: float = 0.8,
        matching_method: Literal['correlation', 'dtw', 'mahalanobis', 'optimal'] = 'optimal',
        n_pre_periods: int = 8,
        balance_features: Optional[List[str]] = None,
        random_state: int = 42
    ):
        self.min_r2 = min_r2
        self.matching_method = matching_method
        self.n_pre_periods = n_pre_periods
        self.balance_features = balance_features or []
        self.random_state = random_state
        self.scaler = StandardScaler()
        np.random.seed(random_state)
    
    def fit(
        self,
        geo_data: pd.DataFrame,
        geo_col: str = 'geo_id',
        time_col: str = 'date',
        metric_col: str = 'revenue',
        treatment_fraction: float = 0.5
    ) -> MatchingResult:
        """
        Find optimal treatment/control geo assignments.
        
        Parameters
        ----------
        geo_data : pd.DataFrame
            Panel data with geo, time, and metric columns
        geo_col : str
            Column name for geo identifier
        time_col : str
            Column name for time period
        metric_col : str
            Column name for outcome metric
        treatment_fraction : float
            Fraction of geos to assign to treatment (default: 0.5)
            
        Returns
        -------
        MatchingResult
            Container with treatment/control assignments and quality metrics
        """
        # Pivot to geo x time matrix
        pivot_data = geo_data.pivot(
            index=geo_col, 
            columns=time_col, 
            values=metric_col
        )
        
        # Use last n_pre_periods for matching
        if pivot_data.shape[1] > self.n_pre_periods:
            pre_period_data = pivot_data.iloc[:, -self.n_pre_periods:]
        else:
            pre_period_data = pivot_data
        
        # Normalize for matching
        normalized = self.scaler.fit_transform(pre_period_data.T).T
        normalized_df = pd.DataFrame(
            normalized, 
            index=pre_period_data.index,
            columns=pre_period_data.columns
        )
        
        # Calculate distance matrix based on method
        distance_matrix = self._calculate_distances(normalized_df)
        
        # Find optimal pairs using Hungarian algorithm
        n_geos = len(normalized_df)
        n_treatment = int(n_geos * treatment_fraction)
        
        # Get optimal assignment
        treatment_geos, control_geos, match_pairs = self._optimal_assignment(
            distance_matrix, 
            normalized_df.index.tolist(),
            n_treatment
        )
        
        # Calculate match quality
        match_quality = {}
        for t_geo, c_geo in match_pairs.items():
            t_series = pre_period_data.loc[t_geo].values
            c_series = pre_period_data.loc[c_geo].values
            match_quality[f"{t_geo}->{c_geo}"] = r2_score(t_series, c_series)
        
        # Overall metrics
        treatment_series = pre_period_data.loc[treatment_geos].mean(axis=0)
        control_series = pre_period_data.loc[control_geos].mean(axis=0)
        
        overall_r2 = r2_score(treatment_series, control_series)
        correlation = np.corrcoef(treatment_series, control_series)[0, 1]
        
        # Balance metrics
        balance_metrics = self._calculate_balance(
            geo_data, geo_col, treatment_geos, control_geos
        )
        
        # Diagnostics
        diagnostics = {
            'distance_matrix': distance_matrix,
            'pre_period_data': pre_period_data,
            'treatment_series': treatment_series,
            'control_series': control_series,
            'matching_method': self.matching_method
        }
        
        return MatchingResult(
            treatment_geos=treatment_geos,
            control_geos=control_geos,
            match_pairs=match_pairs,
            match_quality=match_quality,
            overall_r2=overall_r2,
            pre_period_correlation=correlation,
            balance_metrics=balance_metrics,
            diagnostics=diagnostics
        )
    
    def _calculate_distances(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate pairwise distances between geos."""
        
        if self.matching_method == 'correlation':
            # 1 - correlation as distance
            corr_matrix = np.corrcoef(data.values)
            return 1 - corr_matrix
        
        elif self.matching_method == 'dtw':
            # Dynamic Time Warping distance
            return self._dtw_distance_matrix(data.values)
        
        elif self.matching_method == 'mahalanobis':
            # Mahalanobis distance
            cov = np.cov(data.values.T)
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov)
            
            n = len(data)
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist[i, j] = mahalanobis(
                        data.values[i], 
                        data.values[j], 
                        cov_inv
                    )
            return dist
        
        else:  # 'optimal' - combination
            # Euclidean distance on normalized data
            return cdist(data.values, data.values, metric='euclidean')
    
    def _dtw_distance_matrix(self, data: np.ndarray) -> np.ndarray:
        """Calculate DTW distance matrix."""
        n = len(data)
        dist = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = self._dtw(data[i], data[j])
                dist[i, j] = d
                dist[j, i] = d
        
        return dist
    
    def _dtw(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Dynamic Time Warping distance between two series."""
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i-1] - s2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )
        
        return dtw_matrix[n, m]
    
    def _optimal_assignment(
        self,
        distance_matrix: np.ndarray,
        geo_ids: List[str],
        n_treatment: int
    ) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Find optimal treatment/control assignment using modified Hungarian algorithm.
        """
        n_geos = len(geo_ids)
        
        # For pair matching, we need equal groups
        n_pairs = min(n_treatment, n_geos - n_treatment)
        
        # Use Hungarian algorithm on distance matrix
        # First, randomly split into two groups, then optimize pairs
        indices = np.arange(n_geos)
        np.random.shuffle(indices)
        
        group1 = indices[:n_geos // 2]
        group2 = indices[n_geos // 2:]
        
        # Create submatrix of distances between groups
        sub_dist = distance_matrix[np.ix_(group1, group2)]
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(sub_dist)
        
        # Assign treatment to group1, control to group2
        treatment_geos = [geo_ids[group1[i]] for i in row_ind[:n_pairs]]
        control_geos = [geo_ids[group2[j]] for j in col_ind[:n_pairs]]
        
        match_pairs = {t: c for t, c in zip(treatment_geos, control_geos)}
        
        return treatment_geos, control_geos, match_pairs
    
    def _calculate_balance(
        self,
        data: pd.DataFrame,
        geo_col: str,
        treatment_geos: List[str],
        control_geos: List[str]
    ) -> Dict[str, float]:
        """Calculate balance metrics between treatment and control groups."""
        
        balance = {}
        
        # Group by geo and calculate aggregates
        geo_agg = data.groupby(geo_col).agg({
            col: 'mean' for col in data.select_dtypes(include=[np.number]).columns
        }).reset_index()
        
        treatment_data = geo_agg[geo_agg[geo_col].isin(treatment_geos)]
        control_data = geo_agg[geo_agg[geo_col].isin(control_geos)]
        
        for col in geo_agg.select_dtypes(include=[np.number]).columns:
            if col == geo_col:
                continue
            
            t_mean = treatment_data[col].mean()
            c_mean = control_data[col].mean()
            pooled_std = np.sqrt(
                (treatment_data[col].var() + control_data[col].var()) / 2
            )
            
            if pooled_std > 0:
                # Standardized mean difference
                smd = (t_mean - c_mean) / pooled_std
            else:
                smd = 0
            
            balance[f'{col}_smd'] = smd
            balance[f'{col}_ratio'] = t_mean / c_mean if c_mean != 0 else np.nan
        
        return balance
    
    def validate_match_quality(self, result: MatchingResult) -> Dict[str, any]:
        """
        Validate match quality with statistical tests.
        
        Returns
        -------
        dict
            Validation results including tests and recommendations
        """
        validation = {
            'r2_threshold_met': result.overall_r2 >= self.min_r2,
            'overall_r2': result.overall_r2,
            'correlation': result.pre_period_correlation,
            'n_treatment': len(result.treatment_geos),
            'n_control': len(result.control_geos),
            'tests': {}
        }
        
        # Check individual pair quality
        poor_matches = [
            pair for pair, r2 in result.match_quality.items() 
            if r2 < self.min_r2
        ]
        validation['poor_match_pairs'] = poor_matches
        validation['poor_match_rate'] = len(poor_matches) / len(result.match_quality)
        
        # Pre-period parallel trends test
        pre_data = result.diagnostics.get('pre_period_data')
        if pre_data is not None:
            t_series = result.diagnostics['treatment_series']
            c_series = result.diagnostics['control_series']
            
            # Test for parallel trends (ratio should be constant)
            ratios = t_series / c_series
            ratio_cv = ratios.std() / ratios.mean() if ratios.mean() != 0 else np.inf
            
            validation['tests']['parallel_trends'] = {
                'ratio_cv': ratio_cv,
                'passed': ratio_cv < 0.1  # Less than 10% variation
            }
            
            # Granger causality proxy - check for anticipation effects
            # (last pre-period should not diverge)
            last_ratio = ratios.iloc[-1]
            mean_ratio = ratios.iloc[:-1].mean()
            anticipation = abs(last_ratio - mean_ratio) / mean_ratio if mean_ratio != 0 else 0
            
            validation['tests']['no_anticipation'] = {
                'divergence': anticipation,
                'passed': anticipation < 0.05  # Less than 5% divergence
            }
        
        # Recommendations
        recommendations = []
        if not validation['r2_threshold_met']:
            recommendations.append(
                f"R² ({result.overall_r2:.3f}) below threshold ({self.min_r2}). "
                "Consider: more pre-periods, different matching method, or larger geo pool."
            )
        if validation['poor_match_rate'] > 0.2:
            recommendations.append(
                f"{validation['poor_match_rate']*100:.0f}% of pairs have poor R². "
                "Consider dropping poorly matched geos."
            )
        
        validation['recommendations'] = recommendations
        validation['ready_for_test'] = (
            validation['r2_threshold_met'] and 
            validation['poor_match_rate'] <= 0.2
        )
        
        return validation


def create_synthetic_geo_data(
    n_geos: int = 50,
    n_periods: int = 52,
    base_revenue: float = 10000,
    seasonality_strength: float = 0.3,
    noise_level: float = 0.1,
    geo_heterogeneity: float = 0.5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic geo-level panel data for testing.
    
    Parameters
    ----------
    n_geos : int
        Number of geographic regions (DMAs)
    n_periods : int
        Number of time periods (weeks)
    base_revenue : float
        Base weekly revenue per geo
    seasonality_strength : float
        Amplitude of seasonal pattern (0-1)
    noise_level : float
        Random noise level (0-1)
    geo_heterogeneity : float
        Variation in base revenue across geos (0-1)
        
    Returns
    -------
    pd.DataFrame
        Panel data with geo_id, date, revenue columns
    """
    np.random.seed(random_state)
    
    records = []
    
    for geo in range(n_geos):
        # Geo-specific baseline
        geo_multiplier = 1 + geo_heterogeneity * np.random.randn()
        geo_multiplier = max(0.3, min(2.0, geo_multiplier))  # Clip
        
        # Geo-specific seasonality phase
        phase_shift = np.random.uniform(0, np.pi / 4)
        
        for period in range(n_periods):
            # Seasonal component (annual cycle)
            seasonal = seasonality_strength * np.sin(
                2 * np.pi * period / 52 + phase_shift
            )
            
            # Trend component (slight growth)
            trend = 0.001 * period
            
            # Weekly pattern (weekday effect)
            weekly = 0.05 * np.sin(2 * np.pi * period / 4)
            
            # Random noise
            noise = noise_level * np.random.randn()
            
            # Combine components
            revenue = base_revenue * geo_multiplier * (
                1 + seasonal + trend + weekly + noise
            )
            
            records.append({
                'geo_id': f'DMA_{geo:03d}',
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(weeks=period),
                'period': period,
                'revenue': max(0, revenue),
                'population': int(100000 * geo_multiplier * (1 + 0.1 * np.random.randn())),
                'historical_spend': base_revenue * geo_multiplier * 0.1
            })
    
    return pd.DataFrame(records)


if __name__ == '__main__':
    # Demo
    print("=" * 60)
    print("GEO MATCHER DEMONSTRATION")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic geo data (50 DMAs, 52 weeks)...")
    data = create_synthetic_geo_data(n_geos=50, n_periods=52)
    print(f"   Data shape: {data.shape}")
    print(f"   Geos: {data['geo_id'].nunique()}")
    print(f"   Periods: {data['period'].nunique()}")
    
    # Run matching
    print("\n2. Running geo matching (optimal method, R² > 0.8)...")
    matcher = GeoMatcher(min_r2=0.8, matching_method='optimal')
    result = matcher.fit(data)
    
    print(f"\n3. Results:")
    print(f"   Treatment geos: {len(result.treatment_geos)}")
    print(f"   Control geos: {len(result.control_geos)}")
    print(f"   Overall R²: {result.overall_r2:.4f}")
    print(f"   Pre-period correlation: {result.pre_period_correlation:.4f}")
    
    # Validate
    print("\n4. Validation:")
    validation = matcher.validate_match_quality(result)
    print(f"   R² threshold met: {validation['r2_threshold_met']}")
    print(f"   Poor match rate: {validation['poor_match_rate']*100:.1f}%")
    print(f"   Ready for test: {validation['ready_for_test']}")
    
    if validation['recommendations']:
        print("\n   Recommendations:")
        for rec in validation['recommendations']:
            print(f"   - {rec}")
    
    print("\n" + "=" * 60)
    print("MATCHING COMPLETE")
    print("=" * 60)
