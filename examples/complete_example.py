"""
Complete Incrementality Testing Example
========================================

Demonstrates the full workflow with a realistic scenario:
- Facebook campaign incrementality test
- 50 DMAs, 8-week test
- Measuring true incremental lift vs attributed revenue

Run: python examples/complete_example.py
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.geo_matcher import GeoMatcher, create_synthetic_geo_data
from core.synthetic_control import SyntheticControlMethod
from core.causal_impact import CausalImpactAnalyzer, calculate_incremental_lift
from core.power_analyzer import GeoPowerAnalyzer, plan_geo_test
from core.incrementality_runner import IncrementalityRunner, ExperimentConfig


def example_1_power_analysis():
    """
    Example 1: Pre-Test Power Analysis
    
    Before running an incrementality test, determine:
    - How many geos you need
    - How long the test should run
    - What effect size you can detect
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: PRE-TEST POWER ANALYSIS")
    print("=" * 70)
    
    print("\nScenario: You're planning a Facebook campaign test with 50 DMAs")
    print("Question: What's the minimum lift you can detect with 80% power?")
    
    analyzer = GeoPowerAnalyzer(alpha=0.05, power=0.80)
    
    # Your test design
    result = analyzer.calculate_mde(
        n_geos=50,
        n_periods=8,  # 8 weeks
        baseline_mean=50000,  # $50K weekly revenue per DMA
        baseline_std=10000,   # $10K standard deviation
        icc=0.15              # 15% intra-class correlation
    )
    
    print(f"\nPOWER ANALYSIS RESULTS:")
    print(f"  ├─ Minimum Detectable Effect: {result.mde:.1%}")
    print(f"  ├─ Effect Size (Cohen's d): {result.effect_size:.3f}")
    print(f"  ├─ Design Effect: {result.design_effect:.2f}")
    print(f"  └─ Treatment/Control Split: {result.n_treatment}/{result.n_control}")
    
    print(f"\nINTERPRETATION:")
    print(f"  You can detect a {result.mde:.0%}+ lift with 80% probability.")
    print(f"  If your expected lift is 10%, you have sufficient power.")
    print(f"  If your expected lift is only 3%, consider more geos or longer test.")
    
    # What if we want to detect 5% lift?
    print(f"\nSCENARIO: Want to detect 5% lift")
    result_5pct = analyzer.calculate_required_geos(
        target_mde=0.05,
        n_periods=8,
        baseline_mean=50000,
        baseline_std=10000,
        icc=0.15
    )
    print(f"  Required geos: {result_5pct.required_geos}")
    
    # Or how many weeks with current geos?
    result_duration = analyzer.calculate_required_duration(
        target_mde=0.05,
        n_geos=50,
        baseline_mean=50000,
        baseline_std=10000,
        icc=0.15
    )
    print(f"  Or required weeks: {result_duration.required_duration}")


def example_2_geo_matching():
    """
    Example 2: Geo Matching for Treatment/Control Assignment
    
    Create matched treatment and control groups that:
    - Have similar pre-period trends (R² > 0.8)
    - Allow for valid parallel trends assumption
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: GEO MATCHING")
    print("=" * 70)
    
    print("\nScenario: Generate matched treatment/control geo pairs")
    
    # Generate some historical data (8 weeks pre-period)
    data = create_synthetic_geo_data(
        n_geos=50,
        n_periods=8,
        base_revenue=50000,
        seasonality_strength=0.2,
        noise_level=0.1,
        random_state=42
    )
    
    print(f"  ├─ Data: {data['geo_id'].nunique()} geos × {data['period'].nunique()} weeks")
    print(f"  └─ Total records: {len(data):,}")
    
    # Perform matching
    matcher = GeoMatcher(min_r2=0.8, matching_method='optimal')
    result = matcher.fit(data)
    
    print(f"\nMATCHING RESULTS:")
    print(f"  ├─ Overall R²: {result.overall_r2:.4f}")
    print(f"  ├─ Pre-period Correlation: {result.pre_period_correlation:.4f}")
    print(f"  ├─ Treatment Geos: {len(result.treatment_geos)}")
    print(f"  └─ Control Geos: {len(result.control_geos)}")
    
    # Validate
    validation = matcher.validate_match_quality(result)
    print(f"\nVALIDATION:")
    print(f"  ├─ R² Threshold Met: {validation['r2_threshold_met']}")
    print(f"  ├─ Poor Match Rate: {validation['poor_match_rate']:.1%}")
    print(f"  └─ Ready for Test: {validation['ready_for_test']}")
    
    # Show some matched pairs
    print(f"\nTOP 5 MATCHED PAIRS:")
    sorted_pairs = sorted(
        result.match_quality.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for pair, r2 in sorted_pairs:
        treatment, control = pair.split('->')
        print(f"  {treatment} ↔ {control}: R² = {r2:.3f}")


def example_3_causal_impact():
    """
    Example 3: Causal Impact Analysis with Known Ground Truth
    
    Simulates an experiment with known lift to verify methodology.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: CAUSAL IMPACT ANALYSIS")
    print("=" * 70)
    
    print("\nScenario: Measure incremental lift from a campaign with 12% true effect")
    
    # Generate data with treatment effect
    np.random.seed(42)
    n_geos = 50
    n_periods = 20
    treatment_start = 12
    true_lift = 0.12
    
    data = create_synthetic_geo_data(
        n_geos=n_geos,
        n_periods=n_periods,
        base_revenue=50000,
        random_state=42
    )
    
    # Apply treatment to half the geos
    treatment_geos = data['geo_id'].unique()[:n_geos // 2]
    mask = (
        data['geo_id'].isin(treatment_geos) & 
        (data['period'] >= treatment_start)
    )
    data.loc[mask, 'revenue'] *= (1 + true_lift)
    
    print(f"  ├─ True Lift: {true_lift:.0%}")
    print(f"  ├─ Treatment Geos: {len(treatment_geos)}")
    print(f"  ├─ Treatment Start: Week {treatment_start}")
    print(f"  └─ Test Duration: {n_periods - treatment_start} weeks")
    
    # Match geos using pre-period data
    matcher = GeoMatcher(min_r2=0.7)
    pre_data = data[data['period'] < treatment_start]
    match_result = matcher.fit(pre_data)
    
    # Aggregate to series
    t_data = data[data['geo_id'].isin(match_result.treatment_geos)]
    c_data = data[data['geo_id'].isin(match_result.control_geos)]
    
    t_series = t_data.groupby('period')['revenue'].sum().values
    c_series = c_data.groupby('period')['revenue'].sum().values
    
    # Analyze with DiD
    analyzer = CausalImpactAnalyzer(method='did', n_bootstrap=500)
    result = analyzer.analyze(t_series, c_series, treatment_start)
    
    print(f"\nCAUSAL IMPACT RESULTS:")
    print(f"  ├─ Estimated Lift: {result.relative_effect:.1%}")
    print(f"  ├─ True Lift: {true_lift:.0%}")
    print(f"  ├─ Estimation Error: {abs(result.relative_effect - true_lift)*100:.1f}pp")
    print(f"  ├─ 95% CI: [{result.effect_lower:.0f}, {result.effect_upper:.0f}]")
    print(f"  ├─ P-value: {result.p_value:.4f}")
    print(f"  └─ Significant: {result.significant}")
    
    print(f"\nDIAGNOSTICS:")
    print(f"  ├─ Pre-period MAPE: {result.pre_period_mape:.1f}%")
    print(f"  └─ Parallel Trends P-value: {result.parallel_trends_pvalue:.3f}")
    
    # Robustness checks
    robustness = analyzer.run_robustness_checks(result, t_series, c_series)
    print(f"\nROBUSTNESS:")
    print(f"  ├─ Leave-one-out Stable: {robustness['loo_stable']}")
    print(f"  ├─ Timing Stable: {robustness['timing_stable']}")
    print(f"  └─ Robustness Score: {robustness['robustness_score']*100:.0f}%")


def example_4_business_metrics():
    """
    Example 4: Business Impact Calculation
    
    Convert statistical results to business metrics (iROAS, incremental revenue).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: BUSINESS METRICS")
    print("=" * 70)
    
    print("\nScenario: Calculate incremental ROAS for $100K Facebook spend")
    
    # Generate experiment data
    np.random.seed(42)
    data = create_synthetic_geo_data(n_geos=50, n_periods=20, random_state=42)
    
    # Simulate 15% lift
    treatment_geos = data['geo_id'].unique()[:25]
    treatment_start = 12
    mask = (data['geo_id'].isin(treatment_geos)) & (data['period'] >= treatment_start)
    data.loc[mask, 'revenue'] *= 1.15
    
    # Match and analyze
    matcher = GeoMatcher()
    pre_data = data[data['period'] < treatment_start]
    match_result = matcher.fit(pre_data)
    
    t_data = data[data['geo_id'].isin(match_result.treatment_geos)]
    c_data = data[data['geo_id'].isin(match_result.control_geos)]
    
    t_series = t_data.groupby('period')['revenue'].sum().values
    c_series = c_data.groupby('period')['revenue'].sum().values
    
    analyzer = CausalImpactAnalyzer(method='did')
    result = analyzer.analyze(t_series, c_series, treatment_start)
    
    # Calculate business metrics
    total_spend = 100000  # $100K
    metrics = calculate_incremental_lift(result, total_spend)
    
    print(f"\nBUSINESS METRICS:")
    print(f"  ├─ Incremental Revenue: ${metrics['incremental_revenue']:,.0f}")
    print(f"  ├─ Incremental ROAS: ${metrics['incremental_roas']:.2f}")
    print(f"  ├─ iROAS 95% CI: [${metrics['iroas_lower']:.2f}, ${metrics['iroas_upper']:.2f}]")
    print(f"  ├─ Relative Lift: {metrics['relative_lift']:.1%}")
    print(f"  ├─ Total Spend: ${metrics['total_spend']:,.0f}")
    print(f"  └─ Cost per Incremental $: ${metrics['cost_per_incremental_dollar']:.2f}")
    
    print(f"\nINTERPRETATION:")
    print(f"  For every $1 spent on Facebook, you generated ${metrics['incremental_roas']:.2f}")
    print(f"  in truly incremental revenue (that wouldn't have happened otherwise).")
    
    if metrics['incremental_roas'] >= 1.0:
        print(f"  ✓ Campaign is profitable (iROAS ≥ $1.00)")
    else:
        print(f"  ✗ Campaign is NOT profitable (iROAS < $1.00)")


def example_5_full_workflow():
    """
    Example 5: Complete End-to-End Workflow
    
    Uses the IncrementalityRunner for the full analysis.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: COMPLETE WORKFLOW")
    print("=" * 70)
    
    print("\nScenario: Full incrementality test with automated workflow")
    
    # Generate realistic data
    np.random.seed(42)
    data = create_synthetic_geo_data(
        n_geos=50,
        n_periods=24,
        base_revenue=50000,
        seasonality_strength=0.25,
        noise_level=0.08,
        random_state=42
    )
    
    # Add treatment effect
    treatment_geos = data['geo_id'].unique()[:25]
    treatment_start = 16
    true_lift = 0.10
    
    mask = (data['geo_id'].isin(treatment_geos)) & (data['period'] >= treatment_start)
    data.loc[mask, 'revenue'] *= (1 + true_lift)
    
    # Configure experiment
    config = ExperimentConfig(
        name="Facebook_Q1_Incrementality",
        description="Q1 Facebook campaign incrementality test",
        method='did',
        alpha=0.05,
        target_power=0.80,
        expected_lift=0.10,
        total_spend=150000,
        pre_period_weeks=8,
        test_period_weeks=8
    )
    
    # Run full analysis
    runner = IncrementalityRunner(config, verbose=True)
    result = runner.run_full_analysis(data, treatment_start)
    
    # Summary
    print(f"\nFINAL RESULTS:")
    print(f"  ├─ Conclusion: {result.summary['conclusion']}")
    print(f"  ├─ Lift: {result.causal_result.relative_effect:.1%}")
    print(f"  ├─ True Lift: {true_lift:.0%}")
    print(f"  ├─ Error: {abs(result.causal_result.relative_effect - true_lift)*100:.1f}pp")
    print(f"  └─ Recommendation: {result.summary['recommendation']}")


def main():
    """Run all examples."""
    print("=" * 70)
    print("     GEO-BASED INCREMENTALITY TESTING FRAMEWORK")
    print("                  COMPLETE EXAMPLES")
    print("=" * 70)
    
    example_1_power_analysis()
    example_2_geo_matching()
    example_3_causal_impact()
    example_4_business_metrics()
    example_5_full_workflow()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)
    
    print("\nKey Takeaways:")
    print("  1. Always do power analysis BEFORE running a test")
    print("  2. Match quality (R² > 0.8) is critical for valid inference")
    print("  3. Use bootstrap CIs for robust uncertainty quantification")
    print("  4. Incremental ROAS often differs significantly from attributed ROAS")
    print("  5. Run robustness checks to validate your conclusions")


if __name__ == '__main__':
    main()
