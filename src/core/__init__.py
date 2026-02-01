# Incrementality Testing Core Module
"""
Geo-Based Incrementality Framework

Core components:
- GeoMatcher: Synthetic control matching for treatment/control assignment
- CausalImpact: Difference-in-differences with Bayesian structural time series
- PowerAnalyzer: Geo-level power calculation and test duration planning
- IncrementalityRunner: Unified interface for running incrementality tests
"""

from .geo_matcher import GeoMatcher, MatchingResult
from .causal_impact import CausalImpactAnalyzer, CausalImpactResult
from .power_analyzer import GeoPowerAnalyzer, PowerResult
from .synthetic_control import SyntheticControlMethod
from .incrementality_runner import IncrementalityRunner, ExperimentConfig

__all__ = [
    'GeoMatcher',
    'MatchingResult',
    'CausalImpactAnalyzer', 
    'CausalImpactResult',
    'GeoPowerAnalyzer',
    'PowerResult',
    'SyntheticControlMethod',
    'IncrementalityRunner',
    'ExperimentConfig'
]

__version__ = '1.0.0'
