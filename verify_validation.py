import sys
sys.path.insert(0, 'src')
from validation.validator import IncrementalityValidator
v = IncrementalityValidator(n_simulations=10, n_geos=40, n_periods=52, treatment_start=40)
res = v.run_full_validation()
print(f"\nFINAL_CHECK: {'SUCCESS' if res['all_passed'] else 'FAILURE'}")
