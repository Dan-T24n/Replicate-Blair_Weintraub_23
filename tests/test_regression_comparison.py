#!/usr/bin/env python3
"""
Compare our matrix regression with statsmodels regression to identify coefficient differences
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from config import PATHS
import importlib.util
import statsmodels.formula.api as smf

# Import our optimization module
spec = importlib.util.spec_from_file_location('precompute', 'src/02b_precompute_ri_coefficients.py')
precompute = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precompute)

def test_single_simulation_comparison():
    """Compare our method vs statsmodels for a single simulation"""
    
    # Load survey data (Column 3)
    df, outcome_var, control_vars, cluster_var = precompute.load_column_data(3)
    
    # Build our design matrix
    X_matrix, y_vector, weights_vector, manzana_mapping_info, df_clean = precompute.build_design_matrix_optimized(
        df, outcome_var, control_vars, cluster_var
    )
    cluster_groups = df_clean[cluster_var].values
    
    # Load simulation data
    sim_file = PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv'
    df_sim_file = pd.read_csv(sim_file)
    
    # Test with first simulation
    sim_col = 'treatment_ri_1'
    
    print("🔍 REGRESSION COMPARISON TEST")
    print("=" * 50)
    print(f"Testing Column 3 (Survey) with {sim_col}")
    print(f"Data shape: {df_clean.shape}")
    print(f"Outcome variable: {outcome_var}")
    
    # Method 1: Our optimized matrix method
    print(f"\n📊 METHOD 1: Our Matrix Method")
    print("-" * 30)
    
    # Create vectorized alignment
    sim_manzanas = df_sim_file['manzana'].values
    manzana_to_row = manzana_mapping_info['manzana_to_row']
    sim_to_data_indices = np.array([manzana_to_row.get(manzana, -1) for manzana in sim_manzanas])
    valid_mask = sim_to_data_indices >= 0
    
    # Run our method
    our_result = precompute.run_single_simulation_vectorized(
        sim_col, X_matrix, y_vector, weights_vector, 
        df_sim_file, cluster_var, cluster_groups,
        sim_to_data_indices, valid_mask
    )
    
    print(f"Our treatment coef: {our_result[0]:.6f}")
    print(f"Our spillover coef: {our_result[1]:.6f}")
    
    # Method 2: Statsmodels method (like original)
    print(f"\n📊 METHOD 2: Statsmodels Method")
    print("-" * 30)
    
    # Create treatment assignment for this simulation
    treatment_assignment = df_sim_file[sim_col].values
    sim_manzanas = df_sim_file['manzana'].values
    
    # Create treatment mapping
    treatment_map = dict(zip(sim_manzanas, treatment_assignment))
    
    # Apply to survey data
    df_with_sim = df_clean.copy()
    df_with_sim['treatment_sim'] = df_with_sim['manzana_code'].map(treatment_map).fillna(0)
    
    # Build formula
    control_formula = ' + '.join(control_vars)
    formula = f"{outcome_var} ~ C(treatment_sim) + C(barrio_code) + {control_formula}"
    
    # Run statsmodels regression
    model = smf.wls(formula, data=df_with_sim, weights=df_with_sim['iweight'])
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df_with_sim['manzana_code']})
    
    # Extract coefficients
    sm_treatment = results.params.get('C(treatment_sim)[T.1.0]', np.nan)
    sm_spillover = results.params.get('C(treatment_sim)[T.2.0]', np.nan)
    
    print(f"Statsmodels treatment coef: {sm_treatment:.6f}")
    print(f"Statsmodels spillover coef: {sm_spillover:.6f}")
    
    # Comparison
    print(f"\n🔍 COMPARISON")
    print("-" * 30)
    treatment_diff = abs(our_result[0] - sm_treatment)
    spillover_diff = abs(our_result[1] - sm_spillover)
    
    print(f"Treatment difference: {treatment_diff:.8f}")
    print(f"Spillover difference: {spillover_diff:.8f}")
    
    if treatment_diff < 1e-6 and spillover_diff < 1e-6:
        print("✅ Methods match within tolerance")
    else:
        print("❌ Methods differ significantly")
        
        # Debug information
        print(f"\n🔧 DEBUG INFO:")
        print(f"Treatment assignment distribution:")
        unique, counts = np.unique(df_with_sim['treatment_sim'], return_counts=True)
        for val, count in zip(unique, counts):
            print(f"  {val}: {count}")
            
        print(f"\nDesign matrix last 2 columns (first 10 rows):")
        print("Treatment | Spillover")
        for i in range(min(10, len(X_matrix))):
            print(f"{X_matrix[i, -2]:.0f}       | {X_matrix[i, -1]:.0f}")

if __name__ == "__main__":
    test_single_simulation_comparison()
