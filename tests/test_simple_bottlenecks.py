"""
test_simple_bottlenecks.py

Simple diagnostic tests to identify performance bottlenecks without multiprocessing issues
"""

import time
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config import PATHS

# Import functions
import importlib.util
spec = importlib.util.spec_from_file_location("precompute_module", Path(__file__).parent.parent / 'src' / '02b_precompute_ri_coefficients.py')
precompute_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precompute_module)

def test_sequential_performance():
    """Test sequential performance without multiprocessing"""
    print("="*60)
    print("SEQUENTIAL PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Setup
    col_num = 1
    df, outcome_var, control_vars, cluster_var = precompute_module.load_column_data(col_num)
    
    print("\n1. DESIGN MATRIX CONSTRUCTION")
    print("-" * 40)
    start_time = time.time()
    X_matrix, y_vector, weights_vector, manzana_mapping_info, df_clean = precompute_module.build_design_matrix_optimized(
        df, outcome_var, control_vars, cluster_var
    )
    design_time = time.time() - start_time
    print(f"Design matrix time: {design_time:.4f}s")
    print(f"Matrix shape: {X_matrix.shape}")
    
    print("\n2. SIMULATION DATA LOADING")
    print("-" * 40)
    start_time = time.time()
    sim_file = PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv'
    df_sim_file = pd.read_csv(sim_file)
    sim_cols = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')]
    load_time = time.time() - start_time
    print(f"File load time: {load_time:.4f}s")
    print(f"File size: {sim_file.stat().st_size / 1024 / 1024:.1f}MB")
    print(f"Simulations: {len(sim_cols)}")
    
    cluster_groups = df_clean[cluster_var].values if cluster_var else None
    
    print("\n3. SINGLE SIMULATION BREAKDOWN")
    print("-" * 40)
    
    # Test first simulation with detailed timing
    sim_col = sim_cols[0]
    
    # 3a. Data extraction
    start_time = time.time()
    treatment_assignment = df_sim_file[sim_col].values
    sim_manzanas = df_sim_file['manzana'].values
    extraction_time = time.time() - start_time
    
    # 3b. Data alignment
    start_time = time.time()
    manzana_to_row = manzana_mapping_info['manzana_to_row']
    aligned_treatment = np.zeros(len(y_vector))
    for sim_idx, sim_manzana in enumerate(sim_manzanas):
        if sim_manzana in manzana_to_row:
            data_row = manzana_to_row[sim_manzana]
            aligned_treatment[data_row] = treatment_assignment[sim_idx]
    alignment_time = time.time() - start_time
    
    # 3c. Matrix operations
    start_time = time.time()
    X_sim = X_matrix.copy()
    X_sim[:, -2] = (aligned_treatment == 1).astype(float)
    X_sim[:, -1] = (aligned_treatment == 2).astype(float)
    matrix_time = time.time() - start_time
    
    # 3d. Regression
    start_time = time.time()
    treat_coef, spill_coef = precompute_module.run_optimized_regression_matrix(
        y_vector, X_sim, weights_vector, cluster_var, cluster_groups
    )
    regression_time = time.time() - start_time
    
    single_sim_total = extraction_time + alignment_time + matrix_time + regression_time
    
    print(f"  Data extraction: {extraction_time:.4f}s ({extraction_time/single_sim_total*100:.1f}%)")
    print(f"  Data alignment: {alignment_time:.4f}s ({alignment_time/single_sim_total*100:.1f}%)")
    print(f"  Matrix operations: {matrix_time:.4f}s ({matrix_time/single_sim_total*100:.1f}%)")
    print(f"  Regression: {regression_time:.4f}s ({regression_time/single_sim_total*100:.1f}%)")
    print(f"  Total: {single_sim_total:.4f}s")
    print(f"  Rate: {1/single_sim_total:.1f} simulations/second")
    
    print("\n4. BATCH SEQUENTIAL PROCESSING")
    print("-" * 40)
    
    # Test different batch sizes sequentially
    test_sizes = [10, 50, 100, 500]
    
    for batch_size in test_sizes:
        if batch_size > len(sim_cols):
            continue
            
        test_sim_cols = sim_cols[:batch_size]
        
        start_time = time.time()
        results = []
        for sim_col in test_sim_cols:
            result = precompute_module.run_single_simulation_matrix_optimized(
                sim_col, X_matrix, y_vector, weights_vector, 
                df_sim_file, cluster_var, manzana_mapping_info, cluster_groups
            )
            results.append(result)
        
        batch_time = time.time() - start_time
        batch_rate = len(test_sim_cols) / batch_time
        
        successful = sum(1 for r in results if r[0] != -10)
        
        print(f"  Batch {batch_size:3d}: {batch_time:.3f}s ({batch_rate:.1f} sims/sec, {successful}/{len(results)} successful)")
    
    return {
        'design_time': design_time,
        'load_time': load_time,
        'single_sim_time': single_sim_total,
        'single_sim_rate': 1/single_sim_total,
        'extraction_time': extraction_time,
        'alignment_time': alignment_time,
        'matrix_time': matrix_time,
        'regression_time': regression_time
    }

def test_regression_optimization():
    """Test regression optimization potential"""
    print("\n5. REGRESSION OPTIMIZATION ANALYSIS")
    print("-" * 40)
    
    # Setup single regression
    col_num = 1
    df, outcome_var, control_vars, cluster_var = precompute_module.load_column_data(col_num)
    X_matrix, y_vector, weights_vector, manzana_mapping_info, df_clean = precompute_module.build_design_matrix_optimized(
        df, outcome_var, control_vars, cluster_var
    )
    
    sim_file = PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv'
    df_sim_file = pd.read_csv(sim_file)
    sim_col = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')][0]
    
    # Prepare data
    treatment_assignment = df_sim_file[sim_col].values
    sim_manzanas = df_sim_file['manzana'].values
    manzana_to_row = manzana_mapping_info['manzana_to_row']
    
    aligned_treatment = np.zeros(len(y_vector))
    for sim_idx, sim_manzana in enumerate(sim_manzanas):
        if sim_manzana in manzana_to_row:
            data_row = manzana_to_row[sim_manzana]
            aligned_treatment[data_row] = treatment_assignment[sim_idx]
    
    X_sim = X_matrix.copy()
    X_sim[:, -2] = (aligned_treatment == 1).astype(float)
    X_sim[:, -1] = (aligned_treatment == 2).astype(float)
    
    # Test different regression approaches
    import statsmodels.api as sm
    from scipy import linalg
    
    n_tests = 1000  # Run many times for accurate timing
    
    # Current approach
    start_time = time.time()
    for _ in range(n_tests):
        model = sm.WLS(y_vector, X_sim, weights=weights_vector)
        results = model.fit()
        treat_coef = results.params[-2]
        spill_coef = results.params[-1]
    current_time = (time.time() - start_time) / n_tests
    
    print(f"Current (statsmodels WLS): {current_time:.6f}s per regression")
    print(f"Coefficients: treat={treat_coef:.6f}, spill={spill_coef:.6f}")
    
    # Direct matrix calculation (only for non-clustered)
    if cluster_var is None:
        try:
            start_time = time.time()
            for _ in range(n_tests):
                # Weighted least squares: (X'WX)^-1 X'Wy
                W = np.diag(weights_vector)
                XtWX = X_sim.T @ W @ X_sim
                XtWy = X_sim.T @ W @ y_vector
                coeffs = linalg.solve(XtWX, XtWy)
                treat_coef_direct = coeffs[-2]
                spill_coef_direct = coeffs[-1]
            direct_time = (time.time() - start_time) / n_tests
            
            print(f"Direct matrix calculation: {direct_time:.6f}s per regression")
            print(f"Direct coefficients: treat={treat_coef_direct:.6f}, spill={spill_coef_direct:.6f}")
            
            # Check accuracy
            treat_diff = abs(treat_coef - treat_coef_direct)
            spill_diff = abs(spill_coef - spill_coef_direct)
            print(f"Coefficient differences: treat={treat_diff:.8f}, spill={spill_diff:.8f}")
            
            speedup = current_time / direct_time
            print(f"Potential speedup: {speedup:.1f}x")
            
            return {
                'current_time': current_time,
                'direct_time': direct_time,
                'speedup': speedup,
                'accurate': treat_diff < 1e-6 and spill_diff < 1e-6
            }
        except Exception as e:
            print(f"Direct calculation failed: {e}")
            return {'current_time': current_time}
    else:
        print("Clustering required - direct calculation not applicable")
        return {'current_time': current_time}

def test_data_alignment_optimization():
    """Test data alignment optimization"""
    print("\n6. DATA ALIGNMENT OPTIMIZATION")
    print("-" * 40)
    
    # Setup
    col_num = 1
    df, outcome_var, control_vars, cluster_var = precompute_module.load_column_data(col_num)
    X_matrix, y_vector, weights_vector, manzana_mapping_info, df_clean = precompute_module.build_design_matrix_optimized(
        df, outcome_var, control_vars, cluster_var
    )
    
    sim_file = PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv'
    df_sim_file = pd.read_csv(sim_file)
    sim_cols = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')][:100]
    
    manzana_to_row = manzana_mapping_info['manzana_to_row']
    sim_manzanas = df_sim_file['manzana'].values
    
    n_tests = 100
    
    # Current approach (loop-based)
    start_time = time.time()
    for _ in range(n_tests):
        for sim_col in sim_cols[:10]:  # Test with 10 simulations
            treatment_assignment = df_sim_file[sim_col].values
            aligned_treatment = np.zeros(len(y_vector))
            for sim_idx, sim_manzana in enumerate(sim_manzanas):
                if sim_manzana in manzana_to_row:
                    data_row = manzana_to_row[sim_manzana]
                    aligned_treatment[data_row] = treatment_assignment[sim_idx]
    current_alignment_time = (time.time() - start_time) / n_tests / 10
    
    # Optimized approach (vectorized)
    # Pre-compute index mapping once
    sim_to_data_indices = np.array([manzana_to_row.get(manzana, -1) for manzana in sim_manzanas])
    valid_mask = sim_to_data_indices >= 0
    
    start_time = time.time()
    for _ in range(n_tests):
        for sim_col in sim_cols[:10]:
            treatment_assignment = df_sim_file[sim_col].values
            aligned_treatment = np.zeros(len(y_vector))
            # Vectorized assignment
            valid_indices = sim_to_data_indices[valid_mask]
            valid_treatments = treatment_assignment[valid_mask]
            aligned_treatment[valid_indices] = valid_treatments
    optimized_alignment_time = (time.time() - start_time) / n_tests / 10
    
    print(f"Current alignment: {current_alignment_time:.6f}s per simulation")
    print(f"Optimized alignment: {optimized_alignment_time:.6f}s per simulation")
    print(f"Alignment speedup: {current_alignment_time/optimized_alignment_time:.1f}x")
    
    return {
        'current_alignment_time': current_alignment_time,
        'optimized_alignment_time': optimized_alignment_time,
        'alignment_speedup': current_alignment_time/optimized_alignment_time
    }

def run_analysis():
    """Run comprehensive analysis"""
    print("PERFORMANCE BOTTLENECK DIAGNOSIS")
    print("="*80)
    
    sequential_results = test_sequential_performance()
    regression_results = test_regression_optimization()
    alignment_results = test_data_alignment_optimization()
    
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    
    print(f"\n1. CURRENT PERFORMANCE:")
    print(f"   Single simulation: {sequential_results['single_sim_time']:.4f}s")
    print(f"   Rate: {sequential_results['single_sim_rate']:.1f} simulations/second")
    print(f"   Projected full time: {500000/sequential_results['single_sim_rate']/3600:.1f} hours")
    
    print(f"\n2. BOTTLENECK BREAKDOWN:")
    total_time = sequential_results['single_sim_time']
    components = [
        ('Data extraction', sequential_results['extraction_time']),
        ('Data alignment', sequential_results['alignment_time']),
        ('Matrix operations', sequential_results['matrix_time']),
        ('Regression', sequential_results['regression_time'])
    ]
    
    for name, time_val in components:
        percentage = time_val / total_time * 100
        print(f"   {name}: {time_val:.4f}s ({percentage:.1f}%)")
    
    print(f"\n3. OPTIMIZATION POTENTIAL:")
    
    # Alignment optimization
    if 'alignment_speedup' in alignment_results:
        alignment_savings = sequential_results['alignment_time'] * (1 - 1/alignment_results['alignment_speedup'])
        print(f"   Alignment optimization: {alignment_results['alignment_speedup']:.1f}x speedup ({alignment_savings:.4f}s savings)")
    
    # Regression optimization
    if 'speedup' in regression_results:
        regression_savings = sequential_results['regression_time'] * (1 - 1/regression_results['speedup'])
        print(f"   Regression optimization: {regression_results['speedup']:.1f}x speedup ({regression_savings:.4f}s savings)")
    
    # Calculate optimized performance
    optimized_time = sequential_results['single_sim_time']
    if 'alignment_speedup' in alignment_results:
        optimized_time -= sequential_results['alignment_time'] * (1 - 1/alignment_results['alignment_speedup'])
    if 'speedup' in regression_results:
        optimized_time -= sequential_results['regression_time'] * (1 - 1/regression_results['speedup'])
    
    optimized_rate = 1 / optimized_time
    total_speedup = sequential_results['single_sim_rate'] / optimized_rate
    
    print(f"\n4. PROJECTED OPTIMIZED PERFORMANCE:")
    print(f"   Optimized time per simulation: {optimized_time:.4f}s")
    print(f"   Optimized rate: {optimized_rate:.1f} simulations/second")
    print(f"   Overall speedup: {optimized_rate/sequential_results['single_sim_rate']:.1f}x")
    print(f"   Full processing time: {500000/optimized_rate/60:.1f} minutes")
    
    target_met = (500000/optimized_rate/60) <= 2
    print(f"   Target (1-2 min): {'✅ ACHIEVABLE' if target_met else '❌ STILL TOO SLOW'}")

if __name__ == "__main__":
    run_analysis()
