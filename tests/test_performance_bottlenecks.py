"""
test_performance_bottlenecks.py

Diagnostic tests to identify performance bottlenecks in Level 2 matrix optimization
"""

import time
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import multiprocessing as mp
from functools import partial

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config import PATHS

# Import functions
import importlib.util
spec = importlib.util.spec_from_file_location("precompute_module", Path(__file__).parent.parent / 'src' / '02b_precompute_ri_coefficients.py')
precompute_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precompute_module)

def test_individual_components():
    """Test individual components to identify bottlenecks"""
    print("="*60)
    print("PERFORMANCE BOTTLENECK ANALYSIS")
    print("="*60)
    
    # Setup data
    col_num = 1
    df, outcome_var, control_vars, cluster_var = precompute_module.load_column_data(col_num)
    
    print("\n1. DESIGN MATRIX CONSTRUCTION")
    print("-" * 40)
    start_time = time.time()
    X_matrix, y_vector, weights_vector, manzana_mapping_info, df_clean = precompute_module.build_design_matrix_optimized(
        df, outcome_var, control_vars, cluster_var
    )
    design_matrix_time = time.time() - start_time
    print(f"Time to build design matrix: {design_matrix_time:.3f}s")
    print(f"Matrix shape: {X_matrix.shape}")
    
    # Load simulation data
    sim_file = PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv'
    df_sim_file = pd.read_csv(sim_file)
    sim_cols = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')]
    cluster_groups = df_clean[cluster_var].values if cluster_var else None
    
    print(f"\n2. SIMULATION DATA LOADING")
    print("-" * 40)
    start_time = time.time()
    df_sim_file = pd.read_csv(sim_file)
    sim_cols = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')]
    load_time = time.time() - start_time
    print(f"Time to load simulation file: {load_time:.3f}s")
    print(f"File size: {sim_file.stat().st_size / 1024 / 1024:.1f}MB")
    print(f"Simulations: {len(sim_cols)}")
    
    print(f"\n3. SINGLE SIMULATION PROCESSING")
    print("-" * 40)
    
    # Test single simulation with timing breakdown
    sim_col = sim_cols[0]
    
    # 3a. Data alignment
    start_time = time.time()
    treatment_assignment = df_sim_file[sim_col].values
    sim_manzanas = df_sim_file['manzana'].values
    manzana_to_row = manzana_mapping_info['manzana_to_row']
    
    aligned_treatment = np.zeros(len(y_vector))
    for sim_idx, sim_manzana in enumerate(sim_manzanas):
        if sim_manzana in manzana_to_row:
            data_row = manzana_to_row[sim_manzana]
            aligned_treatment[data_row] = treatment_assignment[sim_idx]
    alignment_time = time.time() - start_time
    print(f"  3a. Data alignment: {alignment_time:.3f}s")
    
    # 3b. Matrix operations
    start_time = time.time()
    X_sim = X_matrix.copy()
    X_sim[:, -2] = (aligned_treatment == 1).astype(float)
    X_sim[:, -1] = (aligned_treatment == 2).astype(float)
    matrix_ops_time = time.time() - start_time
    print(f"  3b. Matrix operations: {matrix_ops_time:.3f}s")
    
    # 3c. Regression
    start_time = time.time()
    treat_coef, spill_coef = precompute_module.run_optimized_regression_matrix(
        y_vector, X_sim, weights_vector, cluster_var, cluster_groups
    )
    regression_time = time.time() - start_time
    print(f"  3c. Regression: {regression_time:.3f}s")
    
    total_single_sim = alignment_time + matrix_ops_time + regression_time
    print(f"  Total single simulation: {total_single_sim:.3f}s")
    print(f"  Rate: {1/total_single_sim:.1f} simulations/second")
    
    return {
        'design_matrix_time': design_matrix_time,
        'load_time': load_time,
        'alignment_time': alignment_time,
        'matrix_ops_time': matrix_ops_time,
        'regression_time': regression_time,
        'total_single_sim': total_single_sim,
        'single_sim_rate': 1/total_single_sim
    }

def test_multiprocessing_overhead():
    """Test multiprocessing overhead vs sequential processing"""
    print(f"\n4. MULTIPROCESSING vs SEQUENTIAL")
    print("-" * 40)
    
    # Setup
    col_num = 1
    df, outcome_var, control_vars, cluster_var = precompute_module.load_column_data(col_num)
    X_matrix, y_vector, weights_vector, manzana_mapping_info, df_clean = precompute_module.build_design_matrix_optimized(
        df, outcome_var, control_vars, cluster_var
    )
    cluster_groups = df_clean[cluster_var].values if cluster_var else None
    
    sim_file = PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv'
    df_sim_file = pd.read_csv(sim_file)
    sim_cols = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')][:50]  # Test with 50
    
    # Sequential processing
    print(f"Testing with {len(sim_cols)} simulations...")
    
    start_time = time.time()
    sequential_results = []
    for sim_col in sim_cols:
        result = precompute_module.run_single_simulation_matrix_optimized(
            sim_col, X_matrix, y_vector, weights_vector, 
            df_sim_file, cluster_var, manzana_mapping_info, cluster_groups
        )
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    sequential_rate = len(sim_cols) / sequential_time
    
    print(f"Sequential: {sequential_time:.3f}s ({sequential_rate:.1f} sims/sec)")
    
    # Multiprocessing
    start_time = time.time()
    mp_results = precompute_module.process_simulation_batch_optimized(
        X_matrix, y_vector, weights_vector, df_sim_file, sim_cols, 
        cluster_var, manzana_mapping_info, cluster_groups
    )
    mp_time = time.time() - start_time
    mp_rate = len(sim_cols) / mp_time
    
    print(f"Multiprocessing: {mp_time:.3f}s ({mp_rate:.1f} sims/sec)")
    
    overhead = mp_time - sequential_time
    efficiency = sequential_rate / mp_rate
    
    print(f"Overhead: {overhead:.3f}s")
    print(f"Efficiency ratio: {efficiency:.2f} (>1 means MP is slower)")
    
    return {
        'sequential_time': sequential_time,
        'sequential_rate': sequential_rate,
        'mp_time': mp_time,
        'mp_rate': mp_rate,
        'overhead': overhead,
        'efficiency_ratio': efficiency
    }

def test_batch_size_optimization():
    """Test different batch sizes to find optimal processing size"""
    print(f"\n5. BATCH SIZE OPTIMIZATION")
    print("-" * 40)
    
    # Setup
    col_num = 1
    df, outcome_var, control_vars, cluster_var = precompute_module.load_column_data(col_num)
    X_matrix, y_vector, weights_vector, manzana_mapping_info, df_clean = precompute_module.build_design_matrix_optimized(
        df, outcome_var, control_vars, cluster_var
    )
    cluster_groups = df_clean[cluster_var].values if cluster_var else None
    
    sim_file = PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv'
    df_sim_file = pd.read_csv(sim_file)
    all_sim_cols = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')]
    
    batch_sizes = [10, 25, 50, 100, 200]
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(all_sim_cols):
            continue
            
        sim_cols = all_sim_cols[:batch_size]
        
        start_time = time.time()
        batch_results = precompute_module.process_simulation_batch_optimized(
            X_matrix, y_vector, weights_vector, df_sim_file, sim_cols, 
            cluster_var, manzana_mapping_info, cluster_groups
        )
        elapsed_time = time.time() - start_time
        rate = len(sim_cols) / elapsed_time
        
        results[batch_size] = {
            'time': elapsed_time,
            'rate': rate
        }
        
        print(f"Batch size {batch_size:3d}: {elapsed_time:.3f}s ({rate:.1f} sims/sec)")
    
    # Find optimal batch size
    optimal_batch = max(results.keys(), key=lambda x: results[x]['rate'])
    print(f"Optimal batch size: {optimal_batch} ({results[optimal_batch]['rate']:.1f} sims/sec)")
    
    return results

def test_regression_alternatives():
    """Test alternative regression approaches"""
    print(f"\n6. REGRESSION ALTERNATIVES")
    print("-" * 40)
    
    # Setup single simulation data
    col_num = 1
    df, outcome_var, control_vars, cluster_var = precompute_module.load_column_data(col_num)
    X_matrix, y_vector, weights_vector, manzana_mapping_info, df_clean = precompute_module.build_design_matrix_optimized(
        df, outcome_var, control_vars, cluster_var
    )
    
    sim_file = PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv'
    df_sim_file = pd.read_csv(sim_file)
    sim_col = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')][0]
    
    # Prepare aligned data
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
    
    # 1. Current approach (statsmodels WLS)
    start_time = time.time()
    for _ in range(100):  # Run 100 times for better timing
        model = sm.WLS(y_vector, X_sim, weights=weights_vector)
        results = model.fit()
        treat_coef = results.params[-2]
        spill_coef = results.params[-1]
    current_time = (time.time() - start_time) / 100
    print(f"Current (statsmodels WLS): {current_time:.4f}s per regression")
    
    # 2. Direct matrix calculation (if no clustering)
    if cluster_var is None:
        start_time = time.time()
        for _ in range(100):
            # Weighted least squares: (X'WX)^-1 X'Wy
            W = np.diag(weights_vector)
            XtWX = X_sim.T @ W @ X_sim
            XtWy = X_sim.T @ W @ y_vector
            coeffs = linalg.solve(XtWX, XtWy)
            treat_coef_direct = coeffs[-2]
            spill_coef_direct = coeffs[-1]
        direct_time = (time.time() - start_time) / 100
        
        print(f"Direct matrix calculation: {direct_time:.4f}s per regression")
        print(f"Speedup potential: {current_time/direct_time:.1f}x")
        
        # Verify results are similar
        diff_treat = abs(treat_coef - treat_coef_direct)
        diff_spill = abs(spill_coef - spill_coef_direct)
        print(f"Coefficient differences: treat={diff_treat:.6f}, spill={diff_spill:.6f}")
        
        return {
            'current_time': current_time,
            'direct_time': direct_time,
            'speedup_potential': current_time/direct_time
        }
    else:
        print("Clustering required - direct calculation not applicable")
        return {'current_time': current_time}

def run_comprehensive_analysis():
    """Run all diagnostic tests"""
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Run all tests
    component_results = test_individual_components()
    mp_results = test_multiprocessing_overhead()
    batch_results = test_batch_size_optimization()
    regression_results = test_regression_alternatives()
    
    print(f"\n" + "="*80)
    print("BOTTLENECK ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n1. SINGLE SIMULATION BREAKDOWN:")
    print(f"   Data alignment: {component_results['alignment_time']:.3f}s ({component_results['alignment_time']/component_results['total_single_sim']*100:.1f}%)")
    print(f"   Matrix operations: {component_results['matrix_ops_time']:.3f}s ({component_results['matrix_ops_time']/component_results['total_single_sim']*100:.1f}%)")
    print(f"   Regression: {component_results['regression_time']:.3f}s ({component_results['regression_time']/component_results['total_single_sim']*100:.1f}%)")
    print(f"   Total: {component_results['total_single_sim']:.3f}s")
    
    print(f"\n2. MULTIPROCESSING EFFICIENCY:")
    print(f"   Sequential rate: {mp_results['sequential_rate']:.1f} sims/sec")
    print(f"   Multiprocessing rate: {mp_results['mp_rate']:.1f} sims/sec")
    print(f"   Efficiency ratio: {mp_results['efficiency_ratio']:.2f} ({'INEFFICIENT' if mp_results['efficiency_ratio'] > 1 else 'EFFICIENT'})")
    print(f"   Overhead: {mp_results['overhead']:.3f}s")
    
    print(f"\n3. OPTIMIZATION RECOMMENDATIONS:")
    
    # Identify biggest bottleneck
    bottlenecks = [
        ('Data alignment', component_results['alignment_time']),
        ('Matrix operations', component_results['matrix_ops_time']),
        ('Regression', component_results['regression_time'])
    ]
    biggest_bottleneck = max(bottlenecks, key=lambda x: x[1])
    
    print(f"   Biggest bottleneck: {biggest_bottleneck[0]} ({biggest_bottleneck[1]:.3f}s)")
    
    if mp_results['efficiency_ratio'] > 1:
        print(f"   Multiprocessing is INEFFICIENT - consider sequential or larger batches")
    
    if 'speedup_potential' in regression_results:
        print(f"   Regression speedup potential: {regression_results['speedup_potential']:.1f}x with direct matrix calculation")
    
    # Project optimized performance
    current_rate = component_results['single_sim_rate']
    if 'speedup_potential' in regression_results:
        optimized_regression_time = component_results['regression_time'] / regression_results['speedup_potential']
        optimized_total = component_results['alignment_time'] + component_results['matrix_ops_time'] + optimized_regression_time
        optimized_rate = 1 / optimized_total
        
        print(f"\n4. OPTIMIZED PROJECTION:")
        print(f"   Current rate: {current_rate:.1f} sims/sec")
        print(f"   Optimized rate: {optimized_rate:.1f} sims/sec")
        print(f"   Improvement: {optimized_rate/current_rate:.1f}x")
        
        # Project full processing time
        total_sims = 500000
        optimized_total_time = total_sims / optimized_rate
        print(f"   Full processing time: {optimized_total_time/60:.1f} minutes")

if __name__ == "__main__":
    run_comprehensive_analysis()
