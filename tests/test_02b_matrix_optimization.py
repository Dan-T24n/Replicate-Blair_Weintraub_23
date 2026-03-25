"""
test_02b_matrix_optimization.py

Test suite for Level 2 matrix optimization in 02b_precompute_ri_coefficients.py

Tests validate:
1. Design matrix construction
2. Single simulation regression
3. Batch processing performance
4. Numerical equivalence with formula-based approach
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config import PATHS

# Import functions from 02b script (using module name without .py)
import importlib.util
spec = importlib.util.spec_from_file_location("precompute_module", Path(__file__).parent.parent / 'src' / '02b_precompute_ri_coefficients.py')
precompute_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precompute_module)

# Import functions
load_column_data = precompute_module.load_column_data
build_design_matrix_optimized = precompute_module.build_design_matrix_optimized
run_single_simulation_matrix_optimized = precompute_module.run_single_simulation_matrix_optimized
process_simulation_batch_optimized = precompute_module.process_simulation_batch_optimized
run_optimized_regression_matrix = precompute_module.run_optimized_regression_matrix

# Import formula-based approach for comparison
import statsmodels.formula.api as smf

class TestMatrixOptimization:
    
    def test_load_column_data(self):
        """Test data loading for all columns"""
        for col_num in [1, 2, 3, 4, 5]:
            df, outcome_var, control_vars, cluster_var = load_column_data(col_num)
            
            # Basic validation
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert outcome_var in df.columns
            assert 'treatment' in df.columns
            assert 'barrio_code' in df.columns
            assert 'iweight' in df.columns
            assert 'manzana' in df.columns
            
            # Column-specific validation
            if col_num in [1, 2]:
                assert cluster_var is None  # Admin columns
            else:
                assert cluster_var == 'manzana_code'  # Survey columns
                assert cluster_var in df.columns
            
            print(f"✓ Column {col_num}: {len(df)} obs, outcome={outcome_var}")
    
    def test_design_matrix_construction(self):
        """Test design matrix construction"""
        for col_num in [1, 3]:  # Test admin and survey
            df, outcome_var, control_vars, cluster_var = load_column_data(col_num)
            
            X_matrix, y_vector, weights_vector, manzana_index, df_clean = build_design_matrix_optimized(
                df, outcome_var, control_vars, cluster_var
            )
            
            # Dimension validation
            assert X_matrix.shape[0] == len(y_vector) == len(weights_vector)
            assert X_matrix.shape[0] == len(df_clean)
            assert X_matrix.shape[1] >= 2  # At least intercept + treatment/spillover
            
            # Content validation
            assert np.all(X_matrix[:, 0] == 1), "First column should be intercept"
            assert np.all(X_matrix[:, -2] == 0), "Treatment placeholder should be zeros"
            assert np.all(X_matrix[:, -1] == 0), "Spillover placeholder should be zeros"
            
            # Index validation
            assert len(manzana_index) == 1255
            valid_mappings = np.sum(manzana_index >= 0)
            assert valid_mappings > 0
            assert valid_mappings <= len(df_clean)
            
            print(f"✓ Column {col_num}: Matrix {X_matrix.shape}, {valid_mappings} valid mappings")
    
    def test_single_simulation_regression(self):
        """Test single simulation processing"""
        # Test with Column 1 (admin data)
        col_num = 1
        df, outcome_var, control_vars, cluster_var = load_column_data(col_num)
        X_matrix, y_vector, weights_vector, manzana_index, df_clean = build_design_matrix_optimized(
            df, outcome_var, control_vars, cluster_var
        )
        cluster_groups = df_clean[cluster_var].values if cluster_var else None
        
        # Load simulation data
        sim_file = PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv'
        assert sim_file.exists(), f"Simulation file not found: {sim_file}"
        
        df_sim_file = pd.read_csv(sim_file)
        sim_cols = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')]
        assert len(sim_cols) > 0, "No simulation columns found"
        
        # Test first simulation
        sim_col = sim_cols[0]
        treat_coef, spill_coef = run_single_simulation_matrix_optimized(
            sim_col, X_matrix, y_vector, weights_vector, 
            df_sim_file, cluster_var, manzana_index, cluster_groups
        )
        
        # Validate results
        assert isinstance(treat_coef, (int, float))
        assert isinstance(spill_coef, (int, float))
        assert not (treat_coef == -10 and spill_coef == -10), "Regression should not fail"
        
        print(f"✓ Single simulation: treat={treat_coef:.6f}, spill={spill_coef:.6f}")
    
    def test_batch_processing_performance(self):
        """Test batch processing with performance measurement"""
        col_num = 1
        df, outcome_var, control_vars, cluster_var = load_column_data(col_num)
        X_matrix, y_vector, weights_vector, manzana_index, df_clean = build_design_matrix_optimized(
            df, outcome_var, control_vars, cluster_var
        )
        cluster_groups = df_clean[cluster_var].values if cluster_var else None
        
        # Load simulation data
        sim_file = PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv'
        df_sim_file = pd.read_csv(sim_file)
        sim_cols = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')][:100]  # Test 100
        
        # Measure performance
        start_time = time.time()
        batch_coefficients = process_simulation_batch_optimized(
            X_matrix, y_vector, weights_vector, df_sim_file, sim_cols, 
            cluster_var, manzana_index, cluster_groups
        )
        elapsed_time = time.time() - start_time
        
        # Validate results
        assert len(batch_coefficients) == len(sim_cols)
        successful = sum(1 for c in batch_coefficients if c[0] != -10)
        success_rate = successful / len(batch_coefficients)
        rate = len(sim_cols) / elapsed_time
        
        assert success_rate > 0.95, f"Success rate too low: {success_rate:.2%}"
        assert rate > 10, f"Processing rate too slow: {rate:.1f} sims/sec"
        
        print(f"✓ Batch processing: {successful}/{len(sim_cols)} successful ({success_rate:.1%})")
        print(f"✓ Performance: {rate:.1f} simulations/second")
    
    def test_numerical_equivalence_formula_vs_matrix(self):
        """Test numerical equivalence between formula and matrix approaches"""
        col_num = 1  # Test with admin data (simpler case)
        
        # Load data
        df, outcome_var, control_vars, cluster_var = load_column_data(col_num)
        
        # Matrix approach
        X_matrix, y_vector, weights_vector, manzana_index, df_clean = build_design_matrix_optimized(
            df, outcome_var, control_vars, cluster_var
        )
        cluster_groups = df_clean[cluster_var].values if cluster_var else None
        
        # Load one simulation
        sim_file = PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv'
        df_sim_file = pd.read_csv(sim_file)
        sim_col = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')][0]
        
        # Matrix-based result
        treat_matrix, spill_matrix = run_single_simulation_matrix_optimized(
            sim_col, X_matrix, y_vector, weights_vector, 
            df_sim_file, cluster_var, manzana_index, cluster_groups
        )
        
        # Formula-based result (reference implementation)
        treatment_assignment = df_sim_file[sim_col].values
        sim_treatment_data = pd.DataFrame({
            'manzana': df_sim_file['manzana'],
            'treatment_ri': treatment_assignment
        })
        
        df_merged = df.merge(sim_treatment_data, on='manzana', how='inner')
        
        # Build formula (matching original approach)
        formula = f"{outcome_var} ~ C(treatment_ri, Treatment(reference=0))"
        formula += " + C(barrio_code)"
        for var in control_vars:
            if var in df_merged.columns:
                formula += f" + {var}"
        
        # Drop missing values
        essential_vars = [outcome_var, 'treatment_ri', 'barrio_code', 'iweight']
        df_formula_clean = df_merged.dropna(subset=essential_vars)
        
        # Run formula regression
        model = smf.wls(formula, data=df_formula_clean, weights=df_formula_clean['iweight'])
        results = model.fit()
        
        treat_formula = results.params.get('C(treatment_ri, Treatment(reference=0))[T.1]', 0.0)
        spill_formula = results.params.get('C(treatment_ri, Treatment(reference=0))[T.2]', 0.0)
        
        # Compare results
        treat_diff = abs(treat_matrix - treat_formula)
        spill_diff = abs(spill_matrix - spill_formula)
        
        tolerance = 1e-6
        
        print(f"Matrix approach:  treat={treat_matrix:.8f}, spill={spill_matrix:.8f}")
        print(f"Formula approach: treat={treat_formula:.8f}, spill={spill_formula:.8f}")
        print(f"Differences:      treat={treat_diff:.8f}, spill={spill_diff:.8f}")
        
        assert treat_diff < tolerance, f"Treatment coefficient difference too large: {treat_diff}"
        assert spill_diff < tolerance, f"Spillover coefficient difference too large: {spill_diff}"
        
        print("✓ Numerical equivalence confirmed (differences < 1e-6)")
    
    def test_memory_efficiency(self):
        """Test memory usage is reasonable"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            print("⚠️ psutil not available, skipping memory test")
            return
        
        # Process one column
        col_num = 1
        df, outcome_var, control_vars, cluster_var = load_column_data(col_num)
        X_matrix, y_vector, weights_vector, manzana_index, df_clean = build_design_matrix_optimized(
            df, outcome_var, control_vars, cluster_var
        )
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - initial_memory
        
        # Should use less than 100MB for design matrix
        assert memory_usage < 100, f"Memory usage too high: {memory_usage:.1f}MB"
        
        print(f"✓ Memory usage: {memory_usage:.1f}MB (within limits)")

def run_tests():
    """Run all tests"""
    test_suite = TestMatrixOptimization()
    
    print("="*60)
    print("LEVEL 2 MATRIX OPTIMIZATION TEST SUITE")
    print("="*60)
    
    tests = [
        ("Data Loading", test_suite.test_load_column_data),
        ("Design Matrix Construction", test_suite.test_design_matrix_construction),
        ("Single Simulation Regression", test_suite.test_single_simulation_regression),
        ("Batch Processing Performance", test_suite.test_batch_processing_performance),
        ("Numerical Equivalence", test_suite.test_numerical_equivalence_formula_vs_matrix),
        ("Memory Efficiency", test_suite.test_memory_efficiency),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"TEST: {test_name}")
        print(f"{'='*50}")
        
        try:
            test_func()
            print(f"🎉 {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Matrix optimization is ready.")
    else:
        print("❌ Some tests failed. Please fix issues before proceeding.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
