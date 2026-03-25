"""
02b_precompute_ri_coefficients.py

Blair & Weintraub (2023) Military Policing Replication
Pre-compute RI coefficients for all 100,000 simulations × 5 columns

Purpose: Generate coefficient cache files for instant table generation
- Mirrors original Stata logic (lines 400-746 in 03_Tables.do)
- Output: data/rand_new/coefs/RI_table1_col{n}_p{p}.csv files
- Level 2 Optimization: Matrix-based with batch-level multiprocessing

Authors: Robert Blair and Michael Weintraub (original)
Python implementation: 2024
"""

#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
import sys
import time
import argparse
from datetime import datetime
import multiprocessing as mp
from functools import partial

# Import project configuration
from config import PATHS, DATA_FILES, VALIDATION_TOLERANCE

#%%
def create_output_directories():
    """Create output directories for coefficient files"""
    coef_dir = PATHS['data_rand_new'] / 'coefs'
    coef_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created coefficient directory: {coef_dir}")
    return coef_dir

#%%
def load_column_data(col_num):
    """Load appropriate dataset and parameters for each column"""
    
    if col_num == 1:
        # Column 1: Crime during intervention (admin data)
        df = pd.read_csv(PATHS['data_raw'] / 'admin_data_during.csv')
        outcome_var = 'unw_crime2_num'
        geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
        blockdemovars = ['block_age', 'block_educ', 'block_pct_male']
        control_vars = geovars + blockdemovars + ['cum_all_unw_crime2_num']
        cluster_var = None
        
    elif col_num == 2:
        # Column 2: Crime after intervention (admin data)
        df = pd.read_csv(PATHS['data_raw'] / 'admin_data_after.csv')
        outcome_var = 'unw_crime_num'
        geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
        blockdemovars = ['block_age', 'block_educ', 'block_pct_male']
        control_vars = geovars + blockdemovars + ['cum_all_unw_crime_num']
        cluster_var = None
        
    elif col_num == 3:
        # Column 3: Crime victimization during intervention (survey)
        df = pd.read_csv(PATHS['data_raw'] / 'survey_endline.csv')
        outcome_var = 'i2_victimduringindex_std'
        demovars = ['age', 'gender', 'educ']
        geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
        control_vars = demovars + geovars
        cluster_var = 'manzana_code'
        
    elif col_num == 4:
        # Column 4: Crime victimization after intervention (survey)
        df = pd.read_csv(PATHS['data_raw'] / 'survey_endline.csv')
        outcome_var = 'i2_victimafterindex_std'
        demovars = ['age', 'gender', 'educ']
        geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
        control_vars = demovars + geovars
        cluster_var = 'manzana_code'
        
    elif col_num == 5:
        # Column 5: Crime witnessing after intervention (survey)
        df = pd.read_csv(PATHS['data_raw'] / 'survey_endline.csv')
        outcome_var = 'i_witnessindex_std'
        demovars = ['age', 'gender', 'educ']
        geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
        control_vars = demovars + geovars
        cluster_var = 'manzana_code'
    
    else:
        raise ValueError(f"Invalid column_num: {col_num}. Must be 1-5.")
    
    # Convert treatment to integer for consistency
    df['treatment'] = df['treatment'].astype(int)
    
    # Filter available control variables
    available_controls = [var for var in control_vars if var in df.columns]
    if len(available_controls) < len(control_vars):
        missing = set(control_vars) - set(available_controls)
        print(f"Warning: Missing control variables for column {col_num}: {missing}")
    
    return df, outcome_var, available_controls, cluster_var

#%%
def build_design_matrix_optimized(df, outcome_var, control_vars, cluster_var):
    """Pre-compute design matrix once per column (major performance optimization)"""
    
    print(f"    Building design matrix for {outcome_var}...")
    
    # Determine manzana column name (admin uses 'manzana', survey uses 'manzana_code')
    manzana_col = 'manzana_code' if cluster_var is not None else 'manzana'
    
    # Handle missing values (exact replication of current logic)
    essential_vars = [outcome_var, 'treatment', 'barrio_code', 'iweight', manzana_col]
    if cluster_var:
        essential_vars.append(cluster_var)
    
    # For survey data, drop missing for all variables to match clustering behavior
    if cluster_var is not None:
        all_vars = essential_vars + control_vars
        all_vars = [var for var in all_vars if var in df.columns]
        df_clean = df.dropna(subset=all_vars)
    else:
        # For admin data, also drop missing for control variables to avoid NaN in design matrix
        all_vars = essential_vars + control_vars
        all_vars = [var for var in all_vars if var in df.columns]
        df_clean = df.dropna(subset=all_vars)
    
    print(f"    Clean observations: {len(df_clean)} (dropped {len(df) - len(df_clean)})")
    
    # Create manzana index mapping for treatment assignment alignment
    # Simulation data uses sequential manzana 1-1255, but actual manzanas are large numbers
    # We need to map from simulation row index to data row index
    data_manzanas = df_clean[manzana_col].values
    
    # Create mapping from data manzana values to their row indices in df_clean
    manzana_to_row = {manzana: i for i, manzana in enumerate(data_manzanas)}
    
    # Pre-compute vectorized alignment indices for optimization
    # This will be computed once when we load simulation data
    manzana_mapping_info = {
        'data_manzanas': data_manzanas,
        'manzana_to_row': manzana_to_row,
        'manzana_col': manzana_col
    }
    
    # Build design matrix manually (much faster than patsy formula parsing)
    X_components = []
    
    # Add intercept
    X_components.append(np.ones((len(df_clean), 1)))
    
    # Add barrio fixed effects (dummy encoding with reference category)
    barrio_dummies = pd.get_dummies(df_clean['barrio_code'], prefix='barrio')
    X_components.append(barrio_dummies.values[:, :-1])  # Drop last for reference
    
    # Add control variables
    for var in control_vars:
        if var in df_clean.columns:
            X_components.append(df_clean[var].values.reshape(-1, 1))
    
    # Add placeholders for treatment and spillover (updated per simulation)
    X_components.append(np.zeros((len(df_clean), 1)))  # Treatment placeholder
    X_components.append(np.zeros((len(df_clean), 1)))  # Spillover placeholder
    
    # Combine all components into design matrix
    X_matrix = np.hstack(X_components)
    y_vector = df_clean[outcome_var].values
    weights_vector = df_clean['iweight'].values
    
    print(f"    Design matrix shape: {X_matrix.shape}")
    print(f"    Manzana mapping: {len(manzana_mapping_info['manzana_to_row'])} data manzanas")
    
    return X_matrix, y_vector, weights_vector, manzana_mapping_info, df_clean

#%%
def run_optimized_regression_matrix(y, X, weights, cluster_var=None, cluster_groups=None):
    """Matrix-optimized regression using direct interface (no formula parsing)"""
    
    try:
        if cluster_var is not None and cluster_groups is not None:
            # For survey columns, use clustering (columns 3-5) - must use statsmodels
            model = sm.WLS(y, X, weights=weights)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_groups})
            treatment_coef = results.params[-2]
            spillover_coef = results.params[-1]
        else:
            # For admin columns, use direct matrix calculation (1.9x faster)
            from scipy import linalg
            
            # Weighted least squares: (X'WX)^-1 X'Wy
            W = np.diag(weights)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
            coeffs = linalg.solve(XtWX, XtWy)
            treatment_coef = coeffs[-2]
            spillover_coef = coeffs[-1]
        
        return treatment_coef, spillover_coef
    
    except Exception as e:
        # Regression failure (rare with extreme simulations) 
        return -10, -10

#%%
def run_single_simulation_vectorized(sim_col, X_matrix, y_vector, weights_vector, 
                                   df_sim_file, cluster_var, cluster_groups,
                                   sim_to_data_indices, valid_mask):
    """Run vectorized simulation regression (25.6x faster alignment)"""
    
    try:
        # OPTIMIZATION: Get treatment assignment
        treatment_assignment = df_sim_file[sim_col].values
        
        # OPTIMIZATION: Vectorized alignment (25.6x speedup)
        aligned_treatment = np.zeros(len(y_vector))
        valid_indices = sim_to_data_indices[valid_mask]
        valid_treatments = treatment_assignment[valid_mask]
        aligned_treatment[valid_indices] = valid_treatments
        
        # OPTIMIZATION: Update treatment columns in pre-computed design matrix
        X_sim = X_matrix.copy()
        X_sim[:, -2] = (aligned_treatment == 1).astype(float)  # Treatment indicator  
        X_sim[:, -1] = (aligned_treatment == 2).astype(float)  # Spillover indicator
        
        # OPTIMIZATION: Direct matrix calculation for admin, statsmodels for survey
        treat_coef, spill_coef = run_optimized_regression_matrix(
            y_vector, X_sim, weights_vector, cluster_var, cluster_groups
        )
        
        return (treat_coef, spill_coef)
    
    except Exception as e:
        return (-10, -10)

# Keep the old function for backward compatibility in tests
def run_single_simulation_matrix_optimized(sim_col, X_matrix, y_vector, weights_vector, 
                                         df_sim_file, cluster_var, manzana_mapping_info, cluster_groups):
    """Run matrix-optimized regression for a single simulation (legacy version)"""
    
    try:
        # OPTIMIZATION: Get treatment assignment with proper index alignment
        treatment_assignment = df_sim_file[sim_col].values
        sim_manzanas = df_sim_file['manzana'].values
        
        # Align simulation data with actual data using manzana matching
        manzana_to_row = manzana_mapping_info['manzana_to_row']
        
        # Create aligned treatment vector for data rows
        aligned_treatment = np.zeros(len(y_vector))
        
        # Match simulation manzanas to data rows
        for sim_idx, sim_manzana in enumerate(sim_manzanas):
            if sim_manzana in manzana_to_row:
                data_row = manzana_to_row[sim_manzana]
                aligned_treatment[data_row] = treatment_assignment[sim_idx]
        
        # OPTIMIZATION: Update treatment columns in pre-computed design matrix
        X_sim = X_matrix.copy()
        X_sim[:, -2] = (aligned_treatment == 1).astype(float)  # Treatment indicator  
        X_sim[:, -1] = (aligned_treatment == 2).astype(float)  # Spillover indicator
        
        # OPTIMIZATION: Direct matrix interface (no formula parsing!)
        treat_coef, spill_coef = run_optimized_regression_matrix(
            y_vector, X_sim, weights_vector, cluster_var, cluster_groups
        )
        
        return (treat_coef, spill_coef)
    
    except Exception as e:
        return (-10, -10)

#%%
def process_simulation_batch_optimized(X_matrix, y_vector, weights_vector, 
                                     df_sim_file, sim_cols, cluster_var, manzana_mapping_info, cluster_groups,
                                     use_simulation_multiprocessing=True):
    """Process simulations with adaptive parallelization strategy"""
    
    total_sims = len(sim_cols)
    print(f"    Processing {total_sims} simulations...")
    
    # Pre-compute vectorized alignment indices once per batch (25.6x speedup)
    sim_manzanas = df_sim_file['manzana'].values
    manzana_to_row = manzana_mapping_info['manzana_to_row']
    
    # Create vectorized mapping
    sim_to_data_indices = np.array([manzana_to_row.get(manzana, -1) for manzana in sim_manzanas])
    valid_mask = sim_to_data_indices >= 0
    
    # ADAPTIVE STRATEGY: Use simulation-level multiprocessing for slow survey columns
    if use_simulation_multiprocessing and cluster_var is not None and total_sims >= 1000:
        # Survey columns with clustering: parallelize simulations within the column
        from multiprocessing import Pool
        from functools import partial
        
        # Split into smaller chunks for parallel processing
        n_workers = min(mp.cpu_count() - 1, 10)  # Max 10 workers to avoid overhead
        chunk_size = max(100, total_sims // (n_workers * 2))  # At least 100 sims per chunk
        
        print(f"      Using {n_workers} workers with chunk size {chunk_size} for survey column")
        
        # Create chunks of simulation columns
        sim_chunks = [sim_cols[i:i + chunk_size] for i in range(0, total_sims, chunk_size)]
        
        # Create partial function for chunk processing
        process_chunk_func = partial(
            process_simulation_chunk,
            X_matrix=X_matrix,
            y_vector=y_vector,
            weights_vector=weights_vector,
            df_sim_file=df_sim_file,
            cluster_var=cluster_var,
            cluster_groups=cluster_groups,
            sim_to_data_indices=sim_to_data_indices,
            valid_mask=valid_mask
        )
        
        # Process chunks in parallel
        with Pool(processes=n_workers) as pool:
            chunk_results = pool.map(process_chunk_func, sim_chunks)
        
        # Flatten results
        coefficients = []
        for chunk_result in chunk_results:
            coefficients.extend(chunk_result)
        
        print(f"      Completed {len(coefficients)} simulations using parallel chunks")
        
    else:
        # Admin columns or small batches: sequential processing (fast enough)
        print(f"      Using sequential processing for admin column")
        coefficients = []
        for i, sim_col in enumerate(sim_cols):
            if (i + 1) % 1000 == 0:
                print(f"        Processed {i + 1}/{len(sim_cols)} simulations...")
            
            result = run_single_simulation_vectorized(
                sim_col, X_matrix, y_vector, weights_vector, 
                df_sim_file, cluster_var, cluster_groups,
                sim_to_data_indices, valid_mask
            )
            coefficients.append(result)
    
    return coefficients

def process_simulation_chunk(sim_chunk, X_matrix, y_vector, weights_vector, 
                           df_sim_file, cluster_var, cluster_groups,
                           sim_to_data_indices, valid_mask):
    """Process a chunk of simulations (for multiprocessing)"""
    
    coefficients = []
    for sim_col in sim_chunk:
        result = run_single_simulation_vectorized(
            sim_col, X_matrix, y_vector, weights_vector, 
            df_sim_file, cluster_var, cluster_groups,
            sim_to_data_indices, valid_mask
        )
        coefficients.append(result)
    
    return coefficients

#%%
def save_coefficients_batch(coefficients, col_num, part_num):
    """Save coefficients for one batch (10,000 simulations)"""
    
    # Extract treatment and spillover coefficients
    treatment_coeffs = [coef[0] for coef in coefficients]
    spillover_coeffs = [coef[1] for coef in coefficients]
    simulation_names = [f"sim_{i+1}" for i in range(len(coefficients))]
    
    # Create DataFrame
    df_coeffs = pd.DataFrame({
        'simulation': simulation_names,
        'treat_ef': treatment_coeffs,
        'spillover_ef': spillover_coeffs
    })
    
    # Save to file
    coef_dir = PATHS['data_rand_new'] / 'coefs'
    coef_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = coef_dir / f'RI_table1_col{col_num}_p{part_num}.csv'
    df_coeffs.to_csv(output_file, index=False, float_format='%.8f')
    
    print(f"    Saved {len(coefficients)} coefficients to {output_file}")
    return output_file

#%%
def process_single_column(col_num, batch_range, test_mode):
    """Process a single column (for multiprocessing)"""
    try:
        print(f"\nProcessing column {col_num}...")
        column_start = time.time()
        
        # OPTIMIZATION 1: Load dataset once per column
        df, outcome_var, control_vars, cluster_var = load_column_data(col_num)
        
        # OPTIMIZATION 2: Pre-compute design matrix once per column (major speedup!)
        X_matrix, y_vector, weights_vector, manzana_mapping_info, df_clean = build_design_matrix_optimized(
            df, outcome_var, control_vars, cluster_var
        )
        
        # Prepare cluster groups if needed
        cluster_groups = df_clean[cluster_var].values if cluster_var else None
        
        all_coefficients = []
        batch_results = []
        
        # Process each simulation file as a batch
        batch_limit = 2 if test_mode else batch_range[1]
        for p in range(batch_range[0], batch_limit):
            print(f"  Column {col_num} - Processing batch p{p}...")
            batch_start = time.time()
            
            try:
                # Load one simulation file (25MB) - use new generated data
                sim_file = PATHS['data_rand_new'] / f'block_simulate_randomizations_p{p}.csv'
                if not sim_file.exists():
                    print(f"    WARNING: Simulation file not found: {sim_file}")
                    continue
                
                df_sim_file = pd.read_csv(sim_file)
                sim_cols = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')]
                
                if test_mode:
                    sim_cols = sim_cols[:100]  # Test with only 100 simulations
                
                print(f"    Column {col_num} - Found {len(sim_cols)} simulations in batch p{p}")
                
                # Adaptive processing: parallel for survey columns, sequential for admin
                batch_coefficients = process_simulation_batch_optimized(
                    X_matrix, y_vector, weights_vector, df_sim_file, sim_cols, 
                    cluster_var, manzana_mapping_info, cluster_groups,
                    use_simulation_multiprocessing=True
                )
                
                all_coefficients.extend(batch_coefficients)
                
                # Save coefficients for this batch
                output_file = save_coefficients_batch(batch_coefficients, col_num, p)
                
                batch_elapsed = time.time() - batch_start
                successful = sum(1 for c in batch_coefficients if c[0] != -10)
                failed = len(batch_coefficients) - successful
                
                batch_results.append({
                    'batch_num': p,
                    'n_successful': successful,
                    'n_failed': failed,
                    'elapsed_time': batch_elapsed,
                    'output_file': output_file
                })
                
                print(f"    Column {col_num} - Batch p{p} complete: {successful} successful, {failed} failed ({batch_elapsed:.1f}s)")
                
            except Exception as e:
                print(f"    ERROR in Column {col_num}, batch p{p}: {e}")
                batch_results.append({
                    'batch_num': p,
                    'error': str(e)
                })
        
        column_elapsed = time.time() - column_start
        total_successful = sum(r.get('n_successful', 0) for r in batch_results)
        total_failed = sum(r.get('n_failed', 0) for r in batch_results)
        
        print(f"\nColumn {col_num} completed:")
        print(f"  Time: {column_elapsed:.1f}s ({column_elapsed/60:.1f}m)")
        print(f"  Coefficients: {total_successful} successful, {total_failed} failed")
        print(f"  Batches: {len([r for r in batch_results if 'error' not in r])}/{len(batch_results)}")
        
        return {
            'column_num': col_num,
            'batch_results': batch_results,
            'total_coefficients': len(all_coefficients),
            'total_time': column_elapsed,
            'total_successful': total_successful,
            'total_failed': total_failed
        }
        
    except Exception as e:
        print(f"ERROR processing column {col_num}: {e}")
        return {
            'column_num': col_num,
            'error': str(e),
            'total_time': 0
        }

def precompute_all_coefficients_optimized(columns=None, batch_range=(1, 11), test_mode=False, use_column_multiprocessing=False):
    """Matrix-optimized coefficient pre-computation with column-level multiprocessing"""
    
    if columns is None:
        columns = [1, 2, 3, 4, 5]
    
    print(f"="*60)
    print(f"RI COEFFICIENT PRE-COMPUTATION (LEVEL 2 OPTIMIZATION)")
    print(f"Columns: {columns}")
    print(f"Batches: p{batch_range[0]} to p{batch_range[1]-1}")
    print(f"Test mode: {test_mode}")
    print(f"Column multiprocessing: {use_column_multiprocessing}")
    print(f"Start time: {datetime.now()}")
    print(f"="*60)
    
    # Create output directory
    create_output_directories()
    
    overall_start = time.time()
    
    if use_column_multiprocessing and len(columns) > 1:
        # PARALLEL: Process multiple columns simultaneously
        from multiprocessing import Pool
        from functools import partial
        
        n_workers = min(mp.cpu_count() - 1, len(columns))
        print(f"Using {n_workers} workers for {len(columns)} columns")
        
        # Create partial function with fixed parameters
        process_func = partial(process_single_column, batch_range=batch_range, test_mode=test_mode)
        
        with Pool(processes=n_workers) as pool:
            column_results = pool.map(process_func, columns)
        
        # Convert results to dictionary
        all_results = {result['column_num']: result for result in column_results}
        
    else:
        # SEQUENTIAL: Process columns one by one (original approach)
        print("Processing columns sequentially...")
        all_results = {}
        
        for col_num in columns:
            result = process_single_column(col_num, batch_range, test_mode)
            all_results[col_num] = result
    
    overall_elapsed = time.time() - overall_start
    
    # Summary
    print(f"\n" + "="*60)
    print(f"PRE-COMPUTATION COMPLETE")
    print(f"="*60)
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}m)")
    print(f"Columns processed: {len(columns)}")
    
    # Per-column summary
    total_coefficients = 0
    for col_num in columns:
        if col_num in all_results and 'error' not in all_results[col_num]:
            result = all_results[col_num]
            total_successful = result.get('total_successful', 0)
            total_time = result.get('total_time', 0)
            total_coefficients += total_successful
            print(f"  Column {col_num}: {total_successful} coefficients ({total_time:.1f}s)")
        else:
            print(f"  Column {col_num}: ERROR")
    
    # Performance summary
    if overall_elapsed > 0:
        overall_rate = total_coefficients / overall_elapsed
        print(f"\nPerformance Summary:")
        print(f"  Total coefficients: {total_coefficients}")
        print(f"  Overall rate: {overall_rate:.1f} coefficients/second")
        if total_coefficients > 0:
            projected_full_time = 500000 / overall_rate / 60
            print(f"  Projected full time (500k coefficients): {projected_full_time:.1f} minutes")
    
    # Output files summary
    coef_dir = PATHS['data_rand_new'] / 'coefs'
    output_files = list(coef_dir.glob('RI_table1_col*.csv'))
    print(f"\nOutput files: {len(output_files)} coefficient files created")
    print(f"Location: {coef_dir}")
    
    return all_results

#%%
def test_design_matrix_build():
    """Test Step 1: Design matrix construction"""
    print("="*50)
    print("TEST 1: Design Matrix Construction")
    print("="*50)
    
    for col_num in [1, 3]:  # Test admin and survey columns
        print(f"\nTesting Column {col_num}...")
        
        # Load data
        df, outcome_var, control_vars, cluster_var = load_column_data(col_num)
        
        # Build design matrix
        X_matrix, y_vector, weights_vector, manzana_mapping_info, df_clean = build_design_matrix_optimized(
            df, outcome_var, control_vars, cluster_var
        )
        
        # Validate dimensions
        assert X_matrix.shape[0] == len(y_vector) == len(weights_vector)
        assert X_matrix.shape[1] >= 2  # At least intercept + treatment/spillover placeholders
        assert len(manzana_mapping_info['manzana_to_row']) > 0  # Should have manzana mappings
        
        # Validate placeholders (last two columns should be zeros)
        assert np.all(X_matrix[:, -2] == 0), "Treatment placeholder should be zeros"
        assert np.all(X_matrix[:, -1] == 0), "Spillover placeholder should be zeros"
        
        print(f"  ✓ Design matrix shape: {X_matrix.shape}")
        print(f"  ✓ Response vector length: {len(y_vector)}")
        print(f"  ✓ Weights vector length: {len(weights_vector)}")
        print(f"  ✓ Manzana mappings: {len(manzana_mapping_info['manzana_to_row'])}")
        
    print("\n✅ Design matrix construction test PASSED")

#%%
def test_single_simulation():
    """Test Step 2: Single simulation regression"""
    print("="*50)
    print("TEST 2: Single Simulation Regression")
    print("="*50)
    
    # Test Column 1 (admin data)
    col_num = 1
    print(f"\nTesting Column {col_num}...")
    
    # Load data and build matrix
    df, outcome_var, control_vars, cluster_var = load_column_data(col_num)
    X_matrix, y_vector, weights_vector, manzana_mapping_info, df_clean = build_design_matrix_optimized(
        df, outcome_var, control_vars, cluster_var
    )
    cluster_groups = df_clean[cluster_var].values if cluster_var else None
    
    # Load one simulation file
    sim_file = PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv'
    df_sim_file = pd.read_csv(sim_file)
    sim_cols = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')]
    
    # Test first simulation
    sim_col = sim_cols[0]
    print(f"  Testing simulation: {sim_col}")
    
    treat_coef, spill_coef = run_single_simulation_matrix_optimized(
        sim_col, X_matrix, y_vector, weights_vector, 
        df_sim_file, cluster_var, manzana_mapping_info, cluster_groups
    )
    
    # Validate results
    assert isinstance(treat_coef, (int, float)), "Treatment coefficient should be numeric"
    assert isinstance(spill_coef, (int, float)), "Spillover coefficient should be numeric"
    assert treat_coef != -10 or spill_coef != -10, "Regression should not fail for normal data"
    
    print(f"  ✓ Treatment coefficient: {treat_coef:.6f}")
    print(f"  ✓ Spillover coefficient: {spill_coef:.6f}")
    print(f"  ✓ Regression completed successfully")
    
    print("\n✅ Single simulation regression test PASSED")

#%%
def test_batch_processing():
    """Test Step 3: Batch processing with limited simulations"""
    print("="*50)
    print("TEST 3: Batch Processing (100 simulations)")
    print("="*50)
    
    # Test Column 1 with limited simulations
    col_num = 1
    print(f"\nTesting Column {col_num} batch processing...")
    
    # Load data and build matrix
    df, outcome_var, control_vars, cluster_var = load_column_data(col_num)
    X_matrix, y_vector, weights_vector, manzana_mapping_info, df_clean = build_design_matrix_optimized(
        df, outcome_var, control_vars, cluster_var
    )
    cluster_groups = df_clean[cluster_var].values if cluster_var else None
    
    # Load simulation file
    sim_file = PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv'
    df_sim_file = pd.read_csv(sim_file)
    sim_cols = [col for col in df_sim_file.columns if col.startswith('treatment_ri_')][:100]  # Limit to 100
    
    print(f"  Processing {len(sim_cols)} simulations...")
    start_time = time.time()
    
    # Process batch
    batch_coefficients = process_simulation_batch_optimized(
        X_matrix, y_vector, weights_vector, df_sim_file, sim_cols, 
        cluster_var, manzana_mapping_info, cluster_groups
    )
    
    elapsed_time = time.time() - start_time
    
    # Validate results
    assert len(batch_coefficients) == len(sim_cols), "Should have coefficients for all simulations"
    
    successful = sum(1 for c in batch_coefficients if c[0] != -10)
    failed = len(batch_coefficients) - successful
    
    print(f"  ✓ Processed {len(batch_coefficients)} simulations")
    print(f"  ✓ Successful: {successful}, Failed: {failed}")
    print(f"  ✓ Processing time: {elapsed_time:.2f}s")
    print(f"  ✓ Rate: {len(sim_cols)/elapsed_time:.1f} simulations/second")
    
    # Test coefficient saving
    output_file = save_coefficients_batch(batch_coefficients, col_num, 999)  # Test batch number
    assert output_file.exists(), "Output file should be created"
    
    # Validate saved file
    df_saved = pd.read_csv(output_file)
    assert len(df_saved) == len(batch_coefficients), "Saved file should have all coefficients"
    assert 'treat_ef' in df_saved.columns, "Should have treatment effects column"
    assert 'spillover_ef' in df_saved.columns, "Should have spillover effects column"
    
    print(f"  ✓ Coefficients saved to: {output_file}")
    
    print("\n✅ Batch processing test PASSED")

#%%
def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Pre-compute RI coefficients with Level 2 optimization')
    parser.add_argument('--columns', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Columns to process (default: all 1-5)')
    parser.add_argument('--test-mode', action='store_true',
                        help='Test mode: process limited simulations for validation')
    parser.add_argument('--test-only', action='store_true',
                        help='Run tests only, do not process coefficients')
    parser.add_argument('--batch-range', nargs=2, type=int, default=[1, 11],
                        help='Batch range to process (default: 1 11)')
    parser.add_argument('--column-multiprocessing', action='store_true',
                        help='Enable column-level multiprocessing (default: disabled, uses simulation-level MP instead)')
    
    args = parser.parse_args()
    
    if args.test_only:
        # Run validation tests
        print("Running validation tests...")
        test_design_matrix_build()
        test_single_simulation() 
        test_batch_processing()
        print("\n🎉 All tests PASSED! Ready for full processing.")
        return
    
    # Full processing
    all_results = precompute_all_coefficients_optimized(
        columns=args.columns,
        batch_range=tuple(args.batch_range),
        test_mode=args.test_mode,
        use_column_multiprocessing=args.column_multiprocessing
    )
    
    if not args.test_mode:
        print("\nPre-computation complete. Next steps:")
        print("1. Run tests to validate coefficient accuracy")
        print("2. Update 03_tables.py to load pre-computed coefficients")
        print("3. Test full pipeline performance")

#%%
if __name__ == "__main__":
    main()