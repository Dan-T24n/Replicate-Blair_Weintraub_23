"""
validate_simulation_data.py

Blair & Weintraub (2023) Military Policing Replication
Simple validation of randomization inference simulation data quality

Purpose: Validate treatment proportion distributions in generated simulation data
- Compare treatment proportions with expected values from original data
- Report data quality metrics

Usage:
    python validate_simulation_data.py data/rand_new
    python validate_simulation_data.py data/rand

Authors: Robert Blair and Michael Weintraub (original)
Python validation: 2024
"""

#%%
# Import required packages
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Import project configuration
sys.path.append('..')
from config import PATHS, DATA_FILES

#%%
def load_simulation_data(data_dir):
    """
    Load all simulation files and return first file for validation
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Find simulation files
    sim_files = list(data_path.glob('block_simulate_randomizations_p*.csv'))
    if not sim_files:
        raise FileNotFoundError(f"No simulation files found in {data_path}")
    
    print(f"Found {len(sim_files)} simulation files in {data_path}")
    
    # Load first file for validation
    first_file = sorted(sim_files)[0]
    df = pd.read_csv(first_file)
    
    # Count total simulations across all files
    total_simulations = 0
    for file_path in sim_files:
        temp_df = pd.read_csv(file_path)
        sim_cols = [col for col in temp_df.columns if col.startswith('treatment_ri_')]
        total_simulations += len(sim_cols)
    
    return df, len(sim_files), total_simulations

#%%
def validate_data_quality(df, n_files, total_simulations):
    """
    Validate simulation data quality by checking treatment proportions
    """
    print("\n" + "="*60)
    print("DATA QUALITY VALIDATION")
    print("="*60)
    
    # Expected proportions from original data analysis
    expected_props = {
        0: 0.219,  # Control (from original data: 21.9%)
        1: 0.171,  # Treatment (from original data: 17.1%)
        2: 0.610   # Spillover (from original data: 61.0%)
    }
    
    print(f"Files: {n_files}")
    print(f"Total simulations: {total_simulations:,}")
    print(f"Expected proportions: Control={expected_props[0]:.1%}, Treatment={expected_props[1]:.1%}, Spillover={expected_props[2]:.1%}")
    
    # Get simulation columns
    sim_cols = [col for col in df.columns if col.startswith('treatment_ri_')]
    print(f"Analyzing {len(sim_cols):,} simulations from first file...")
    
    # Check first 5 simulations in detail
    print("\nFirst 5 simulations:")
    valid_count = 0
    tolerance = 0.05
    
    for i, sim_col in enumerate(sim_cols[:5]):
        props = df[sim_col].value_counts(normalize=True).sort_index()
        actual_props = {j: props.get(j, 0) for j in [0, 1, 2]}
        
        # Check validity
        is_valid = all(abs(actual_props[j] - expected_props[j]) < tolerance for j in [0, 1, 2])
        valid_count += is_valid
        
        status = "✓" if is_valid else "⚠"
        print(f"  {sim_col}: {status} Control={actual_props[0]:.1%}, Treatment={actual_props[1]:.1%}, Spillover={actual_props[2]:.1%}")
    
    # Overall statistics
    print(f"\nOverall statistics ({len(sim_cols):,} simulations):")
    
    all_valid = 0
    prop_stats = {0: [], 1: [], 2: []}
    
    for sim_col in sim_cols:
        props = df[sim_col].value_counts(normalize=True).sort_index()
        actual_props = {j: props.get(j, 0) for j in [0, 1, 2]}
        
        for treatment_code in [0, 1, 2]:
            prop_stats[treatment_code].append(actual_props[treatment_code])
        
        is_valid = all(abs(actual_props[j] - expected_props[j]) < tolerance for j in [0, 1, 2])
        all_valid += is_valid
    
    # Report statistics
    for treatment_code in [0, 1, 2]:
        treatment_name = ['Control', 'Treatment', 'Spillover'][treatment_code]
        props_array = np.array(prop_stats[treatment_code])
        mean_prop = np.mean(props_array)
        std_prop = np.std(props_array)
        
        print(f"  {treatment_name}: mean={mean_prop:.1%} ±{std_prop:.3f} (expected={expected_props[treatment_code]:.1%})")
    
    # Quality assessment
    valid_pct = all_valid / len(sim_cols)
    print(f"\nQuality assessment: {all_valid:,}/{len(sim_cols):,} ({valid_pct:.1%}) simulations within ±{tolerance:.0%} tolerance")
    
    if valid_pct > 0.95:
        print("✅ EXCELLENT simulation quality")
        return "excellent"
    elif valid_pct > 0.90:
        print("✅ GOOD simulation quality")
        return "good"
    elif valid_pct > 0.80:
        print("⚠️ ACCEPTABLE simulation quality")
        return "acceptable"
    else:
        print("❌ POOR simulation quality")
        return "poor"



#%%
def main():
    """
    Main validation function
    """
    if len(sys.argv) != 2:
        print("Usage: python validate_simulation_data.py <data_directory>")
        print("Example: python validate_simulation_data.py data/rand_new")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    print("="*70)
    print("SIMULATION DATA QUALITY VALIDATION")
    print("Blair & Weintraub (2023) Military Policing Replication")
    print("="*70)
    
    try:
        # Load simulation data
        df, n_files, total_simulations = load_simulation_data(data_dir)
        
        # Validate data quality
        quality = validate_data_quality(df, n_files, total_simulations)
        
        # Final summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"Data directory: {data_dir}")
        print(f"Quality rating: {quality.upper()}")
        
        if quality in ['excellent', 'good']:
            print("✅ VALIDATION PASSED")
            sys.exit(0)
        else:
            print("⚠️ VALIDATION CONCERNS")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(1)

#%%
if __name__ == "__main__":
    main()
