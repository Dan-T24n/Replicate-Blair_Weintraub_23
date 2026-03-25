#!/usr/bin/env python3
"""
Validate pre-computed RI coefficients against Table 1 results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('src')
from config import PATHS

def analyze_coefficient_distribution(col_num, expected_treatment, expected_spillover):
    """Analyze coefficient distribution for a column"""
    
    # Load coefficient file
    coef_file = PATHS['data_rand'] / 'coefs' / f'RI_table1_col{col_num}_p1.csv'
    
    if not coef_file.exists():
        print(f"❌ Coefficient file not found: {coef_file}")
        return
    
    df = pd.read_csv(coef_file)
    
    print(f"\n📊 COLUMN {col_num} ANALYSIS")
    print(f"=" * 50)
    
    # Basic statistics
    treatment_coefs = df['treat_ef'].values
    spillover_coefs = df['spillover_ef'].values
    
    print(f"Number of simulations: {len(treatment_coefs)}")
    print(f"\nTREATMENT COEFFICIENTS:")
    print(f"  Mean: {np.mean(treatment_coefs):.6f}")
    print(f"  Std:  {np.std(treatment_coefs):.6f}")
    print(f"  Min:  {np.min(treatment_coefs):.6f}")
    print(f"  Max:  {np.max(treatment_coefs):.6f}")
    print(f"  Expected from Table 1: {expected_treatment}")
    
    print(f"\nSPILLOVER COEFFICIENTS:")
    print(f"  Mean: {np.mean(spillover_coefs):.6f}")
    print(f"  Std:  {np.std(spillover_coefs):.6f}")
    print(f"  Min:  {np.min(spillover_coefs):.6f}")
    print(f"  Max:  {np.max(spillover_coefs):.6f}")
    print(f"  Expected from Table 1: {expected_spillover}")
    
    # Check if observed coefficient is within the simulated range
    treatment_in_range = np.min(treatment_coefs) <= expected_treatment <= np.max(treatment_coefs)
    spillover_in_range = np.min(spillover_coefs) <= expected_spillover <= np.max(spillover_coefs)
    
    print(f"\n✅ VALIDATION:")
    print(f"  Treatment coef in simulated range: {treatment_in_range}")
    print(f"  Spillover coef in simulated range: {spillover_in_range}")
    
    # Calculate approximate p-values (proportion of simulated coefficients >= observed)
    treatment_pval = np.mean(np.abs(treatment_coefs) >= abs(expected_treatment))
    spillover_pval = np.mean(np.abs(spillover_coefs) >= abs(expected_spillover))
    
    print(f"\n🎯 APPROXIMATE RI P-VALUES:")
    print(f"  Treatment: {treatment_pval:.3f}")
    print(f"  Spillover: {spillover_pval:.3f}")
    
    return {
        'column': col_num,
        'treatment_mean': np.mean(treatment_coefs),
        'treatment_std': np.std(treatment_coefs),
        'treatment_in_range': treatment_in_range,
        'treatment_pval': treatment_pval,
        'spillover_mean': np.mean(spillover_coefs),
        'spillover_std': np.std(spillover_coefs),
        'spillover_in_range': spillover_in_range,
        'spillover_pval': spillover_pval
    }

def main():
    """Main validation function"""
    
    print("🔍 RANDOMIZATION INFERENCE COEFFICIENT VALIDATION")
    print("=" * 60)
    print("Comparing pre-computed coefficients to Table 1 results")
    
    # Table 1 results (treatment, spillover coefficients)
    table1_results = {
        1: (0.003, -0.038),   # Crime during intervention (admin)
        2: (0.110, 0.083),    # Crime after intervention (admin)  
        3: (0.006, 0.026),    # Victimization during (survey)
        4: (-0.007, 0.013),   # Victimization after (survey)
        5: (0.153, 0.186)     # Witnessing after (survey)
    }
    
    # Expected p-values from Table 1
    table1_pvalues = {
        1: (0.959, 0.411),    # Treatment, Spillover p-values
        2: (0.136, 0.138),
        3: (0.927, 0.610),
        4: (0.914, 0.802),
        5: (0.038, 0.001)
    }
    
    results = []
    
    # Analyze each column
    for col_num in [1, 2, 3, 4, 5]:
        if col_num in table1_results:
            expected_treat, expected_spill = table1_results[col_num]
            result = analyze_coefficient_distribution(col_num, expected_treat, expected_spill)
            if result:
                results.append(result)
    
    # Summary comparison
    print(f"\n" + "=" * 60)
    print("📋 SUMMARY COMPARISON")
    print("=" * 60)
    
    print(f"{'Col':<3} {'Treatment':<12} {'Expected':<10} {'In Range':<9} {'P-val':<7} {'Expected P':<11}")
    print("-" * 60)
    
    for result in results:
        col = result['column']
        expected_treat, expected_spill = table1_results[col]
        expected_treat_p, expected_spill_p = table1_pvalues[col]
        
        print(f"{col:<3} {result['treatment_mean']:<12.6f} {expected_treat:<10.3f} "
              f"{'✅' if result['treatment_in_range'] else '❌':<9} "
              f"{result['treatment_pval']:<7.3f} {expected_treat_p:<11.3f}")
    
    print("\nSpillover coefficients:")
    print(f"{'Col':<3} {'Spillover':<12} {'Expected':<10} {'In Range':<9} {'P-val':<7} {'Expected P':<11}")
    print("-" * 60)
    
    for result in results:
        col = result['column']
        expected_treat, expected_spill = table1_results[col]
        expected_treat_p, expected_spill_p = table1_pvalues[col]
        
        print(f"{col:<3} {result['spillover_mean']:<12.6f} {expected_spill:<10.3f} "
              f"{'✅' if result['spillover_in_range'] else '❌':<9} "
              f"{result['spillover_pval']:<7.3f} {expected_spill_p:<11.3f}")
    
    # Overall validation
    all_in_range = all(r['treatment_in_range'] and r['spillover_in_range'] for r in results)
    
    print(f"\n🎯 OVERALL VALIDATION: {'✅ PASS' if all_in_range else '❌ FAIL'}")
    
    if all_in_range:
        print("✅ All observed coefficients fall within simulated ranges")
        print("✅ RI coefficient pre-computation appears to be working correctly")
    else:
        print("❌ Some coefficients are outside expected ranges")
        print("❌ May indicate issues with coefficient calculation")

if __name__ == "__main__":
    main()
