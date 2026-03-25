#!/usr/bin/env python3
"""
Validate the complete randomization inference results with all 100,000 simulations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('src')
from config import PATHS

def load_all_coefficients(col_num):
    """Load coefficients from all 10 simulation files for a column"""
    
    all_coefficients = []
    coef_dir = PATHS['data_rand'] / 'coefs'
    
    print(f"Loading coefficients for Column {col_num}...")
    
    for p in range(1, 11):
        coef_file = coef_dir / f'RI_table1_col{col_num}_p{p}.csv'
        
        if coef_file.exists():
            df = pd.read_csv(coef_file)
            all_coefficients.append(df)
            print(f"  ✓ Loaded p{p}: {len(df)} coefficients")
        else:
            print(f"  ❌ Missing p{p}: {coef_file}")
            return None
    
    if len(all_coefficients) == 10:
        combined_df = pd.concat(all_coefficients, ignore_index=True)
        print(f"  📊 Total coefficients: {len(combined_df)}")
        return combined_df
    else:
        print(f"  ❌ Only found {len(all_coefficients)}/10 files")
        return None

def validate_full_column(col_num, expected_treatment, expected_spillover, expected_treatment_p, expected_spillover_p):
    """Validate a column with full 100,000 simulations"""
    
    print(f"\n{'='*60}")
    print(f"📊 COLUMN {col_num} FULL VALIDATION (100,000 simulations)")
    print(f"{'='*60}")
    
    # Load all coefficients
    df = load_all_coefficients(col_num)
    
    if df is None:
        print("❌ Cannot validate - missing coefficient files")
        return None
    
    treatment_coefs = df['treat_ef'].values
    spillover_coefs = df['spillover_ef'].values
    
    print(f"\nTREATMENT COEFFICIENTS:")
    print(f"  Count: {len(treatment_coefs):,}")
    print(f"  Mean:  {np.mean(treatment_coefs):.6f}")
    print(f"  Std:   {np.std(treatment_coefs):.6f}")
    print(f"  Min:   {np.min(treatment_coefs):.6f}")
    print(f"  Max:   {np.max(treatment_coefs):.6f}")
    print(f"  Expected: {expected_treatment:.6f}")
    
    print(f"\nSPILLOVER COEFFICIENTS:")
    print(f"  Count: {len(spillover_coefs):,}")
    print(f"  Mean:  {np.mean(spillover_coefs):.6f}")
    print(f"  Std:   {np.std(spillover_coefs):.6f}")
    print(f"  Min:   {np.min(spillover_coefs):.6f}")
    print(f"  Max:   {np.max(spillover_coefs):.6f}")
    print(f"  Expected: {expected_spillover:.6f}")
    
    # Check if expected coefficients are in range
    treatment_in_range = np.min(treatment_coefs) <= expected_treatment <= np.max(treatment_coefs)
    spillover_in_range = np.min(spillover_coefs) <= expected_spillover <= np.max(spillover_coefs)
    
    print(f"\n✅ RANGE VALIDATION:")
    print(f"  Treatment in range: {'✅' if treatment_in_range else '❌'}")
    print(f"  Spillover in range: {'✅' if spillover_in_range else '❌'}")
    
    # Calculate precise RI p-values
    treatment_pval = np.mean(np.abs(treatment_coefs) >= abs(expected_treatment))
    spillover_pval = np.mean(np.abs(spillover_coefs) >= abs(expected_spillover))
    
    print(f"\n🎯 RANDOMIZATION INFERENCE P-VALUES:")
    print(f"  Treatment: {treatment_pval:.4f} (expected: {expected_treatment_p:.3f})")
    print(f"  Spillover:  {spillover_pval:.4f} (expected: {expected_spillover_p:.3f})")
    
    # P-value accuracy
    treatment_p_diff = abs(treatment_pval - expected_treatment_p)
    spillover_p_diff = abs(spillover_pval - expected_spillover_p)
    
    print(f"\n📊 P-VALUE ACCURACY:")
    print(f"  Treatment difference: {treatment_p_diff:.4f}")
    print(f"  Spillover difference:  {spillover_p_diff:.4f}")
    
    # Overall validation
    range_ok = treatment_in_range and spillover_in_range
    pval_ok = treatment_p_diff < 0.1 and spillover_p_diff < 0.1  # Within 10% is good for replication
    
    if range_ok and pval_ok:
        print(f"\n🎉 COLUMN {col_num} VALIDATION: ✅ PASS")
    elif range_ok:
        print(f"\n⚠️  COLUMN {col_num} VALIDATION: 🔶 PARTIAL (ranges good, p-values off)")
    else:
        print(f"\n❌ COLUMN {col_num} VALIDATION: ❌ FAIL")
    
    return {
        'column': col_num,
        'total_coefficients': len(treatment_coefs),
        'treatment_in_range': treatment_in_range,
        'spillover_in_range': spillover_in_range,
        'treatment_pval': treatment_pval,
        'spillover_pval': spillover_pval,
        'treatment_p_diff': treatment_p_diff,
        'spillover_p_diff': spillover_p_diff,
        'validation_pass': range_ok and pval_ok
    }

def main():
    """Run full validation for all columns"""
    
    print("🔍 FULL RANDOMIZATION INFERENCE VALIDATION")
    print("=" * 60)
    print("Testing with complete 100,000 simulations per column")
    
    # Table 1 results and expected p-values
    table1_results = {
        1: (0.003, -0.038, 0.959, 0.411),   # treatment, spillover, treatment_p, spillover_p
        2: (0.110, 0.083, 0.136, 0.138),
        3: (0.006, 0.026, 0.927, 0.610),
        4: (-0.007, 0.013, 0.914, 0.802),
        5: (0.153, 0.186, 0.038, 0.001)
    }
    
    results = []
    
    # Validate each column
    for col_num in [1, 2, 3, 4, 5]:
        expected_treat, expected_spill, expected_treat_p, expected_spill_p = table1_results[col_num]
        result = validate_full_column(col_num, expected_treat, expected_spill, expected_treat_p, expected_spill_p)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 OVERALL VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    if results:
        total_coefficients = sum(r['total_coefficients'] for r in results)
        passed_columns = sum(1 for r in results if r['validation_pass'])
        
        print(f"Total coefficients processed: {total_coefficients:,}")
        print(f"Columns validated: {len(results)}/5")
        print(f"Columns passed: {passed_columns}/{len(results)}")
        
        print(f"\nPer-column results:")
        print(f"{'Col':<3} {'Coeffs':<8} {'Range':<7} {'P-val Acc':<10} {'Status':<8}")
        print("-" * 40)
        
        for result in results:
            col = result['column']
            coeffs = f"{result['total_coefficients']:,}"
            range_status = "✅" if result['treatment_in_range'] and result['spillover_in_range'] else "❌"
            pval_acc = f"{max(result['treatment_p_diff'], result['spillover_p_diff']):.3f}"
            status = "✅ PASS" if result['validation_pass'] else "❌ FAIL"
            
            print(f"{col:<3} {coeffs:<8} {range_status:<7} {pval_acc:<10} {status:<8}")
        
        if passed_columns == len(results):
            print(f"\n🎉 OVERALL RESULT: ✅ ALL COLUMNS VALIDATED!")
            print("✅ Randomization inference is working correctly with full simulation data")
        else:
            print(f"\n⚠️  OVERALL RESULT: {passed_columns}/{len(results)} columns passed")
    else:
        print("❌ No results to summarize")

if __name__ == "__main__":
    main()
