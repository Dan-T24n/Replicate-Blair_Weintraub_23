"""
test_table1_numerical_comparison.py

Comprehensive test to identify numerical differences between reproduced and original Table 1 results
Blair & Weintraub (2023) Military Policing Replication

Purpose:
- Load reproduced results from 03_tables.py
- Compare with original published results
- Identify specific numerical differences and root causes
- Generate detailed comparison report

Authors: Python Replication Team
Date: 2024
"""

#%%
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config import PATHS
# Import from 03_tables module (using importlib to handle numeric module name)
import importlib.util
spec = importlib.util.spec_from_file_location("tables_module", Path(__file__).parent.parent / 'src' / '03_tables.py')
tables_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables_module)

# Import the functions we need
load_admin_data_during = tables_module.load_admin_data_during
load_admin_data_after = tables_module.load_admin_data_after
load_survey_endline = tables_module.load_survey_endline
run_table1_column1 = tables_module.run_table1_column1
run_table1_column2 = tables_module.run_table1_column2
run_table1_column3 = tables_module.run_table1_column3
run_table1_column4 = tables_module.run_table1_column4
run_table1_column5 = tables_module.run_table1_column5

#%%
# Original Table 1 results from Blair & Weintraub (2023)
ORIGINAL_RESULTS = {
    'column1': {  # Admin data - Crime during intervention
        'treatment_coef': 0.003,
        'treatment_ci': [-0.068, 0.074],
        'treatment_pval': 0.934,
        'spillover_coef': -0.038,
        'spillover_ci': [-0.097, 0.022],
        'spillover_pval': 0.212,
        'n_obs': 1167,
        'r_squared': 0.33,
        'control_mean': 0.160,
        'ri_pval_treatment': 0.959,
        'ri_pval_spillover': 0.411
    },
    'column2': {  # Admin data - Crime after intervention
        'treatment_coef': 0.110,
        'treatment_ci': [0.011, 0.208],
        'treatment_pval': 0.029,
        'spillover_coef': 0.083,
        'spillover_ci': [-0.003, 0.169],
        'spillover_pval': 0.059,
        'n_obs': 1167,
        'r_squared': 0.48,
        'control_mean': 0.160,
        'ri_pval_treatment': 0.136,
        'ri_pval_spillover': 0.138
    },
    'column3': {  # Survey - Crime victimization during intervention
        'treatment_coef': 0.006,
        'treatment_ci': [-0.077, 0.089],
        'treatment_pval': 0.886,
        'spillover_coef': 0.026,
        'spillover_ci': [-0.034, 0.086],
        'spillover_pval': 0.389,
        'n_obs': 7845,
        'r_squared': 0.03,
        'control_mean': -0.021,
        'ri_pval_treatment': 0.927,
        'ri_pval_spillover': 0.610
    },
    'column4': {  # Survey - Crime victimization after intervention
        'treatment_coef': -0.007,
        'treatment_ci': [-0.098, 0.085],
        'treatment_pval': 0.886,
        'spillover_coef': 0.013,
        'spillover_ci': [-0.061, 0.087],
        'spillover_pval': 0.729,
        'n_obs': 7845,
        'r_squared': 0.03,
        'control_mean': -0.016,
        'ri_pval_treatment': 0.914,
        'ri_pval_spillover': 0.802
    },
    'column5': {  # Survey - Crime witnessing after intervention
        'treatment_coef': 0.153,
        'treatment_ci': [0.051, 0.256],
        'treatment_pval': 0.003,
        'spillover_coef': 0.186,
        'spillover_ci': [0.101, 0.270],
        'spillover_pval': '<0.001',  # Special case
        'n_obs': 7837,
        'r_squared': 0.12,
        'control_mean': -0.119,
        'ri_pval_treatment': 0.038,
        'ri_pval_spillover': 0.001
    }
}

#%%
def run_reproduced_regressions():
    """Run all Table 1 regressions and return results"""
    print("Running reproduced regressions...")
    
    # Load data
    admin_during = load_admin_data_during()
    admin_after = load_admin_data_after()  
    survey_endline = load_survey_endline()
    
    # Run regressions
    results = {}
    results['column1'] = run_table1_column1(admin_during)
    results['column2'] = run_table1_column2(admin_after)
    results['column3'] = run_table1_column3(survey_endline)
    results['column4'] = run_table1_column4(survey_endline)
    results['column5'] = run_table1_column5(survey_endline)
    
    return results

#%%
def compare_single_result(original, reproduced, column_name, tolerance=0.005):
    """Compare a single column's results and identify differences"""
    differences = {}
    
    # Define comparison fields
    comparison_fields = [
        'treatment_coef', 'spillover_coef', 'control_mean', 
        'n_obs', 'r_squared', 'treatment_pval', 'spillover_pval'
    ]
    
    for field in comparison_fields:
        orig_val = original[field]
        repro_val = reproduced[field] if field in reproduced else None
        
        # Handle special cases
        if field == 'n_obs':
            # Integer comparison
            diff = orig_val != int(repro_val) if repro_val is not None else True
        elif isinstance(orig_val, str):
            # String comparison (e.g., '<0.001')
            diff = orig_val != str(repro_val) if repro_val is not None else True
        else:
            # Numerical comparison
            diff = abs(orig_val - repro_val) > tolerance if repro_val is not None else True
        
        if diff:
            differences[field] = {
                'original': orig_val,
                'reproduced': repro_val,
                'difference': orig_val - repro_val if isinstance(orig_val, (int, float)) and repro_val is not None else 'N/A'
            }
    
    # Compare confidence intervals
    for treatment_type in ['treatment', 'spillover']:
        ci_field = f'{treatment_type}_ci'
        if ci_field in original and ci_field in reproduced:
            orig_ci = original[ci_field]
            repro_ci = reproduced[ci_field]
            
            ci_diff = (abs(orig_ci[0] - repro_ci[0]) > tolerance or 
                      abs(orig_ci[1] - repro_ci[1]) > tolerance)
            
            if ci_diff:
                differences[ci_field] = {
                    'original': orig_ci,
                    'reproduced': repro_ci,
                    'difference': [orig_ci[0] - repro_ci[0], orig_ci[1] - repro_ci[1]]
                }
    
    return differences

#%%
def generate_comparison_report(original_results, reproduced_results):
    """Generate comprehensive comparison report"""
    print("\n" + "="*80)
    print("TABLE 1 NUMERICAL COMPARISON REPORT")
    print("Blair & Weintraub (2023) Military Policing Replication")
    print("="*80)
    
    all_differences = {}
    
    for col_name in ['column1', 'column2', 'column3', 'column4', 'column5']:
        print(f"\n{col_name.upper()}: {get_column_description(col_name)}")
        print("-" * 60)
        
        differences = compare_single_result(
            original_results[col_name], 
            reproduced_results[col_name], 
            col_name
        )
        
        if differences:
            all_differences[col_name] = differences
            
            for field, diff_info in differences.items():
                print(f"  {field}:")
                print(f"    Original:   {diff_info['original']}")
                print(f"    Reproduced: {diff_info['reproduced']}")
                if diff_info['difference'] != 'N/A':
                    print(f"    Difference: {diff_info['difference']}")
                print()
        else:
            print("  ✓ All values match within tolerance")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_columns = len(original_results)
    columns_with_differences = len(all_differences)
    
    print(f"Total columns compared: {total_columns}")
    print(f"Columns with differences: {columns_with_differences}")
    print(f"Columns matching exactly: {total_columns - columns_with_differences}")
    
    if all_differences:
        print("\nColumns with differences:")
        for col_name in all_differences.keys():
            print(f"  - {col_name}: {len(all_differences[col_name])} fields differ")
    else:
        print("\n✓ All results match the original table within tolerance!")
    
    return all_differences

#%%
def get_column_description(col_name):
    """Get descriptive name for each column"""
    descriptions = {
        'column1': 'Admin Crime During Intervention',
        'column2': 'Admin Crime After Intervention', 
        'column3': 'Survey Victimization During Intervention',
        'column4': 'Survey Victimization After Intervention',
        'column5': 'Survey Crime Witnessing After Intervention'
    }
    return descriptions.get(col_name, col_name)

#%%
def save_detailed_comparison(original_results, reproduced_results, differences, output_path):
    """Save detailed comparison to JSON file for further analysis"""
    comparison_data = {
        'metadata': {
            'comparison_date': datetime.now().isoformat(),
            'tolerance': 0.005,
            'total_columns': len(original_results),
            'columns_with_differences': len(differences)
        },
        'original_results': original_results,
        'reproduced_results': {
            col: {k: float(v) if isinstance(v, np.floating) else v 
                  for k, v in result.items()} 
            for col, result in reproduced_results.items()
        },
        'differences': differences
    }
    
    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    print(f"Detailed comparison saved to: {output_path}")

#%%
def analyze_potential_causes(differences):
    """Analyze potential causes of differences"""
    print("\n" + "="*80)
    print("POTENTIAL CAUSES ANALYSIS")
    print("="*80)
    
    if not differences:
        print("No differences found - analysis not needed.")
        return
    
    # Common causes of differences between Stata and Python
    potential_causes = []
    
    # Check for missing value handling differences
    if any('n_obs' in diff for diff in differences.values()):
        potential_causes.append("Missing value handling: Stata and Python may handle missing values differently")
    
    # Check for coefficient differences
    coef_diffs = []
    for col, diffs in differences.items():
        for field in diffs:
            if 'coef' in field:
                coef_diffs.append((col, field, diffs[field]['difference']))
    
    if coef_diffs:
        potential_causes.append("Coefficient estimation: Differences in numerical optimization or matrix calculations")
    
    # Check for standard error differences (via CI differences)
    se_diffs = []
    for col, diffs in differences.items():
        for field in diffs:
            if '_ci' in field:
                se_diffs.append((col, field))
    
    if se_diffs:
        potential_causes.append("Standard error calculation: Differences in variance-covariance matrix estimation")
    
    # Check for R-squared differences
    if any('r_squared' in diff for diff in differences.values()):
        potential_causes.append("R-squared calculation: Different formulas for weighted regression R-squared")
    
    print("Likely causes of differences:")
    for i, cause in enumerate(potential_causes, 1):
        print(f"{i}. {cause}")
    
    print("\nRecommended actions:")
    print("1. Check variable encoding (integer vs float for treatment)")
    print("2. Verify missing value handling matches Stata exactly")
    print("3. Compare intermediate regression outputs (coefficients matrix)")
    print("4. Check clustering implementation for survey data")
    print("5. Verify inverse probability weight calculations")

#%%
def main():
    """Main function to run comprehensive comparison"""
    print("Starting Table 1 numerical comparison...")
    
    try:
        # Run reproduced regressions
        reproduced_results = run_reproduced_regressions()
        
        # Generate comparison report
        differences = generate_comparison_report(ORIGINAL_RESULTS, reproduced_results)
        
        # Analyze potential causes
        analyze_potential_causes(differences)
        
        # Save detailed comparison
        output_path = PATHS['tests'] / 'table1_comparison_detailed.json'
        save_detailed_comparison(ORIGINAL_RESULTS, reproduced_results, differences, output_path)
        
        print(f"\nComparison completed successfully!")
        
        return differences
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return None

#%%
if __name__ == "__main__":
    differences = main()
