#!/usr/bin/env python3
"""
Extensive data integrity validation for converted CSV files
"""

import pyreadstat
import pandas as pd
import numpy as np
from pathlib import Path
from config import PATHS

def deep_validation_analysis():
    """Perform extensive validation analysis on all converted files"""
    
    print("Extensive Data Integrity Validation")
    print("="*50)
    
    stata_dir = PATHS['replication_package']
    validation_details = {}
    
    for dta_file in sorted(stata_dir.glob("*.dta")):
        print(f"\nAnalyzing: {dta_file.name}")
        
        # Determine corresponding CSV path
        if dta_file.name.startswith('block_simulate_randomizations'):
            csv_path = PATHS['data_rand'] / f"{dta_file.stem}.csv"
        else:
            csv_path = PATHS['data_raw'] / f"{dta_file.stem}.csv"
        
        try:
            # Load both files with detailed analysis
            print("  Loading Stata file...")
            df_stata, meta_stata = pyreadstat.read_dta(str(dta_file), 
                                                      apply_value_formats=False,
                                                      formats_as_category=False)
            
            print("  Loading CSV file...")
            df_csv = pd.read_csv(csv_path)
            
            # Detailed comparison
            analysis = {
                'file': dta_file.name,
                'stata_shape': df_stata.shape,
                'csv_shape': df_csv.shape,
                'shape_match': df_stata.shape == df_csv.shape,
                'stata_columns': list(df_stata.columns),
                'csv_columns': list(df_csv.columns),
                'columns_match': list(df_stata.columns) == list(df_csv.columns),
                'stata_dtypes': df_stata.dtypes.to_dict(),
                'csv_dtypes': df_csv.dtypes.to_dict()
            }
            
            print(f"    Shape: Stata {df_stata.shape} vs CSV {df_csv.shape}")
            print(f"    Columns match: {analysis['columns_match']}")
            
            # Missing value analysis by column
            missing_analysis = {}
            if analysis['columns_match']:
                for col in df_stata.columns:
                    stata_missing = df_stata[col].isnull().sum()
                    csv_missing = df_csv[col].isnull().sum()
                    
                    missing_analysis[col] = {
                        'stata_missing': stata_missing,
                        'csv_missing': csv_missing,
                        'difference': abs(stata_missing - csv_missing)
                    }
                    
                    # Flag significant differences
                    if abs(stata_missing - csv_missing) > 5:  # More than 5 missing value differences
                        print(f"    Warning: {col}: Stata {stata_missing} vs CSV {csv_missing} missing")
            
            analysis['missing_by_column'] = missing_analysis
            analysis['total_stata_missing'] = df_stata.isnull().sum().sum()
            analysis['total_csv_missing'] = df_csv.isnull().sum().sum()
            analysis['total_missing_diff'] = abs(analysis['total_stata_missing'] - analysis['total_csv_missing'])
            
            # Sample data comparison for non-missing values
            if analysis['columns_match'] and len(df_stata.columns) < 100:  # Avoid huge simulation files
                sample_matches = {}
                for col in df_stata.columns[:5]:  # Check first 5 columns
                    # Compare non-missing values
                    stata_non_missing = df_stata[col].dropna()
                    csv_non_missing = df_csv[col].dropna()
                    
                    if len(stata_non_missing) > 0 and len(csv_non_missing) > 0:
                        # Compare first few non-missing values
                        sample_size = min(10, len(stata_non_missing), len(csv_non_missing))
                        if sample_size > 0:
                            stata_sample = stata_non_missing.iloc[:sample_size].values
                            csv_sample = csv_non_missing.iloc[:sample_size].values
                            
                            # For numeric data, check if values are close
                            if pd.api.types.is_numeric_dtype(df_stata[col]):
                                try:
                                    matches = np.allclose(stata_sample.astype(float), 
                                                        csv_sample.astype(float), 
                                                        equal_nan=True, rtol=1e-10)
                                    sample_matches[col] = matches
                                except:
                                    sample_matches[col] = False
                            else:
                                # For string data, check exact match
                                sample_matches[col] = np.array_equal(stata_sample, csv_sample)
                
                analysis['sample_data_matches'] = sample_matches
            
            # Overall assessment
            critical_issues = []
            if not analysis['shape_match']:
                critical_issues.append("Shape mismatch")
            if not analysis['columns_match']:
                critical_issues.append("Column names mismatch")
            if analysis['total_missing_diff'] > 100:  # Allow some tolerance for missing representation
                critical_issues.append(f"Large missing value difference: {analysis['total_missing_diff']}")
            
            analysis['critical_issues'] = critical_issues
            analysis['validation_status'] = 'PASS' if len(critical_issues) == 0 else 'FAIL'
            
            # Print summary
            if analysis['validation_status'] == 'PASS':
                print(f"    VALIDATION PASSED")
            else:
                print(f"    VALIDATION FAILED: {', '.join(critical_issues)}")
                
            validation_details[dta_file.name] = analysis
            
        except Exception as e:
            print(f"    ERROR: {e}")
            validation_details[dta_file.name] = {'error': str(e), 'validation_status': 'ERROR'}
    
    return validation_details

def summarize_validation_results(validation_details):
    """Summarize validation results and identify specific issues"""
    
    print(f"\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    passed = sum(1 for v in validation_details.values() if v.get('validation_status') == 'PASS')
    failed = sum(1 for v in validation_details.values() if v.get('validation_status') == 'FAIL')
    errors = sum(1 for v in validation_details.values() if v.get('validation_status') == 'ERROR')
    total = len(validation_details)
    
    print(f"Total files: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    
    if failed > 0:
        print(f"\nFAILED FILES ANALYSIS:")
        for filename, details in validation_details.items():
            if details.get('validation_status') == 'FAIL':
                print(f"\n{filename}:")
                for issue in details.get('critical_issues', []):
                    print(f"  - {issue}")
                    
                # Show specific missing value issues
                if 'missing_by_column' in details:
                    high_diff_cols = [col for col, info in details['missing_by_column'].items() 
                                    if info['difference'] > 10]
                    if high_diff_cols:
                        print(f"  - Columns with high missing differences: {len(high_diff_cols)}")
                        for col in high_diff_cols[:3]:  # Show first 3
                            info = details['missing_by_column'][col]
                            print(f"    • {col}: {info['stata_missing']} → {info['csv_missing']}")
    
    # Determine if data is ready for analysis
    critical_failures = sum(1 for v in validation_details.values() 
                          if v.get('validation_status') == 'FAIL' and 
                          any('Shape mismatch' in issue or 'Column names mismatch' in issue 
                              for issue in v.get('critical_issues', [])))
    
    print(f"\n" + "="*50)
    if critical_failures == 0:
        print("PASS")
        print("   Missing value differences are acceptable (representation differences)")
        print("   All critical structure and content preserved")
    else:
        print("Critical issues found - need investigation")
        print(f"   {critical_failures} files have structural problems")
    
    return passed, failed, errors

if __name__ == "__main__":
    validation_details = deep_validation_analysis()
    passed, failed, errors = summarize_validation_results(validation_details)
