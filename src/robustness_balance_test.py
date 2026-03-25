#!/usr/bin/env python3
"""
Balance Test for Blair & Weintraub (2023) Randomization
=======================================================

This script performs a robustness check to test whether the randomization 
successfully balanced observable characteristics across treatment groups.

The null hypothesis is that pre-treatment control variables have identical 
means across the three groups (Control, Spillover, Treatment). Rejection 
would suggest potential randomization issues.

Statistical Approach:
- F-test for joint significance across groups: control_variable ~ C(treatment)
- Small p-values (< 0.05) indicate potential imbalance
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from config import PATHS, DATA_FILES

def load_and_prepare_data():
    """Load and prepare data for balance testing."""
    
    # Load primary dataset with treatment assignments and control variables
    print("Loading admin_data_prior.csv...")
    prior_df = pd.read_csv(PATHS['data_raw'] / 'admin_data_prior.csv')
    
    # Load demographic data to merge in additional control variables
    print("Loading admin_data_during.csv...")
    during_df = pd.read_csv(PATHS['data_raw'] / 'admin_data_during.csv')
    
    # Merge to get demographic variables
    demographic_vars = ['manzana', 'block_age', 'block_educ', 'block_pct_male']
    merged_df = prior_df.merge(
        during_df[demographic_vars], 
        on='manzana', 
        how='left'
    )
    
    # Create treatment group variable (categorical)
    # From the data: control=1,treat=0,spillover=0 means control group
    # control=0,treat=1,spillover=0 means treatment group  
    # control=0,treat=0,spillover=1 means spillover group
    def assign_treatment_group(row):
        if row['control'] == 1:
            return 'Control'
        elif row['treat'] == 1:
            return 'Treatment'  
        elif row['spillover'] == 1:
            return 'Spillover'
        else:
            return 'Unknown'
    
    merged_df['treatment_group'] = merged_df.apply(assign_treatment_group, axis=1)
    
    print(f"\nDataset loaded: {len(merged_df)} blocks")
    print("Treatment group distribution:")
    print(merged_df['treatment_group'].value_counts())
    
    return merged_df

def define_control_variables():
    """Define the control variables to test for balance."""
    
    # Geographic and infrastructure variables
    geographic_vars = [
        'area',           # Block area
        'bat_min',        # Distance to battalion  
        'cai_min',        # Distance to police station
        'ptr_min',        # Distance to police patrol
        'number_buildings_sampling'  # Number of buildings
    ]
    
    # Baseline crime variables (pre-treatment)
    crime_vars = [
        'prior_homicides_num',
        'prior_robberies_num', 
        'prior_drugdeal_num',
        'prior_wbearing_num',
        'prior_unw_crime2_num'
    ]
    
    # Demographic variables (from during dataset)
    demographic_vars = [
        'block_age',
        'block_educ', 
        'block_pct_male'
    ]
    
    # Note: iweight excluded -- it is a function of treatment assignment
    # and barrio-level stratification, making a balance test circular.

    all_vars = geographic_vars + crime_vars + demographic_vars

    return {
        'geographic': geographic_vars,
        'crime': crime_vars,
        'demographic': demographic_vars,
        'all': all_vars
    }

def perform_balance_test(df, variable):
    """
    Perform F-test for balance of a single variable across treatment groups.
    
    Returns:
        dict: Contains F-statistic, p-value, and group means
    """
    
    # Remove missing values
    clean_df = df[['treatment_group', variable]].dropna()
    
    if len(clean_df) == 0:
        return {
            'variable': variable,
            'f_stat': np.nan,
            'p_value': np.nan,
            'n_obs': 0,
            'group_means': {},
            'balanced': 'No data'
        }
    
    # Run ANOVA: variable ~ treatment_group
    try:
        formula = f"{variable} ~ C(treatment_group)"
        model = smf.ols(formula, data=clean_df).fit()
        f_stat = model.fvalue
        p_value = model.f_pvalue
        
        # Calculate group means
        group_means = clean_df.groupby('treatment_group')[variable].mean().to_dict()
        
        # Determine if balanced (p > 0.05 suggests balance)
        balanced = "Yes" if p_value > 0.05 else "No"
        
        return {
            'variable': variable,
            'f_stat': f_stat,
            'p_value': p_value,
            'n_obs': len(clean_df),
            'group_means': group_means,
            'balanced': balanced
        }
        
    except Exception as e:
        print(f"Error testing {variable}: {e}")
        return {
            'variable': variable,
            'f_stat': np.nan,
            'p_value': np.nan,
            'n_obs': len(clean_df),
            'group_means': {},
            'balanced': 'Error'
        }

def run_balance_tests(df, control_vars):
    """Run balance tests for all control variables."""
    
    print("\n" + "="*60)
    print("RUNNING BALANCE TESTS")
    print("="*60)
    
    results = []
    
    for var_category, variables in control_vars.items():
        if var_category == 'all':  # Skip the combined list
            continue
            
        print(f"\n{var_category.upper()} VARIABLES:")
        print("-" * 30)
        
        for var in variables:
            if var in df.columns:
                result = perform_balance_test(df, var)
                results.append(result)
                
                # Print result
                print(f"{var:25s} | F={result['f_stat']:6.3f} | p={result['p_value']:6.3f} | {result['balanced']:8s} | n={result['n_obs']}")
            else:
                print(f"{var:25s} | MISSING FROM DATA")
    
    return results

def create_summary_table(results):
    """Create a summary table of balance test results."""
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Count balanced vs imbalanced
    balanced_count = sum(1 for r in results if r['balanced'] == 'Yes')
    imbalanced_count = sum(1 for r in results if r['balanced'] == 'No')
    total_tests = len([r for r in results if r['balanced'] in ['Yes', 'No']])
    
    print("\n" + "="*60)
    print("BALANCE TEST SUMMARY")
    print("="*60)
    print(f"Total variables tested: {total_tests}")
    print(f"Balanced variables (p > 0.05): {balanced_count}")  
    print(f"Imbalanced variables (p ≤ 0.05): {imbalanced_count}")
    print(f"Balance rate: {balanced_count/total_tests*100:.1f}%")
    
    # Show imbalanced variables if any
    imbalanced = [r for r in results if r['balanced'] == 'No']
    if imbalanced:
        print(f"\nIMBALANCED VARIABLES (p ≤ 0.05):")
        print("-" * 40)
        for r in imbalanced:
            print(f"{r['variable']:25s} | p={r['p_value']:.4f}")
    else:
        print("\n✓ ALL VARIABLES ARE BALANCED!")
    
    return results_df

def display_group_means(results, df):
    """Display means by treatment group for key variables."""
    
    print("\n" + "="*80)
    print("GROUP MEANS FOR KEY VARIABLES")
    print("="*80)
    
    # Select a few key variables to display means
    key_vars = ['area', 'prior_unw_crime2_num', 'block_age', 'block_educ', 'number_buildings_sampling']
    
    for var in key_vars:
        var_result = next((r for r in results if r['variable'] == var), None)
        if var_result and var_result['group_means']:
            print(f"\n{var}:")
            for group, mean in var_result['group_means'].items():
                print(f"  {group:12s}: {mean:8.3f}")

def main():
    """Main function to run the balance test."""
    
    print("="*60)
    print("BLAIR & WEINTRAUB (2023) RANDOMIZATION BALANCE TEST") 
    print("="*60)
    print("\nThis test checks whether observable characteristics are")
    print("balanced across treatment groups, as expected under")
    print("successful randomization.")
    
    # Load data
    df = load_and_prepare_data()
    
    # Define control variables
    control_vars = define_control_variables()
    
    print(f"\nTesting {len(control_vars['all'])} control variables...")
    
    # Run balance tests
    results = run_balance_tests(df, control_vars)
    
    # Create summary
    results_df = create_summary_table(results)
    
    # Display group means for key variables
    display_group_means(results, df)
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("• Balanced variables (p > 0.05): Randomization successfully")
    print("  created similar groups for these characteristics")  
    print("• Imbalanced variables (p ≤ 0.05): May indicate randomization")
    print("  issues or chance imbalances that could affect results")
    print("• Expected: ~5% of tests should be significant by chance")
    
    # Calculate expected vs actual significance rate
    total_valid_tests = len([r for r in results if r['balanced'] in ['Yes', 'No']])
    actual_significant = len([r for r in results if r['balanced'] == 'No'])
    expected_significant = total_valid_tests * 0.05
    
    print(f"\nExpected significant tests by chance: {expected_significant:.1f}")
    print(f"Actual significant tests: {actual_significant}")
    
    if actual_significant <= expected_significant * 2:  # Allow some margin
        print("✓ CONCLUSION: Randomization appears successful")
    else:
        print("⚠ CONCLUSION: Potential randomization concerns")
    
    return results_df

if __name__ == "__main__":
    results = main()
