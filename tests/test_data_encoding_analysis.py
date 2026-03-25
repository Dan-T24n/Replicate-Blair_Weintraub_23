"""
test_data_encoding_analysis.py

Detailed analysis of data encoding and variable handling differences
Blair & Weintraub (2023) Military Policing Replication

Purpose:
- Examine treatment variable encoding (int vs float)
- Check missing value patterns
- Verify control variable availability and encoding
- Compare sample sizes and data filtering

Authors: Python Replication Team  
Date: 2024
"""

#%%
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config import PATHS

#%%
def examine_treatment_encoding():
    """Examine treatment variable encoding across datasets"""
    print("="*60)
    print("TREATMENT VARIABLE ENCODING ANALYSIS")
    print("="*60)
    
    datasets = {
        'admin_during': PATHS['data_raw'] / 'admin_data_during.csv',
        'admin_after': PATHS['data_raw'] / 'admin_data_after.csv',
        'survey_endline': PATHS['data_raw'] / 'survey_endline.csv'
    }
    
    for name, path in datasets.items():
        print(f"\n{name.upper()}:")
        df = pd.read_csv(path)
        
        print(f"  Treatment variable type: {df['treatment'].dtype}")
        print(f"  Treatment unique values: {sorted(df['treatment'].unique())}")
        print(f"  Treatment value counts:")
        print(f"    {df['treatment'].value_counts().sort_index().to_dict()}")
        
        # Check for any non-integer values
        if df['treatment'].dtype == 'float64':
            non_int_mask = df['treatment'] != df['treatment'].astype(int)
            non_int_count = non_int_mask.sum()
            print(f"  Non-integer treatment values: {non_int_count}")
            if non_int_count > 0:
                print(f"    Examples: {df.loc[non_int_mask, 'treatment'].head().tolist()}")

#%%
def examine_control_variables():
    """Examine control variables availability and missing patterns"""
    print("\n" + "="*60)
    print("CONTROL VARIABLES ANALYSIS")
    print("="*60)
    
    # Expected control variables from Stata code
    geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
    blockdemovars = ['block_age', 'block_educ', 'block_pct_male']
    demovars = ['age', 'gender', 'educ']
    
    datasets = {
        'admin_during': {
            'path': PATHS['data_raw'] / 'admin_data_during.csv',
            'expected_vars': geovars + blockdemovars + ['cum_all_unw_crime2_num'],
            'outcome': 'unw_crime2_num'
        },
        'admin_after': {
            'path': PATHS['data_raw'] / 'admin_data_after.csv',
            'expected_vars': geovars + blockdemovars + ['cum_all_unw_crime_num'],
            'outcome': 'unw_crime_num'
        },
        'survey_endline': {
            'path': PATHS['data_raw'] / 'survey_endline.csv',
            'expected_vars': geovars + demovars,
            'outcome': 'i2_victimduringindex_std'
        }
    }
    
    for name, info in datasets.items():
        print(f"\n{name.upper()}:")
        df = pd.read_csv(info['path'])
        
        print(f"  Total observations: {len(df)}")
        print(f"  Total variables: {len(df.columns)}")
        
        # Check availability of expected variables
        missing_vars = [var for var in info['expected_vars'] if var not in df.columns]
        if missing_vars:
            print(f"  ⚠️  Missing expected variables: {missing_vars}")
        else:
            print(f"  ✓ All expected control variables present")
        
        # Check missing value patterns for key variables
        key_vars = ['treatment', 'barrio_code', 'iweight', info['outcome']] + info['expected_vars']
        key_vars = [var for var in key_vars if var in df.columns]
        
        print(f"  Missing value patterns:")
        missing_summary = df[key_vars].isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        
        if len(missing_summary) > 0:
            for var, count in missing_summary.items():
                pct = (count / len(df)) * 100
                print(f"    {var}: {count} ({pct:.1f}%)")
        else:
            print(f"    No missing values in key variables")

#%%
def examine_sample_selection():
    """Examine how sample selection affects final regression samples"""
    print("\n" + "="*60)
    print("SAMPLE SELECTION ANALYSIS")
    print("="*60)
    
    # Admin during data
    print("\nADMIN DURING (Column 1):")
    df = pd.read_csv(PATHS['data_raw'] / 'admin_data_during.csv')
    print(f"  Raw observations: {len(df)}")
    
    essential_vars = ['unw_crime2_num', 'treatment', 'barrio_code', 'iweight']
    df_clean = df.dropna(subset=essential_vars)
    print(f"  After dropping missing essential vars: {len(df_clean)} (dropped {len(df) - len(df_clean)})")
    
    # Check treatment group sizes
    print(f"  Treatment group sizes:")
    for treatment, count in df_clean['treatment'].value_counts().sort_index().items():
        print(f"    Treatment {treatment}: {count}")
    
    # Admin after data  
    print("\nADMIN AFTER (Column 2):")
    df = pd.read_csv(PATHS['data_raw'] / 'admin_data_after.csv')
    print(f"  Raw observations: {len(df)}")
    
    essential_vars = ['unw_crime_num', 'treatment', 'barrio_code', 'iweight']
    df_clean = df.dropna(subset=essential_vars)
    print(f"  After dropping missing essential vars: {len(df_clean)} (dropped {len(df) - len(df_clean)})")
    
    # Survey endline data
    print("\nSURVEY ENDLINE (Columns 3-5):")
    df = pd.read_csv(PATHS['data_raw'] / 'survey_endline.csv')
    print(f"  Raw observations: {len(df)}")
    
    # Check different outcome variables
    outcomes = ['i2_victimduringindex_std', 'i2_victimafterindex_std', 'i_witnessindex_std']
    
    for outcome in outcomes:
        if outcome in df.columns:
            essential_vars = [outcome, 'treatment', 'barrio_code', 'iweight', 'manzana_code']
            df_clean = df.dropna(subset=essential_vars)
            print(f"  {outcome}: {len(df_clean)} obs (dropped {len(df) - len(df_clean)})")
        else:
            print(f"  ⚠️  {outcome}: Variable not found")

#%%
def examine_weight_distributions():
    """Examine inverse probability weight distributions"""
    print("\n" + "="*60)
    print("INVERSE PROBABILITY WEIGHTS ANALYSIS")  
    print("="*60)
    
    datasets = {
        'admin_during': PATHS['data_raw'] / 'admin_data_during.csv',
        'admin_after': PATHS['data_raw'] / 'admin_data_after.csv',
        'survey_endline': PATHS['data_raw'] / 'survey_endline.csv'
    }
    
    for name, path in datasets.items():
        print(f"\n{name.upper()}:")
        df = pd.read_csv(path)
        
        if 'iweight' in df.columns:
            weights = df['iweight'].dropna()
            print(f"  Weight statistics:")
            print(f"    Count: {len(weights)}")
            print(f"    Mean: {weights.mean():.6f}")
            print(f"    Std: {weights.std():.6f}")
            print(f"    Min: {weights.min():.6f}")
            print(f"    Max: {weights.max():.6f}")
            
            # Check weight distribution by treatment group
            print(f"  Weight by treatment group:")
            weight_by_treatment = df.groupby('treatment')['iweight'].agg(['count', 'mean', 'std'])
            print(weight_by_treatment)
        else:
            print(f"  ⚠️  iweight variable not found")

#%%
def examine_outcome_distributions():
    """Examine outcome variable distributions"""
    print("\n" + "="*60)
    print("OUTCOME VARIABLE DISTRIBUTIONS")
    print("="*60)
    
    # Admin outcomes
    admin_outcomes = {
        'admin_during': ('unw_crime2_num', PATHS['data_raw'] / 'admin_data_during.csv'),
        'admin_after': ('unw_crime_num', PATHS['data_raw'] / 'admin_data_after.csv')
    }
    
    for name, (outcome_var, path) in admin_outcomes.items():
        print(f"\n{name.upper()} - {outcome_var}:")
        df = pd.read_csv(path)
        
        if outcome_var in df.columns:
            outcome = df[outcome_var].dropna()
            print(f"  Count: {len(outcome)}")
            print(f"  Mean: {outcome.mean():.6f}")
            print(f"  Std: {outcome.std():.6f}")
            print(f"  Min: {outcome.min():.6f}")
            print(f"  Max: {outcome.max():.6f}")
            
            # By treatment group
            print(f"  By treatment group:")
            outcome_by_treatment = df.groupby('treatment')[outcome_var].agg(['count', 'mean', 'std'])
            print(outcome_by_treatment)
        else:
            print(f"  ⚠️  {outcome_var} not found")
    
    # Survey outcomes
    print(f"\nSURVEY ENDLINE:")
    df = pd.read_csv(PATHS['data_raw'] / 'survey_endline.csv')
    
    survey_outcomes = ['i2_victimduringindex_std', 'i2_victimafterindex_std', 'i_witnessindex_std']
    
    for outcome_var in survey_outcomes:
        if outcome_var in df.columns:
            print(f"\n  {outcome_var}:")
            outcome = df[outcome_var].dropna()
            print(f"    Count: {len(outcome)}")
            print(f"    Mean: {outcome.mean():.6f}")
            print(f"    Std: {outcome.std():.6f}")
            
            # Control group mean (this is reported in the table)
            control_mean = df[df['treatment'] == 0][outcome_var].mean()
            print(f"    Control group mean: {control_mean:.6f}")
        else:
            print(f"  ⚠️  {outcome_var} not found")

#%%
def main():
    """Run complete data encoding analysis"""
    print("Starting data encoding analysis...")
    
    examine_treatment_encoding()
    examine_control_variables()
    examine_sample_selection()
    examine_weight_distributions()
    examine_outcome_distributions()
    
    print("\n" + "="*60)
    print("DATA ENCODING ANALYSIS COMPLETE")
    print("="*60)

#%%
if __name__ == "__main__":
    main()
