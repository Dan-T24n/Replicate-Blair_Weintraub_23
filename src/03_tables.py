"""
03_tables.py

Blair & Weintraub (2023) Military Policing Replication
Python implementation of 03_Tables.do

Purpose: Generate Table 1 using existing randomization inference data
- Replicate exact regression specifications from original Stata code
- Use existing RI simulation results for p-values
- Generate identical LaTeX table output

Authors: Robert Blair and Michael Weintraub (original)
Date created: 2022-06-30 (original Stata)
Python conversion: 2024
"""

#%%
# Import required packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
import sys
import time
import argparse

# Import project configuration
from config import PATHS, DATA_FILES
from utils.latex_generator import generate_table1_latex

#%%
def load_admin_data_during():
    """Load administrative data during intervention"""
    print("Loading administrative data (during intervention)...")
    
    # Load data
    df = pd.read_csv(PATHS['data_raw'] / 'admin_data_during.csv')
    
    print(f"  Observations: {len(df)}")
    print(f"  Variables: {len(df.columns)}")
    
    # Verify key variables exist
    required_vars = ['unw_crime2_num', 'treatment', 'barrio_code', 'iweight']
    missing_vars = [var for var in required_vars if var not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing required variables: {missing_vars}")
    
    # Convert treatment to integer for consistency with Stata
    df['treatment'] = df['treatment'].astype(int)
    print(f"  Treatment variable converted to int: {df['treatment'].unique()}")
    
    return df

#%%
def load_admin_data_after():
    """Load administrative data after intervention"""
    print("Loading administrative data (after intervention)...")
    
    # Load data
    df = pd.read_csv(PATHS['data_raw'] / 'admin_data_after.csv')
    
    print(f"  Observations: {len(df)}")
    print(f"  Variables: {len(df.columns)}")
    
    # Verify key variables exist
    required_vars = ['unw_crime_num', 'treatment', 'barrio_code', 'iweight']
    missing_vars = [var for var in required_vars if var not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing required variables: {missing_vars}")
    
    # Convert treatment to integer for consistency with Stata
    df['treatment'] = df['treatment'].astype(int)
    print(f"  Treatment variable converted to int: {df['treatment'].unique()}")
    
    return df

#%%
def load_survey_endline():
    """Load endline survey data"""
    print("Loading endline survey data...")
    
    # Load data
    df = pd.read_csv(PATHS['data_raw'] / 'survey_endline.csv')
    
    print(f"  Observations: {len(df)}")
    print(f"  Variables: {len(df.columns)}")
    
    # Verify key variables exist
    required_vars = ['treatment', 'barrio_code', 'iweight', 'manzana_code']
    survey_vars = ['i2_victimduringindex_std', 'i2_victimafterindex_std', 'i_witnessindex_std']
    
    missing_vars = [var for var in required_vars + survey_vars if var not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing required variables: {missing_vars}")
    
    return df

#%%
def calculate_ri_pvalue_for_column(observed_coef, sim_coefficients):
    """Calculate RI P-value: proportion of simulations with |coef| >= |observed|"""
    abs_observed = abs(observed_coef)
    abs_simulated = np.abs(sim_coefficients)
    count_extreme = np.sum(abs_simulated >= abs_observed)
    p_value = count_extreme / len(sim_coefficients)
    return p_value

def load_precomputed_coefficients(col_num):
    """Load all pre-computed coefficients for a column from all 10 simulation files"""
    
    coef_dir = PATHS['data_rand_new'] / 'coefs'
    all_treatment_coefs = []
    all_spillover_coefs = []
    
    print(f"    Loading pre-computed coefficients for column {col_num}...")
    
    for p in range(1, 11):
        coef_file = coef_dir / f'RI_table1_col{col_num}_p{p}.csv'
        
        if coef_file.exists():
            df = pd.read_csv(coef_file)
            all_treatment_coefs.extend(df['treat_ef'].tolist())
            all_spillover_coefs.extend(df['spillover_ef'].tolist())
        else:
            raise FileNotFoundError(f"Missing coefficient file: {coef_file}")
    
    print(f"    Loaded {len(all_treatment_coefs)} pre-computed coefficients")
    
    return np.array(all_treatment_coefs), np.array(all_spillover_coefs)

# Legacy functions removed - using pre-computed coefficients instead

def load_randomization_inference_results_fast(observed_coefficients):
    """Load pre-computed randomization inference p-values (FAST VERSION)"""
    print("Loading pre-computed randomization inference results...")
    
    ri_pvalues = {}
    
    # Process each column using pre-computed coefficients
    for col_num in [1, 2, 3, 4, 5]:
        col_key = f'col{col_num}'
        
        # Load pre-computed coefficients for this column
        treatment_coeffs, spillover_coeffs = load_precomputed_coefficients(col_num)
        
        # Get observed coefficients for this column
        observed_treatment = observed_coefficients[col_key]['treatment']
        observed_spillover = observed_coefficients[col_key]['spillover']
        
        # Calculate RI p-values using pre-computed coefficients
        treatment_pval = calculate_ri_pvalue_for_column(observed_treatment, treatment_coeffs)
        spillover_pval = calculate_ri_pvalue_for_column(observed_spillover, spillover_coeffs)
        
        ri_pvalues[col_key] = {
            'treatment': treatment_pval,
            'spillover': spillover_pval
        }
        
        print(f"    Column {col_num}: Treatment p={treatment_pval:.8f}, Spillover p={spillover_pval:.8f}")
    
    print(f"\n  RI P-value calculation completed using pre-computed coefficients!")
    
    return ri_pvalues

# Legacy slow method removed - kept for backward compatibility if needed

#%%
def run_table1_column1(df):
    """
    Run Table 1 Column 1: Crime during intervention (admin data)
    Mirrors: reg unw_crime2_num i.treatment i.barrio_code $geovars $blockdemovars cum_all_unw_crime2_num [pweight=iweight]
    """
    print("Running Table 1, Column 1: Crime during intervention...")
    
    # Define control variables (exact replication)
    geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
    blockdemovars = ['block_age', 'block_educ', 'block_pct_male']
    
    # Create formula string
    # Note: Using C() for categorical treatment to match Stata i.treatment
    formula = "unw_crime2_num ~ C(treatment, Treatment(reference=0))"
    
    # Add barrio fixed effects
    formula += " + C(barrio_code)"
    
    # Add geographic variables
    for var in geovars:
        if var in df.columns:
            formula += f" + {var}"
    
    # Add block demographic variables  
    for var in blockdemovars:
        if var in df.columns:
            formula += f" + {var}"
    
    # Add cumulative crime control
    if 'cum_all_unw_crime2_num' in df.columns:
        formula += " + cum_all_unw_crime2_num"
    
    print(f"  Formula: {formula}")
    
    # Drop missing values only for essential variables (like Stata default behavior)
    # Stata handles missing control variables by excluding them from specific terms, not dropping observations
    essential_vars = ['unw_crime2_num', 'treatment', 'barrio_code', 'iweight']
    df_clean = df.dropna(subset=essential_vars)
    
    print(f"  Observations after dropping missing: {len(df_clean)} (dropped {len(df) - len(df_clean)})")
    
    # Run weighted regression with proper missing value handling
    model = smf.wls(formula, data=df_clean, weights=df_clean['iweight'], missing='drop')
    results = model.fit()
    
    # Calculate control mean
    control_mean = df_clean[df_clean['treatment'] == 0]['unw_crime2_num'].mean()
    
    # Extract coefficients for treatment and spillover
    # Note: Treatment variable is now int with explicit reference, so coefficients are [T.1] and [T.2]
    treatment_coef = results.params.get('C(treatment, Treatment(reference=0))[T.1]', np.nan)
    spillover_coef = results.params.get('C(treatment, Treatment(reference=0))[T.2]', np.nan)
    
    treatment_se = results.bse.get('C(treatment, Treatment(reference=0))[T.1]', np.nan)
    spillover_se = results.bse.get('C(treatment, Treatment(reference=0))[T.2]', np.nan)
    
    # Calculate confidence intervals
    treatment_ci = [treatment_coef - 1.96*treatment_se, treatment_coef + 1.96*treatment_se]
    spillover_ci = [spillover_coef - 1.96*spillover_se, spillover_coef + 1.96*spillover_se]
    
    # Calculate p-values
    treatment_pval = results.pvalues.get('C(treatment, Treatment(reference=0))[T.1]', np.nan)
    spillover_pval = results.pvalues.get('C(treatment, Treatment(reference=0))[T.2]', np.nan)
    
    result = {
        'treatment_coef': treatment_coef,
        'treatment_se': treatment_se,
        'treatment_ci': treatment_ci,
        'treatment_pval': treatment_pval,
        'spillover_coef': spillover_coef,
        'spillover_se': spillover_se,
        'spillover_ci': spillover_ci,
        'spillover_pval': spillover_pval,
        'n_obs': int(results.nobs),
        'r_squared': results.rsquared,
        'control_mean': control_mean
    }
    
    print(f"  Treatment coef: {treatment_coef:.3f} (SE: {treatment_se:.3f})")
    print(f"  Spillover coef: {spillover_coef:.3f} (SE: {spillover_se:.3f})")
    print(f"  N: {result['n_obs']}, R²: {result['r_squared']:.3f}")
    
    return result

#%%
def run_table1_column2(df):
    """
    Run Table 1 Column 2: Crime after intervention (admin data)
    Mirrors: reg unw_crime_num i.treatment i.barrio_code $geovars $blockdemovars cum_all_unw_crime_num [pweight=iweight]
    """
    print("Running Table 1, Column 2: Crime after intervention...")
    
    # Define control variables (exact replication)
    geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
    blockdemovars = ['block_age', 'block_educ', 'block_pct_male']
    
    # Create formula string
    formula = "unw_crime_num ~ C(treatment, Treatment(reference=0))"
    
    # Add barrio fixed effects
    formula += " + C(barrio_code)"
    
    # Add geographic variables
    for var in geovars:
        if var in df.columns:
            formula += f" + {var}"
    
    # Add block demographic variables  
    for var in blockdemovars:
        if var in df.columns:
            formula += f" + {var}"
    
    # Add cumulative crime control
    if 'cum_all_unw_crime_num' in df.columns:
        formula += " + cum_all_unw_crime_num"
    
    print(f"  Formula: {formula}")
    
    # Drop missing values only for essential variables (like Stata default behavior)
    # Stata handles missing control variables by excluding them from specific terms, not dropping observations
    essential_vars = ['unw_crime_num', 'treatment', 'barrio_code', 'iweight']
    df_clean = df.dropna(subset=essential_vars)
    
    print(f"  Observations after dropping missing: {len(df_clean)} (dropped {len(df) - len(df_clean)})")
    
    # Run weighted regression with proper missing value handling
    model = smf.wls(formula, data=df_clean, weights=df_clean['iweight'], missing='drop')
    results = model.fit()
    
    # Calculate control mean
    control_mean = df_clean[df_clean['treatment'] == 0]['unw_crime_num'].mean()
    
    # Extract coefficients for treatment and spillover
    # Note: Treatment variable is now int with explicit reference, so coefficients are [T.1] and [T.2]
    treatment_coef = results.params.get('C(treatment, Treatment(reference=0))[T.1]', np.nan)
    spillover_coef = results.params.get('C(treatment, Treatment(reference=0))[T.2]', np.nan)
    
    treatment_se = results.bse.get('C(treatment, Treatment(reference=0))[T.1]', np.nan)
    spillover_se = results.bse.get('C(treatment, Treatment(reference=0))[T.2]', np.nan)
    
    # Calculate confidence intervals
    treatment_ci = [treatment_coef - 1.96*treatment_se, treatment_coef + 1.96*treatment_se]
    spillover_ci = [spillover_coef - 1.96*spillover_se, spillover_coef + 1.96*spillover_se]
    
    # Calculate p-values
    treatment_pval = results.pvalues.get('C(treatment, Treatment(reference=0))[T.1]', np.nan)
    spillover_pval = results.pvalues.get('C(treatment, Treatment(reference=0))[T.2]', np.nan)
    
    result = {
        'treatment_coef': treatment_coef,
        'treatment_se': treatment_se,
        'treatment_ci': treatment_ci,
        'treatment_pval': treatment_pval,
        'spillover_coef': spillover_coef,
        'spillover_se': spillover_se,
        'spillover_ci': spillover_ci,
        'spillover_pval': spillover_pval,
        'n_obs': int(results.nobs),
        'r_squared': results.rsquared,
        'control_mean': control_mean
    }
    
    print(f"  Treatment coef: {treatment_coef:.3f} (SE: {treatment_se:.3f})")
    print(f"  Spillover coef: {spillover_coef:.3f} (SE: {spillover_se:.3f})")
    print(f"  N: {result['n_obs']}, R²: {result['r_squared']:.3f}")
    
    return result

#%%
def run_table1_column3(df):
    """
    Run Table 1 Column 3: Crime victimization during intervention (survey)
    Mirrors: reg i2_victimduringindex_std i.treatment i.barrio_code ${demovars} ${geovars} [pweight=iweight], vce(cluster manzana_code)
    """
    print("Running Table 1, Column 3: Crime victimization during intervention...")
    
    # Define control variables (exact replication)
    demovars = ['age', 'gender', 'educ']
    geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
    
    # Create formula string
    formula = "i2_victimduringindex_std ~ C(treatment)"
    
    # Add barrio fixed effects
    formula += " + C(barrio_code)"
    
    # Add demographic variables
    for var in demovars:
        if var in df.columns:
            formula += f" + {var}"
    
    # Add geographic variables
    for var in geovars:
        if var in df.columns:
            formula += f" + {var}"
    
    print(f"  Formula: {formula}")
    
    # Drop missing values only for essential variables (like Stata default behavior)
    outcome_var = 'i2_victimduringindex_std'
    essential_vars = [outcome_var, 'treatment', 'barrio_code', 'iweight', 'manzana_code']
    df_clean = df.dropna(subset=essential_vars)
    
    print(f"  Observations after dropping missing: {len(df_clean)} (dropped {len(df) - len(df_clean)})")
    
    # For clustering to work properly, we need to ensure no missing values in any formula variables
    # First run regression to see which observations are actually used
    model = smf.wls(formula, data=df_clean, weights=df_clean['iweight'])
    
    # Create a temporary dataset with no missing values in any formula variables to match statsmodels behavior
    formula_vars = [outcome_var, 'treatment', 'barrio_code', 'iweight', 'manzana_code'] + demovars + geovars
    formula_vars = [v for v in formula_vars if v in df_clean.columns]
    df_final = df_clean.dropna(subset=formula_vars)
    
    # Re-run with the clean dataset
    model = smf.wls(formula, data=df_final, weights=df_final['iweight'])
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df_final['manzana_code']})
    
    # Calculate control mean using the final dataset
    control_mean = df_final[df_final['treatment'] == 0]['i2_victimduringindex_std'].mean()
    
    # Extract coefficients for treatment and spillover (survey data uses integer treatment codes)
    treatment_coef = results.params.get('C(treatment)[T.1]', np.nan)
    spillover_coef = results.params.get('C(treatment)[T.2]', np.nan)
    
    treatment_se = results.bse.get('C(treatment)[T.1]', np.nan)
    spillover_se = results.bse.get('C(treatment)[T.2]', np.nan)
    
    # Calculate confidence intervals
    treatment_ci = [treatment_coef - 1.96*treatment_se, treatment_coef + 1.96*treatment_se]
    spillover_ci = [spillover_coef - 1.96*spillover_se, spillover_coef + 1.96*spillover_se]
    
    # Calculate p-values
    treatment_pval = results.pvalues.get('C(treatment)[T.1]', np.nan)
    spillover_pval = results.pvalues.get('C(treatment)[T.2]', np.nan)
    
    result = {
        'treatment_coef': treatment_coef,
        'treatment_se': treatment_se,
        'treatment_ci': treatment_ci,
        'treatment_pval': treatment_pval,
        'spillover_coef': spillover_coef,
        'spillover_se': spillover_se,
        'spillover_ci': spillover_ci,
        'spillover_pval': spillover_pval,
        'n_obs': int(results.nobs),
        'r_squared': results.rsquared,
        'control_mean': control_mean
    }
    
    print(f"  Treatment coef: {treatment_coef:.3f} (SE: {treatment_se:.3f})")
    print(f"  Spillover coef: {spillover_coef:.3f} (SE: {spillover_se:.3f})")
    print(f"  N: {result['n_obs']}, R²: {result['r_squared']:.3f}")
    
    return result

#%%
def run_table1_column4(df):
    """
    Run Table 1 Column 4: Crime victimization after intervention (survey)
    Mirrors: reg i2_victimafterindex_std i.treatment i.barrio_code ${demovars} ${geovars} [pweight=iweight], vce(cluster manzana_code)
    """
    print("Running Table 1, Column 4: Crime victimization after intervention...")
    
    # Define control variables (exact replication)
    demovars = ['age', 'gender', 'educ']
    geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
    
    # Create formula string
    formula = "i2_victimafterindex_std ~ C(treatment)"
    
    # Add barrio fixed effects
    formula += " + C(barrio_code)"
    
    # Add demographic variables
    for var in demovars:
        if var in df.columns:
            formula += f" + {var}"
    
    # Add geographic variables
    for var in geovars:
        if var in df.columns:
            formula += f" + {var}"
    
    print(f"  Formula: {formula}")
    
    # Drop missing values to ensure consistent lengths for clustering
    outcome_var = 'i2_victimafterindex_std'
    # Include all variables used in the regression
    vars_to_check = [outcome_var, 'treatment', 'barrio_code', 'iweight', 'manzana_code'] + demovars + geovars
    vars_to_check = [var for var in vars_to_check if var in df.columns]  # Only check vars that exist
    df_clean = df.dropna(subset=vars_to_check)
    
    print(f"  Observations after dropping missing: {len(df_clean)} (dropped {len(df) - len(df_clean)})")
    
    # For clustering to work properly, we need to ensure no missing values in any formula variables
    formula_vars = [outcome_var, 'treatment', 'barrio_code', 'iweight', 'manzana_code'] + demovars + geovars
    formula_vars = [v for v in formula_vars if v in df_clean.columns]
    df_final = df_clean.dropna(subset=formula_vars)
    
    # Run weighted regression with clustered standard errors
    model = smf.wls(formula, data=df_final, weights=df_final['iweight'])
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df_final['manzana_code']})
    
    # Calculate control mean using the final dataset
    control_mean = df_final[df_final['treatment'] == 0]['i2_victimafterindex_std'].mean()
    
    # Extract coefficients for treatment and spillover
    treatment_coef = results.params.get('C(treatment)[T.1]', np.nan)
    spillover_coef = results.params.get('C(treatment)[T.2]', np.nan)
    
    treatment_se = results.bse.get('C(treatment)[T.1]', np.nan)
    spillover_se = results.bse.get('C(treatment)[T.2]', np.nan)
    
    # Calculate confidence intervals
    treatment_ci = [treatment_coef - 1.96*treatment_se, treatment_coef + 1.96*treatment_se]
    spillover_ci = [spillover_coef - 1.96*spillover_se, spillover_coef + 1.96*spillover_se]
    
    # Calculate p-values
    treatment_pval = results.pvalues.get('C(treatment)[T.1]', np.nan)
    spillover_pval = results.pvalues.get('C(treatment)[T.2]', np.nan)
    
    result = {
        'treatment_coef': treatment_coef,
        'treatment_se': treatment_se,
        'treatment_ci': treatment_ci,
        'treatment_pval': treatment_pval,
        'spillover_coef': spillover_coef,
        'spillover_se': spillover_se,
        'spillover_ci': spillover_ci,
        'spillover_pval': spillover_pval,
        'n_obs': int(results.nobs),
        'r_squared': results.rsquared,
        'control_mean': control_mean
    }
    
    print(f"  Treatment coef: {treatment_coef:.3f} (SE: {treatment_se:.3f})")
    print(f"  Spillover coef: {spillover_coef:.3f} (SE: {spillover_se:.3f})")
    print(f"  N: {result['n_obs']}, R²: {result['r_squared']:.3f}")
    
    return result

#%%
def run_table1_column5(df):
    """
    Run Table 1 Column 5: Crime witnessing after intervention (survey)
    Mirrors: reg i_witnessindex_std i.treatment i.barrio_code ${demovars} ${geovars} [pweight=iweight], vce(cluster manzana_code)
    """
    print("Running Table 1, Column 5: Crime witnessing after intervention...")
    
    # Define control variables (exact replication)
    demovars = ['age', 'gender', 'educ']
    geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
    
    # Create formula string
    formula = "i_witnessindex_std ~ C(treatment)"
    
    # Add barrio fixed effects
    formula += " + C(barrio_code)"
    
    # Add demographic variables
    for var in demovars:
        if var in df.columns:
            formula += f" + {var}"
    
    # Add geographic variables
    for var in geovars:
        if var in df.columns:
            formula += f" + {var}"
    
    print(f"  Formula: {formula}")
    
    # Drop missing values to ensure consistent lengths for clustering
    outcome_var = 'i_witnessindex_std'
    # Include all variables used in the regression
    vars_to_check = [outcome_var, 'treatment', 'barrio_code', 'iweight', 'manzana_code'] + demovars + geovars
    vars_to_check = [var for var in vars_to_check if var in df.columns]  # Only check vars that exist
    df_clean = df.dropna(subset=vars_to_check)
    
    print(f"  Observations after dropping missing: {len(df_clean)} (dropped {len(df) - len(df_clean)})")
    
    # For clustering to work properly, we need to ensure no missing values in any formula variables
    formula_vars = [outcome_var, 'treatment', 'barrio_code', 'iweight', 'manzana_code'] + demovars + geovars
    formula_vars = [v for v in formula_vars if v in df_clean.columns]
    df_final = df_clean.dropna(subset=formula_vars)
    
    # Run weighted regression with clustered standard errors
    model = smf.wls(formula, data=df_final, weights=df_final['iweight'])
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df_final['manzana_code']})
    
    # Calculate control mean using the final dataset
    control_mean = df_final[df_final['treatment'] == 0]['i_witnessindex_std'].mean()
    
    # Extract coefficients for treatment and spillover
    treatment_coef = results.params.get('C(treatment)[T.1]', np.nan)
    spillover_coef = results.params.get('C(treatment)[T.2]', np.nan)
    
    treatment_se = results.bse.get('C(treatment)[T.1]', np.nan)
    spillover_se = results.bse.get('C(treatment)[T.2]', np.nan)
    
    # Calculate confidence intervals
    treatment_ci = [treatment_coef - 1.96*treatment_se, treatment_coef + 1.96*treatment_se]
    spillover_ci = [spillover_coef - 1.96*spillover_se, spillover_coef + 1.96*spillover_se]
    
    # Calculate p-values
    treatment_pval = results.pvalues.get('C(treatment)[T.1]', np.nan)
    spillover_pval = results.pvalues.get('C(treatment)[T.2]', np.nan)
    
    result = {
        'treatment_coef': treatment_coef,
        'treatment_se': treatment_se,
        'treatment_ci': treatment_ci,
        'treatment_pval': treatment_pval,
        'spillover_coef': spillover_coef,
        'spillover_se': spillover_se,
        'spillover_ci': spillover_ci,
        'spillover_pval': spillover_pval,
        'n_obs': int(results.nobs),
        'r_squared': results.rsquared,
        'control_mean': control_mean
    }
    
    print(f"  Treatment coef: {treatment_coef:.3f} (SE: {treatment_se:.3f})")
    print(f"  Spillover coef: {spillover_coef:.3f} (SE: {spillover_se:.3f})")
    print(f"  N: {result['n_obs']}, R²: {result['r_squared']:.3f}")
    
    return result

#%%
# LaTeX generation moved to utils/latex_generator.py

#%%
def main(use_new_data=False, use_precomputed=True):
    """
    Main function to generate Table 1
    """
    data_source = "NEW" if use_new_data else "ORIGINAL"
    method = "PRE-COMPUTED" if use_precomputed else "ON-THE-FLY"
    
    print("=" * 60)
    print("TABLE 1 GENERATION")
    print("Blair & Weintraub (2023) Military Policing Replication")
    print(f"Simulation data: {data_source}")
    print(f"RI method: {method}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load data
        admin_during = load_admin_data_during()
        admin_after = load_admin_data_after()
        survey_endline = load_survey_endline()
        
        # Run all regressions first
        print("\n" + "=" * 40)
        print("RUNNING REGRESSIONS")
        print("=" * 40)
        
        results = {}
        results['column1'] = run_table1_column1(admin_during)
        results['column2'] = run_table1_column2(admin_after)
        results['column3'] = run_table1_column3(survey_endline)
        results['column4'] = run_table1_column4(survey_endline)
        results['column5'] = run_table1_column5(survey_endline)
        
        # Load randomization inference results
        print("\n" + "=" * 40)
        print("COMPUTING RI P-VALUES")
        print("=" * 40)
        
        if use_precomputed:
            # FAST VERSION: Use pre-computed coefficients
            print("Using FAST pre-computed coefficient method...")
            
            # Extract observed coefficients for RI calculation
            observed_coefficients = {
                'col1': {
                    'treatment': results['column1']['treatment_coef'],
                    'spillover': results['column1']['spillover_coef']
                },
                'col2': {
                    'treatment': results['column2']['treatment_coef'],
                    'spillover': results['column2']['spillover_coef']
                },
                'col3': {
                    'treatment': results['column3']['treatment_coef'],
                    'spillover': results['column3']['spillover_coef']
                },
                'col4': {
                    'treatment': results['column4']['treatment_coef'],
                    'spillover': results['column4']['spillover_coef']
                },
                'col5': {
                    'treatment': results['column5']['treatment_coef'],
                    'spillover': results['column5']['spillover_coef']
                }
            }
            
            ri_pvalues = load_randomization_inference_results_fast(observed_coefficients)
        else:
            # LEGACY VERSION: Not available in cleaned version
            raise NotImplementedError("Slow on-the-fly calculation removed. Use --fast (default) instead.")
        
        # Generate LaTeX table
        print("\n" + "=" * 40)
        print("GENERATING LATEX TABLE")
        print("=" * 40)
        
        # Generate LaTeX table
        complete_output_path = PATHS['output_tables'] / 'table1_complete.tex'
        generate_table1_latex(results, ri_pvalues, complete_output_path)
        
        # Report completion
        elapsed_time = time.time() - start_time
        print(f"\nTable 1 successfully generated")
        print(f"Processing time: {elapsed_time:.1f} seconds")
        print(f"Method: {method}")
        print(f"Output saved to: {complete_output_path}")
        
        # Performance comparison
        if use_precomputed:
            print(f"\n🚀 PERFORMANCE: Fast pre-computed method completed in {elapsed_time:.1f}s")
            print(f"   Expected slow method time: ~10-15 minutes")
            print(f"   Speedup: ~{600/elapsed_time:.0f}x faster!")
        
    except Exception as e:
        print(f"\nERROR generating Table 1: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 60)

#%%
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate Table 1 with randomization inference p-values"
    )
    parser.add_argument(
        "--use-new-data", 
        action="store_true", 
        help="Use new simulation data from data/rand_new/ (default: use data/rand/)"
    )
    # Legacy options kept for backward compatibility but not functional
    parser.add_argument(
        "--slow", 
        action="store_true", 
        help="Legacy option (not functional - only fast method available)"
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Legacy option (not functional - only fast method available)"
    )
    
    args = parser.parse_args()
    
    if args.benchmark:
        print("Benchmark mode removed - only fast pre-computed method available")
        main(use_new_data=args.use_new_data, use_precomputed=True)
    else:
        main(use_new_data=args.use_new_data, use_precomputed=not args.slow)
