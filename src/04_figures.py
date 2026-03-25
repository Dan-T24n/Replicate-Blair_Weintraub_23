"""
04_figures.py

Blair & Weintraub (2023) Military Policing Replication
Python implementation of 04_Figures.do

Purpose: Generate Figures 2-4 using statistical significance testing without RI p-values
- Figure 2: Crime effects during intervention (coefficient plot)
- Figure 3: Safety perceptions (coefficient plot)  
- Figure 4: Abuse reporting (coefficient plot)

Authors: Robert Blair and Michael Weintraub (original)
Date created: 2022-06-30 (original Stata)
Python conversion: 2024
"""

#%%
# Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from pathlib import Path
import sys
import time

# Import project configuration
from config import PATHS, DATA_FILES

# Set matplotlib style to match original
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'

#%%
def load_admin_data_during():
    """Load administrative data during intervention for Figure 2"""
    print("Loading administrative data (during intervention) for Figure 2...")
    
    df = pd.read_csv(PATHS['data_raw'] / 'admin_data_during.csv')
    print(f"  Observations: {len(df)}")
    
    return df

#%%
def load_survey_endline():
    """Load endline survey data for Figures 3 and 4"""
    print("Loading endline survey data for Figures 3 and 4...")
    
    df = pd.read_csv(PATHS['data_raw'] / 'survey_endline.csv')
    print(f"  Observations: {len(df)}")
    
    return df

#%%
def load_survey_monitoring():
    """Load monitoring survey data for Figure 4"""
    print("Loading monitoring survey data for Figure 4...")
    
    df = pd.read_csv(PATHS['data_raw'] / 'survey_monitoring.csv')
    print(f"  Observations: {len(df)}")
    
    return df

#%%
def run_figure2_regressions(df):
    """
    Run regressions for Figure 2: Crime effects during intervention
    Mirrors Stata code lines 71-111 in 04_Figures.do
    """
    print("Running Figure 2 regressions...")
    
    # Define control variables (exact replication)
    geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
    blockdemovars = ['block_age', 'block_educ', 'block_pct_male']
    
    # Crime outcome variables for Figure 2
    outcomes = {
        'Total crime': 'unw_crime2_num',
        'Weekend crime': 'crime2_wend_num', 
        'Weekday crime': 'crime2_wday_num',
        'Daytime crime': 'crime2_day_num',
        'Nighttime crime': 'crime2_night_num'
    }
    
    results = {}
    
    for outcome_name, outcome_var in outcomes.items():
        print(f"  Running regression for {outcome_name}...")
        
        # Create formula (exact replication)
        formula = f"{outcome_var} ~ C(treatment)"
        formula += " + C(barrio_code)"
        
        # Add geographic variables
        for var in geovars:
            if var in df.columns:
                formula += f" + {var}"
        
        # Add block demographic variables  
        for var in blockdemovars:
            if var in df.columns:
                formula += f" + {var}"
        
        # Add cumulative crime control (specific to each outcome)
        cum_var = f"cum_all_{outcome_var}"
        if cum_var in df.columns:
            formula += f" + {cum_var}"
        
        # Run weighted regression
        model = smf.wls(formula, data=df, weights=df['iweight'])
        reg_results = model.fit()
        
        # Extract coefficients for treatment and spillover
        treatment_coef = reg_results.params.get('C(treatment)[T.1.0]', np.nan)
        spillover_coef = reg_results.params.get('C(treatment)[T.2.0]', np.nan)
        
        treatment_se = reg_results.bse.get('C(treatment)[T.1.0]', np.nan)
        spillover_se = reg_results.bse.get('C(treatment)[T.2.0]', np.nan)
        
        # Calculate confidence intervals
        treatment_ci = [treatment_coef - 1.96*treatment_se, treatment_coef + 1.96*treatment_se]
        spillover_ci = [spillover_coef - 1.96*spillover_se, spillover_coef + 1.96*spillover_se]
        
        results[outcome_name] = {
            'treatment_coef': treatment_coef,
            'treatment_ci': treatment_ci,
            'spillover_coef': spillover_coef,
            'spillover_ci': spillover_ci,
            'n_obs': int(reg_results.nobs),
            'r_squared': reg_results.rsquared
        }
        
        print(f"    Treatment: {treatment_coef:.3f} (SE: {treatment_se:.3f})")
        print(f"    Spillover: {spillover_coef:.3f} (SE: {spillover_se:.3f})")
    
    return results

#%%
def run_figure3_regressions(df):
    """
    Run regressions for Figure 3: Safety perceptions
    Mirrors Stata code for safety perception variables
    """
    print("Running Figure 3 regressions...")
    
    # Define control variables (exact replication)
    demovars = ['age', 'gender', 'educ']
    geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
    
    # Safety outcome variables for Figure 3 (exact replication)
    outcomes = {
        'All safety': 'i_securityallindex_std',
        'Personal safety': 'i_securityallindex_ps_std', 
        'Business safety': 'i_businessindex_std'
    }
    
    results = {}
    
    for outcome_name, outcome_var in outcomes.items():
        print(f"  Running regression for {outcome_name}...")
        
        # Create formula (exact replication)
        formula = f"{outcome_var} ~ C(treatment)"
        formula += " + C(barrio_code)"
        
        # Add demographic variables
        for var in demovars:
            if var in df.columns:
                formula += f" + {var}"
        
        # Add geographic variables
        for var in geovars:
            if var in df.columns:
                formula += f" + {var}"
        
        # Drop missing values for clustering
        vars_to_check = [outcome_var, 'treatment', 'barrio_code', 'iweight', 'manzana_code'] + demovars + geovars
        vars_to_check = [var for var in vars_to_check if var in df.columns]
        df_clean = df.dropna(subset=vars_to_check)
        
        print(f"    Observations after dropping missing: {len(df_clean)} (dropped {len(df) - len(df_clean)})")
        
        # Run weighted regression with clustered standard errors
        model = smf.wls(formula, data=df_clean, weights=df_clean['iweight'])
        reg_results = model.fit(cov_type='cluster', cov_kwds={'groups': df_clean['manzana_code']})
        
        # Extract coefficients for treatment and spillover (survey data uses integer treatment codes)
        treatment_coef = reg_results.params.get('C(treatment)[T.1]', np.nan)
        spillover_coef = reg_results.params.get('C(treatment)[T.2]', np.nan)
        
        treatment_se = reg_results.bse.get('C(treatment)[T.1]', np.nan)
        spillover_se = reg_results.bse.get('C(treatment)[T.2]', np.nan)
        
        # Calculate confidence intervals
        treatment_ci = [treatment_coef - 1.96*treatment_se, treatment_coef + 1.96*treatment_se]
        spillover_ci = [spillover_coef - 1.96*spillover_se, spillover_coef + 1.96*spillover_se]
        
        results[outcome_name] = {
            'treatment_coef': treatment_coef,
            'treatment_ci': treatment_ci,
            'spillover_coef': spillover_coef,
            'spillover_ci': spillover_ci,
            'n_obs': int(reg_results.nobs),
            'r_squared': reg_results.rsquared
        }
        
        print(f"    Treatment: {treatment_coef:.3f} (SE: {treatment_se:.3f})")
        print(f"    Spillover: {spillover_coef:.3f} (SE: {spillover_se:.3f})")
    
    return results

#%%
def run_figure4_regressions(df_monitoring, df_endline):
    """
    Run regressions for Figure 4: Abuse reporting
    Combines monitoring survey (during) and endline survey (after)
    """
    print("Running Figure 4 regressions...")
    
    # Define control variables (exact replication)
    demovars = ['age', 'gender', 'educ']
    geovars = ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min']
    
    results = {}
    
    # Monitoring survey outcomes (during intervention) - exact replication
    monitoring_outcomes = {
        'Police abuse (monitoring)': 'abuse_police',
        'Military abuse (monitoring)': 'abuse_military'
    }
    
    for outcome_name, outcome_var in monitoring_outcomes.items():
        print(f"  Running regression for {outcome_name}...")
        
        # Create formula
        formula = f"{outcome_var} ~ C(treatment)"
        formula += " + C(barrio_code)"
        
        # Add demographic variables
        for var in demovars:
            if var in df_monitoring.columns:
                formula += f" + {var}"
        
        # Add geographic variables
        for var in geovars:
            if var in df_monitoring.columns:
                formula += f" + {var}"
        
        # Drop missing values for clustering
        vars_to_check = [outcome_var, 'treatment', 'barrio_code', 'iweight', 'manzana_code'] + demovars + geovars
        vars_to_check = [var for var in vars_to_check if var in df_monitoring.columns]
        df_clean = df_monitoring.dropna(subset=vars_to_check)
        
        print(f"    Observations after dropping missing: {len(df_clean)} (dropped {len(df_monitoring) - len(df_clean)})")
        
        # Run weighted regression with clustered standard errors
        model = smf.wls(formula, data=df_clean, weights=df_clean['iweight'])
        reg_results = model.fit(cov_type='cluster', cov_kwds={'groups': df_clean['manzana_code']})
        
        # Extract coefficients (monitoring data uses float treatment codes)
        treatment_coef = reg_results.params.get('C(treatment)[T.1.0]', np.nan)
        spillover_coef = reg_results.params.get('C(treatment)[T.2.0]', np.nan)
        
        treatment_se = reg_results.bse.get('C(treatment)[T.1.0]', np.nan)
        spillover_se = reg_results.bse.get('C(treatment)[T.2.0]', np.nan)
        
        # Calculate confidence intervals
        treatment_ci = [treatment_coef - 1.96*treatment_se, treatment_coef + 1.96*treatment_se]
        spillover_ci = [spillover_coef - 1.96*spillover_se, spillover_coef + 1.96*spillover_se]
        
        results[outcome_name] = {
            'treatment_coef': treatment_coef,
            'treatment_ci': treatment_ci,
            'spillover_coef': spillover_coef,
            'spillover_ci': spillover_ci,
            'n_obs': int(reg_results.nobs),
            'r_squared': reg_results.rsquared
        }
        
        print(f"    Treatment: {treatment_coef:.3f} (SE: {treatment_se:.3f})")
        print(f"    Spillover: {spillover_coef:.3f} (SE: {spillover_se:.3f})")
    
    # Endline survey outcomes (after intervention) - exact replication  
    endline_outcomes = {
        'Police abuse (survey)': 'abuse_police_end',
        'Military abuse (survey)': 'abuse_military_end'
    }
    
    for outcome_name, outcome_var in endline_outcomes.items():
        print(f"  Running regression for {outcome_name}...")
        
        # Create formula
        formula = f"{outcome_var} ~ C(treatment)"
        formula += " + C(barrio_code)"
        
        # Add demographic variables
        for var in demovars:
            if var in df_endline.columns:
                formula += f" + {var}"
        
        # Add geographic variables
        for var in geovars:
            if var in df_endline.columns:
                formula += f" + {var}"
        
        # Drop missing values for clustering
        vars_to_check = [outcome_var, 'treatment', 'barrio_code', 'iweight', 'manzana_code'] + demovars + geovars
        vars_to_check = [var for var in vars_to_check if var in df_endline.columns]
        df_clean = df_endline.dropna(subset=vars_to_check)
        
        print(f"    Observations after dropping missing: {len(df_clean)} (dropped {len(df_endline) - len(df_clean)})")
        
        # Run weighted regression with clustered standard errors
        model = smf.wls(formula, data=df_clean, weights=df_clean['iweight'])
        reg_results = model.fit(cov_type='cluster', cov_kwds={'groups': df_clean['manzana_code']})
        
        # Extract coefficients (endline data uses integer treatment codes)
        treatment_coef = reg_results.params.get('C(treatment)[T.1]', np.nan)
        spillover_coef = reg_results.params.get('C(treatment)[T.2]', np.nan)
        
        treatment_se = reg_results.bse.get('C(treatment)[T.1]', np.nan)
        spillover_se = reg_results.bse.get('C(treatment)[T.2]', np.nan)
        
        # Calculate confidence intervals
        treatment_ci = [treatment_coef - 1.96*treatment_se, treatment_coef + 1.96*treatment_se]
        spillover_ci = [spillover_coef - 1.96*spillover_se, spillover_coef + 1.96*spillover_se]
        
        results[outcome_name] = {
            'treatment_coef': treatment_coef,
            'treatment_ci': treatment_ci,
            'spillover_coef': spillover_coef,
            'spillover_ci': spillover_ci,
            'n_obs': int(reg_results.nobs),
            'r_squared': reg_results.rsquared
        }
        
        print(f"    Treatment: {treatment_coef:.3f} (SE: {treatment_se:.3f})")
        print(f"    Spillover: {spillover_coef:.3f} (SE: {spillover_se:.3f})")
    
    return results

#%%
def create_coefficient_plot(results, title, output_path, y_labels=None):
    """
    Create coefficient plot matching original Stata coefplot style
    """
    print(f"Creating coefficient plot: {title}")
    
    # Prepare data for plotting
    outcomes = list(results.keys())
    n_outcomes = len(outcomes)
    
    # Extract coefficients and confidence intervals
    treatment_coefs = [results[outcome]['treatment_coef'] for outcome in outcomes]
    treatment_cis = [results[outcome]['treatment_ci'] for outcome in outcomes]
    spillover_coefs = [results[outcome]['spillover_coef'] for outcome in outcomes]
    spillover_cis = [results[outcome]['spillover_ci'] for outcome in outcomes]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Y positions for outcomes (reversed to match Stata output)
    y_positions = np.arange(n_outcomes)[::-1]
    
    # Plot treatment effects
    treatment_offset = 0.07
    for i, (coef, ci) in enumerate(zip(treatment_coefs, treatment_cis)):
        if not np.isnan(coef):
            y_pos = y_positions[i] + treatment_offset
            ax.scatter(coef, y_pos, marker='o', s=80, color='black', zorder=3, label='Treatment' if i == 0 else '')
            ax.plot([ci[0], ci[1]], [y_pos, y_pos], color='black', linewidth=2, zorder=2)
            ax.plot([ci[0], ci[0]], [y_pos-0.02, y_pos+0.02], color='black', linewidth=2, zorder=2)
            ax.plot([ci[1], ci[1]], [y_pos-0.02, y_pos+0.02], color='black', linewidth=2, zorder=2)
    
    # Plot spillover effects
    spillover_offset = -0.07
    for i, (coef, ci) in enumerate(zip(spillover_coefs, spillover_cis)):
        if not np.isnan(coef):
            y_pos = y_positions[i] + spillover_offset
            ax.scatter(coef, y_pos, marker='s', s=80, color='gray', zorder=3, label='Spillover' if i == 0 else '')
            ax.plot([ci[0], ci[1]], [y_pos, y_pos], color='gray', linewidth=2, linestyle='--', zorder=2)
            ax.plot([ci[0], ci[0]], [y_pos-0.02, y_pos+0.02], color='gray', linewidth=2, zorder=2)
            ax.plot([ci[1], ci[1]], [y_pos-0.02, y_pos+0.02], color='gray', linewidth=2, zorder=2)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.7, zorder=1)
    
    # Customize plot
    ax.set_yticks(y_positions)
    if y_labels:
        ax.set_yticklabels(y_labels)
    else:
        ax.set_yticklabels(outcomes)
    
    ax.set_xlabel('ITT effect size', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    
    # Remove grid
    ax.grid(False)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    
    # Set background to white
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Figure saved to: {output_path}")

#%%
def generate_figure2(admin_during):
    """Generate Figure 2: Crime effects during intervention"""
    print("\n" + "=" * 40)
    print("GENERATING FIGURE 2")
    print("=" * 40)
    
    # Run regressions
    results = run_figure2_regressions(admin_during)
    
    # Create coefficient plot
    output_path = PATHS['output_figures'] / 'figure_2.pdf'
    title = "Crime Effects During Intervention"
    
    create_coefficient_plot(results, title, output_path)
    
    return results

#%%
def generate_figure3(survey_endline):
    """Generate Figure 3: Safety perceptions"""
    print("\n" + "=" * 40)
    print("GENERATING FIGURE 3") 
    print("=" * 40)
    
    # Run regressions
    results = run_figure3_regressions(survey_endline)
    
    # Create coefficient plot
    output_path = PATHS['output_figures'] / 'figure_3.pdf'
    title = "Safety Perceptions"
    
    create_coefficient_plot(results, title, output_path)
    
    return results

#%%
def generate_figure4(survey_monitoring, survey_endline):
    """Generate Figure 4: Abuse reporting"""
    print("\n" + "=" * 40)
    print("GENERATING FIGURE 4")
    print("=" * 40)
    
    # Run regressions
    results = run_figure4_regressions(survey_monitoring, survey_endline)
    
    # Create coefficient plot with grouped labels
    output_path = PATHS['output_figures'] / 'figure_4.pdf'
    title = "Abuse Reporting"
    
    create_coefficient_plot(results, title, output_path)
    
    return results

#%%
def main():
    """
    Main function to generate Figures 2-4
    """
    print("=" * 60)
    print("FIGURES 2-4 GENERATION")
    print("Blair & Weintraub (2023) Military Policing Replication")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load data
        admin_during = load_admin_data_during()
        survey_endline = load_survey_endline()
        survey_monitoring = load_survey_monitoring()
        
        # Generate figures
        figure2_results = generate_figure2(admin_during)
        figure3_results = generate_figure3(survey_endline)
        figure4_results = generate_figure4(survey_monitoring, survey_endline)
        
        # Report completion
        elapsed_time = time.time() - start_time
        print(f"\nFigures 2-4 successfully generated")
        print(f"Processing time: {elapsed_time:.1f} seconds")
        print(f"Output saved to: {PATHS['output_figures']}")
        
    except Exception as e:
        print(f"\nERROR generating figures: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 60)

#%%
if __name__ == "__main__":
    main()
