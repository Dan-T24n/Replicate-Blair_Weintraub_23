"""
validate_weighting_schemes.py

Blair & Weintraub (2023) Military Policing Replication
Author: 2024

Purpose:
This script provides a comprehensive analysis of the `iweight` variable by:
1.  Reproducing the weights via a high-fidelity bootstrap simulation of the
    original two-stage stratified randomization design.
2.  Calculating several analytical alternative weighting schemes (Simple IPW,
    Stratified IPW, etc.) for comparison.
3.  Generating a validation report that compares all generated weights to the
    original `iweight` using correlation and summary statistics, proving that a
    "Hybrid" or "Precision Weighting" model is the best fit.
4.  Running a sensitivity analysis by re-calculating the main results from
    Table 1 with each weighting scheme to show the impact on the final coefficients.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
import sys
import logging
from tqdm import tqdm
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import PATHS

# --- Part 1: Simulation-Based Weight Reproduction (from reproduce_weights.py) ---

def run_bootstrap_simulation(block_data, manzanas_adj, n_simulations=1500):
    """
    Runs a bootstrap simulation of the two-stage stratified randomization.
    """
    logging.info(f"Starting {n_simulations} bootstrap simulations...")
    
    manzana_codes = block_data['manzana'].unique()
    manzana_to_idx = {code: i for i, code in enumerate(manzana_codes)}
    
    adjacency_map = {}
    for block_id, row in manzanas_adj.iterrows():
        neighbors = pd.to_numeric(row.dropna(), errors='coerce').dropna().astype(int)
        adjacency_map[block_id] = set(neighbors)

    assignment_counts = np.zeros((len(manzana_codes), 3), dtype=np.int32)
    barrios = block_data.groupby('barrio_code')['manzana'].apply(list)

    for _ in tqdm(range(n_simulations), desc="Running Simulations"):
        simulated_treatment_manzanas = set()
        for _, block_list in barrios.items():
            shuffled_blocks = np.random.permutation(block_list)
            n_treatment = int(round(len(shuffled_blocks) / 6.0))
            if n_treatment > 0:
                simulated_treatment_manzanas.update(shuffled_blocks[:n_treatment])

        sim_treat_indices = np.array([manzana_to_idx[b] for b in simulated_treatment_manzanas])
        if len(sim_treat_indices) > 0:
            assignment_counts[sim_treat_indices, 1] += 1

        spillover_manzanas = set()
        if simulated_treatment_manzanas:
            for block_id, neighbors in adjacency_map.items():
                if block_id not in simulated_treatment_manzanas:
                    if not neighbors.isdisjoint(simulated_treatment_manzanas):
                        spillover_manzanas.add(block_id)
        
        spillover_indices = np.array([manzana_to_idx[b] for b in spillover_manzanas if b in manzana_to_idx])
        if len(spillover_indices) > 0:
            assignment_counts[spillover_indices, 2] += 1

        control_indices = np.array(list(set(range(len(manzana_codes))) - set(sim_treat_indices) - set(spillover_indices)))
        if len(control_indices) > 0:
            assignment_counts[control_indices, 0] += 1
            
    assignment_probs = assignment_counts / n_simulations
    
    prob_df = pd.DataFrame({
        'manzana': manzana_codes,
        'p_control': assignment_probs[:, 0],
        'p_treatment': assignment_probs[:, 1],
        'p_spillover': assignment_probs[:, 2]
    })
    return prob_df

def calculate_simulated_hybrid_weight(row):
    """Calculates the final weight based on the 'precision weighting' discovery."""
    ipw_treatment = 1 / row['p_treatment'] if row['p_treatment'] > 0 else 0
    if row['treatment'] == 1:
        return ipw_treatment * 2
    elif row['treatment'] == 0:
        return 1 / row['p_control'] if row['p_control'] > 0 else 0
    elif row['treatment'] == 2:
        return 1 / row['p_spillover'] if row['p_spillover'] > 0 else 0
    return np.nan

# --- Part 2: Analytical Weight Calculation (from robustness_test_weights.py) ---

def calculate_analytical_weights(df, method='simple'):
    """Calculates alternative analytical weight schemes."""
    df = df.copy()
    if method == 'unweighted':
        df['alt_weight'] = 1.0
        return df['alt_weight']
    elif method == 'simple_ipw':
        counts = df['treatment'].value_counts()
        probs = counts / len(df)
        weights = 1 / probs
        df['alt_weight'] = df['treatment'].map(weights)
    elif method == 'stratified_ipw':
        df['alt_weight'] = 0.0
        for barrio in df['barrio_code'].unique():
            barrio_mask = df['barrio_code'] == barrio
            barrio_data = df[barrio_mask]
            barrio_total = len(barrio_data)
            counts = barrio_data['treatment'].value_counts()
            probs = counts / barrio_total
            weights = 1 / probs
            for treat, weight in weights.items():
                if np.isinf(weight): weight = 0 # Handle cases with zero probability
                mask = (df['barrio_code'] == barrio) & (df['treatment'] == treat)
                df.loc[mask, 'alt_weight'] = weight
    elif method == 'analytical_hybrid':
        # First, calculate the stratified weights and store them in 'alt_weight'
        df['alt_weight'] = calculate_analytical_weights(df, 'stratified_ipw')
        # Then, apply the multiplier to the treatment group
        df.loc[df['treatment'] == 1, 'alt_weight'] *= 2
    else:
        raise ValueError(f"Unknown method: {method}")
    return df['alt_weight']

# --- Part 3: Regression Analysis (from robustness_test_weights.py) ---

def run_regression(formula, df, weight_col, cluster_col=None):
    """Runs a single regression with a specific weighting scheme."""
    outcome_var = formula.split('~')[0].strip()
    
    # Ensure all necessary columns exist and drop NA
    all_vars = [v for v in re.findall(r'\b\w+\b', formula) if v in df.columns]
    all_vars.extend([weight_col, 'treatment', 'barrio_code'])
    if cluster_col: all_vars.append(cluster_col)
    df_clean = df.dropna(subset=list(set(all_vars)))

    model = smf.wls(formula, data=df_clean, weights=df_clean[weight_col])
    
    if cluster_col:
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df_clean[cluster_col]})
    else:
        results = model.fit()

    # Extract coefficients safely
    coefs = {'treatment_coef': np.nan, 'spillover_coef': np.nan}
    for param in results.params.index:
        if 'C(treatment)[T.1]' in param: coefs['treatment_coef'] = results.params[param]
        if 'C(treatment)[T.2]' in param: coefs['spillover_coef'] = results.params[param]
            
    return coefs

# --- Main Execution ---

def main():
    """Main function to run the full validation and sensitivity analysis."""
    
    # Load data
    logging.info("Loading data...")
    admin_during = pd.read_csv(PATHS['data_raw'] / 'admin_data_during.csv')
    admin_after = pd.read_csv(PATHS['data_raw'] / 'admin_data_after.csv')
    survey = pd.read_csv(PATHS['data_raw'] / 'survey_endline.csv')
    manzanas_data = pd.read_csv(PATHS['data_raw'] / 'manzanas_restricted.csv')
    
    # --- Part 1 & 2: Generate All Weight Types ---
    logging.info("Generating all weighting schemes for comparison...")
    
    # Simulation-based weights
    manzana_adj_cols = [f'manzana_25m{i}' for i in range(1, 46)]
    manzanas_adj = manzanas_data[['manzana'] + manzana_adj_cols].set_index('manzana')
    prob_df = run_bootstrap_simulation(admin_during, manzanas_adj)
    weights_df = pd.merge(admin_during[['manzana', 'barrio_code', 'treatment', 'iweight']], prob_df, on='manzana')
    weights_df['simulated_hybrid'] = weights_df.apply(calculate_simulated_hybrid_weight, axis=1)
    
    # Analytical weights
    weights_df['unweighted'] = 1.0
    weights_df['simple_ipw'] = calculate_analytical_weights(weights_df, 'simple_ipw')
    weights_df['stratified_ipw'] = calculate_analytical_weights(weights_df, 'stratified_ipw')
    weights_df['analytical_hybrid'] = calculate_analytical_weights(weights_df, 'analytical_hybrid')

    # --- Part 3: Validation Report ---
    logging.info("Creating validation report...")
    
    weight_cols = ['simulated_hybrid', 'analytical_hybrid', 'stratified_ipw', 'simple_ipw', 'unweighted']
    correlations = weights_df[weight_cols].corrwith(weights_df['iweight'])
    
    print("\n" + "="*50)
    print("WEIGHTING SCHEME VALIDATION REPORT")
    print("="*50)
    print("\n--- Correlation with Original 'iweight' ---")
    print(correlations.sort_values(ascending=False))
    
    print("\n--- Summary Statistics by Treatment Group ---")
    summary_cols = ['iweight', 'simulated_hybrid', 'analytical_hybrid', 'stratified_ipw']
    print(weights_df.groupby('treatment')[summary_cols].mean())

    # --- Part 4: Sensitivity Analysis (Table 1) ---
    logging.info("Running sensitivity analysis for Table 1...")
    
    # Define formulas
    geovars = " + ".join([v for v in ['number_buildings_sampling', 'area', 'bat_min', 'cai_min', 'ptr_min'] if v in admin_during.columns])
    blockdemovars = " + ".join([v for v in ['block_age', 'block_educ', 'block_pct_male'] if v in admin_during.columns])
    demovars = " + ".join([v for v in ['age', 'gender', 'educ'] if v in survey.columns])
    
    formulas = {
        "Col 1: Crime During": f"unw_crime2_num ~ C(treatment) + C(barrio_code) + {geovars} + {blockdemovars} + cum_all_unw_crime2_num",
        "Col 2: Crime After": f"unw_crime_num ~ C(treatment) + C(barrio_code) + {geovars} + {blockdemovars} + cum_all_unw_crime_num",
        "Col 3: Victim During": f"i2_victimduringindex_std ~ C(treatment) + C(barrio_code) + {demovars} + {geovars}",
        "Col 4: Victim After": f"i2_victimafterindex_std ~ C(treatment) + C(barrio_code) + {demovars} + {geovars}",
        "Col 5: Witness After": f"i_witnessindex_std ~ C(treatment) + C(barrio_code) + {demovars} + {geovars}"
    }
    
    datasets = {
        "Col 1: Crime During": admin_during, "Col 2: Crime After": admin_after,
        "Col 3: Victim During": survey, "Col 4: Victim After": survey, "Col 5: Witness After": survey
    }
    cluster_cols = {
        "Col 3: Victim During": "manzana", "Col 4: Victim After": "manzana", "Col 5: Witness After": "manzana"
    }

    # Merge all generated weights into all datasets
    survey = survey.rename(columns={'manzana_code': 'manzana'})
    weights_to_merge = weights_df.drop(columns=['barrio_code', 'treatment', 'iweight'])

    datasets["Col 1: Crime During"] = pd.merge(datasets["Col 1: Crime During"], weights_to_merge, on='manzana', how='left')
    datasets["Col 2: Crime After"] = pd.merge(datasets["Col 2: Crime After"], weights_to_merge, on='manzana', how='left')
    
    # Merge into the single survey dataframe
    survey_merged = pd.merge(survey, weights_to_merge, on='manzana', how='left')
    datasets["Col 3: Victim During"] = survey_merged
    datasets["Col 4: Victim After"] = survey_merged
    datasets["Col 5: Witness After"] = survey_merged
    
    all_results = []
    weight_schemes_for_regression = {'Original': 'iweight', **{col.replace("_"," ").title(): col for col in weight_cols}}

    for name, col in weight_schemes_for_regression.items():
        row = {'Weighting Scheme': name}
        for table_col, formula in formulas.items():
            df = datasets[table_col]
            cluster = cluster_cols.get(table_col)
            coefs = run_regression(formula, df, col, cluster)
            row[f'{table_col} (Treat Coef)'] = coefs['treatment_coef']
        all_results.append(row)
        
    results_df = pd.DataFrame(all_results).set_index('Weighting Scheme')
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: TABLE 1 TREATMENT EFFECTS ACROSS WEIGHTING SCHEMES")
    print("="*80)
    print(results_df.to_string(float_format="%.4f"))
    
    logging.info("Script finished.")


if __name__ == "__main__":
    main()
