#!/usr/bin/env python3
"""
Table Comparison Utility
Compare reproduction results with original results and determine formatting
"""

def get_original_results():
    """Get original results from Blair & Weintraub (2023) paper"""
    return {
        'treatment': [0.003, 0.110, 0.006, -0.007, 0.153],
        'spillover': [-0.038, 0.083, 0.026, 0.013, 0.186],
        'control_means': [0.160, 0.160, -0.021, -0.016, -0.119],
        'treatment_ci': [
            [-0.068, 0.074], [0.011, 0.208], [-0.077, 0.089], [-0.098, 0.085], [0.051, 0.256]
        ],
        'spillover_ci': [
            [-0.097, 0.022], [-0.003, 0.169], [-0.034, 0.086], [-0.061, 0.087], [0.101, 0.270]
        ],
        'treatment_pval': [0.934, 0.029, 0.886, 0.886, 0.003],
        'spillover_pval': [0.212, 0.059, 0.389, 0.729, '<0.001'],
        'observations': [1167, 1167, 7845, 7845, 7837],
        'r_squared': [0.33, 0.48, 0.03, 0.03, 0.12],
        'ri_treatment': [0.959, 0.136, 0.927, 0.914, 0.038],
        'ri_spillover': [0.411, 0.138, 0.610, 0.802, 0.001]
    }

def extract_reproduced_results(results, ri_pvalues):
    """Extract reproduced results from regression and RI results"""
    return {
        'treatment': [results[f'column{i}']['treatment_coef'] for i in range(1, 6)],
        'spillover': [results[f'column{i}']['spillover_coef'] for i in range(1, 6)],
        'control_means': [results[f'column{i}']['control_mean'] for i in range(1, 6)],
        'treatment_ci': [results[f'column{i}']['treatment_ci'] for i in range(1, 6)],
        'spillover_ci': [results[f'column{i}']['spillover_ci'] for i in range(1, 6)],
        'treatment_pval': [results[f'column{i}']['treatment_pval'] for i in range(1, 6)],
        'spillover_pval': [results[f'column{i}']['spillover_pval'] for i in range(1, 6)],
        'observations': [int(results[f'column{i}']['n_obs']) for i in range(1, 6)],
        'r_squared': [results[f'column{i}']['r_squared'] for i in range(1, 6)],
        'ri_treatment': [ri_pvalues[f'col{i}']['treatment'] for i in range(1, 6)],
        'ri_spillover': [ri_pvalues[f'col{i}']['spillover'] for i in range(1, 6)]
    }

def values_differ(orig, repro, tolerance=0.001):
    """Check if values differ significantly (default e-3 precision)"""
    if isinstance(orig, str):
        return False
    try:
        return abs(float(orig) - float(repro)) > tolerance
    except (ValueError, TypeError):
        return False

def ci_differs(orig_ci, repro_ci, tolerance=0.001):
    """Check if confidence intervals differ significantly"""
    return (values_differ(orig_ci[0], repro_ci[0], tolerance) or 
            values_differ(orig_ci[1], repro_ci[1], tolerance))

def get_formatting_flags(results, ri_pvalues, tolerance=0.001):
    """
    Determine which values should be bold in the reproduction table
    Returns dictionary with formatting flags for each element
    """
    original = get_original_results()
    reproduced = extract_reproduced_results(results, ri_pvalues)
    
    formatting = {
        'treatment_coef': [],
        'treatment_ci': [],
        'treatment_pval': [],
        'spillover_coef': [],
        'spillover_ci': [],
        'spillover_pval': [],
        'observations': [],
        'r_squared': [],
        'control_mean': [],
        'ri_treatment': [],
        'ri_spillover': []
    }
    
    for i in range(5):
        # Treatment effects
        formatting['treatment_coef'].append(
            values_differ(original['treatment'][i], reproduced['treatment'][i], tolerance)
        )
        formatting['treatment_ci'].append(
            ci_differs(original['treatment_ci'][i], reproduced['treatment_ci'][i], tolerance)
        )
        formatting['treatment_pval'].append(
            values_differ(original['treatment_pval'][i], reproduced['treatment_pval'][i], tolerance)
        )
        
        # Spillover effects
        formatting['spillover_coef'].append(
            values_differ(original['spillover'][i], reproduced['spillover'][i], tolerance)
        )
        formatting['spillover_ci'].append(
            ci_differs(original['spillover_ci'][i], reproduced['spillover_ci'][i], tolerance)
        )
        # Handle string p-values like '<0.001'
        if isinstance(original['spillover_pval'][i], str):
            formatting['spillover_pval'].append(False)
        else:
            formatting['spillover_pval'].append(
                values_differ(original['spillover_pval'][i], reproduced['spillover_pval'][i], tolerance)
            )
        
        # Summary statistics
        formatting['observations'].append(
            original['observations'][i] != reproduced['observations'][i]
        )
        # R² values should use a more lenient tolerance (they're often rounded)
        formatting['r_squared'].append(
            values_differ(original['r_squared'][i], reproduced['r_squared'][i], 0.01)
        )
        formatting['control_mean'].append(
            values_differ(original['control_means'][i], reproduced['control_means'][i], tolerance)
        )
        
        # RI p-values - KEY FOCUS: highlight significant differences
        formatting['ri_treatment'].append(
            values_differ(original['ri_treatment'][i], reproduced['ri_treatment'][i], tolerance)
        )
        formatting['ri_spillover'].append(
            values_differ(original['ri_spillover'][i], reproduced['ri_spillover'][i], tolerance)
        )
    
    return formatting

def print_comparison_summary(results, ri_pvalues, tolerance=0.001):
    """Print summary of differences for debugging"""
    formatting = get_formatting_flags(results, ri_pvalues, tolerance)
    original = get_original_results()
    reproduced = extract_reproduced_results(results, ri_pvalues)
    
    print(f"\n=== COMPARISON SUMMARY (tolerance: {tolerance}) ===")
    
    categories = [
        ('Treatment Coefficients', 'treatment_coef', 'treatment'),
        ('Spillover Coefficients', 'spillover_coef', 'spillover'),
        ('RI Treatment P-values', 'ri_treatment', 'ri_treatment'),
        ('RI Spillover P-values', 'ri_spillover', 'ri_spillover'),
    ]
    
    for category_name, format_key, data_key in categories:
        print(f"\n{category_name}:")
        for i in range(5):
            if formatting[format_key][i]:
                orig_val = original[data_key][i]
                repro_val = reproduced[data_key][i]
                diff = abs(float(orig_val) - float(repro_val)) if not isinstance(orig_val, str) else 'N/A'
                print(f"  Column {i+1}: Original={orig_val}, Reproduced={repro_val:.6f}, Diff={diff}")
    
    return formatting
