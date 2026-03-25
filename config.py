"""
Configuration file for Blair & Weintraub (2023) Military Policing Replication
Centralized path management matching original Stata global path structure
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Path structure matching original Stata globals
PATHS = {
    # Data directories
    'data_raw': PROJECT_ROOT / 'data' / 'raw',
    'data_rand': PROJECT_ROOT / 'data' / 'rand', 
    'data_rand_new': PROJECT_ROOT / 'data' / 'rand_new',
    
    # Source code
    'src': PROJECT_ROOT / 'src',
    
    # Output directories
    'output_tables': PROJECT_ROOT / 'output' / 'tables',
    'output_figures': PROJECT_ROOT / 'output' / 'figures',
    
    # Reference (original Stata files - read only)
    'replication_package': PROJECT_ROOT / 'replication_package',
    
    # Tests
    'tests': PROJECT_ROOT / 'tests'
}

# Data file mappings (original .dta → converted .csv)
DATA_FILES = {
    # Administrative crime data
    'admin_during': 'admin_data_during.csv',
    'admin_after': 'admin_data_after.csv', 
    'admin_prior': 'admin_data_prior.csv',
    
    # Survey data
    'survey_endline': 'survey_endline.csv',
    'survey_monitoring': 'survey_monitoring.csv',
    
    # Geographic and patrol data
    'manzanas_restricted': 'manzanas_restricted.csv',
    'patrols_block_data': 'patrols_block_data.csv',
    'patrols_data': 'patrols_data.csv',
}

# Validation settings
VALIDATION_TOLERANCE = 1e-4  # Numerical precision requirement for coefficient matching

# Key variables from original Stata code (preserve exact names)
STATA_VARIABLES = {
    # Treatment variables
    'treatment': 'treatment',  # 0=control, 1=spillover, 2=treatment
    'barrio_code': 'barrio_code',
    
    # Primary outcomes
    'crime_during': 'unw_crime2_num',
    'crime_after': 'unw_crime2_num', 
    
    # Geographic controls (original Stata macros)
    'geovars': [
        # Will be populated from original .do files
    ],
    
    # Block demographic controls (original Stata macros)  
    'blockdemovars': [
        # Will be populated from original .do files
    ],
    
    # Inverse probability weights
    'iweight': 'iweight'
}

# Output file names (match original exactly)
OUTPUT_FILES = {
    # Main results
    'table1': 'table1.tex',
    'figure2': 'figure_2.pdf', 
    'figure3': 'figure_3.pdf',
    'figure4': 'figure_4.pdf',
    
    # Appendix tables (examples - full list from original)
    'table_a2': 'table_A2.tex',
    'table_a13': 'table_A13.tex',
    # ... additional appendix files as needed
}

def ensure_directories():
    """Create all necessary directories if they don't exist"""
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
    print("All directories created successfully")

def get_data_path(data_key, data_type='raw'):
    """
    Get full path to data file
    
    Args:
        data_key: Key from DATA_FILES dictionary
        data_type: 'raw' for converted CSV files, 'rand' for simulation data
    
    Returns:
        Path object to the data file
    """
    if data_type == 'raw':
        return PATHS['data_raw'] / DATA_FILES[data_key]
    elif data_type == 'rand':
        return PATHS['data_rand'] / DATA_FILES[data_key] 
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Use 'raw' or 'rand'")

def get_output_path(output_key, output_type='tables'):
    """
    Get full path to output file
    
    Args:
        output_key: Key from OUTPUT_FILES dictionary
        output_type: 'tables' for .tex files, 'figures' for .pdf files
    
    Returns:
        Path object to the output file
    """
    if output_type == 'tables':
        return PATHS['output_tables'] / OUTPUT_FILES[output_key]
    elif output_type == 'figures':
        return PATHS['output_figures'] / OUTPUT_FILES[output_key]
    else:
        raise ValueError(f"Invalid output_type: {output_type}. Use 'tables' or 'figures'")

if __name__ == "__main__":
    # Create directories when run directly
    ensure_directories()
    
    # Display configuration
    print("=== PROJECT CONFIGURATION ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Validation tolerance: {VALIDATION_TOLERANCE}")
    print(f"Data files to convert: {len(DATA_FILES)}")
    print("Configuration loaded successfully!")
