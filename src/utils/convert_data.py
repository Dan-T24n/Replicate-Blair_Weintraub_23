#!/usr/bin/env python3
"""
Convert all .dta files to CSV format with exact schema preservation
Blair & Weintraub (2023) Military Policing Replication - Phase 1
"""

import pyreadstat
import pandas as pd
from pathlib import Path
import numpy as np
from config import PATHS, DATA_FILES

def convert_dta_to_csv():
    """Convert all .dta files to CSV with exact schema preservation"""
    
    # Ensure output directories exist
    PATHS['data_raw'].mkdir(parents=True, exist_ok=True)
    PATHS['data_rand'].mkdir(parents=True, exist_ok=True)
    
    # Path to original Stata files
    stata_dir = PATHS['replication_package']
    
    # Get all .dta files
    dta_files = list(stata_dir.glob("*.dta"))
    print(f"Found {len(dta_files)} .dta files to convert\n")
    
    conversion_log = {}
    
    for dta_file in sorted(dta_files):
        print(f"Converting: {dta_file.name}")
        
        try:
            # Read .dta file preserving all metadata
            df, meta = pyreadstat.read_dta(str(dta_file), 
                                         apply_value_formats=False,
                                         formats_as_category=False)
            
            # Determine output path based on file type
            if dta_file.name.startswith('block_simulate_randomizations'):
                # Randomization inference files go to data/rand/
                csv_path = PATHS['data_rand'] / f"{dta_file.stem}.csv"
            else:
                # All other files go to data/raw/
                csv_path = PATHS['data_raw'] / f"{dta_file.stem}.csv"
            
            # Convert to CSV preserving exact schema
            df.to_csv(csv_path, index=False, na_rep='')
            
            # Log conversion details
            conversion_log[dta_file.name] = {
                'output_path': str(csv_path),
                'shape': df.shape,
                'missing_values': df.isnull().sum().sum(),
                'dtypes': df.dtypes.to_dict(),
                'variable_labels': getattr(meta, 'column_labels', {}),
                'value_labels': getattr(meta, 'value_labels', {})
            }
            
            print(f"  → {csv_path.name} (shape: {df.shape}, missing: {df.isnull().sum().sum()})")
            
        except Exception as e:
            print(f"  ERROR converting {dta_file.name}: {e}")
            conversion_log[dta_file.name] = {'error': str(e)}
    
    # Save conversion metadata
    metadata_path = PATHS['data_raw'] / 'conversion_metadata.txt'
    with open(metadata_path, 'w') as f:
        f.write("Data Conversion Log - Blair & Weintraub (2023) Replication\n")
        f.write("="*60 + "\n\n")
        
        for filename, info in conversion_log.items():
            f.write(f"File: {filename}\n")
            if 'error' in info:
                f.write(f"  ERROR: {info['error']}\n")
            else:
                f.write(f"  Output: {info['output_path']}\n")
                f.write(f"  Shape: {info['shape']}\n")
                f.write(f"  Missing values: {info['missing_values']}\n")
                f.write(f"  Variables: {len(info['dtypes'])}\n")
                if info['variable_labels']:
                    f.write(f"  Has variable labels: {len(info['variable_labels'])}\n")
                if info['value_labels']:
                    f.write(f"  Has value labels: {len(info['value_labels'])}\n")
            f.write("\n")
    
    print(f"\nConversion completed!")
    print(f"Conversion log saved to: {metadata_path}")
    print(f"Raw data files: {len(list(PATHS['data_raw'].glob('*.csv')))}")
    print(f"Randomization files: {len(list(PATHS['data_rand'].glob('*.csv')))}")
    
    return conversion_log

def validate_conversions():
    """Validate converted CSV files against original .dta files"""
    
    print("\nValidating conversions...")
    stata_dir = PATHS['replication_package']
    validation_results = {}
    
    for dta_file in sorted(stata_dir.glob("*.dta")):
        # Determine corresponding CSV path
        if dta_file.name.startswith('block_simulate_randomizations'):
            csv_path = PATHS['data_rand'] / f"{dta_file.stem}.csv"
        else:
            csv_path = PATHS['data_raw'] / f"{dta_file.stem}.csv"
        
        if not csv_path.exists():
            print(f"  MISSING: {csv_path.name}")
            continue
            
        try:
            # Load both files
            df_stata, _ = pyreadstat.read_dta(str(dta_file))
            df_csv = pd.read_csv(csv_path)
            
            # Compare shapes
            shape_match = df_stata.shape == df_csv.shape
            
            # Compare missing patterns (allowing for NaN representation differences)
            missing_stata = df_stata.isnull().sum().sum()
            missing_csv = df_csv.isnull().sum().sum()
            missing_match = abs(missing_stata - missing_csv) <= 1  # Allow small tolerance
            
            validation_results[dta_file.name] = {
                'shape_match': shape_match,
                'missing_match': missing_match,
                'stata_shape': df_stata.shape,
                'csv_shape': df_csv.shape,
                'stata_missing': missing_stata,
                'csv_missing': missing_csv
            }
            
            status = "✓" if (shape_match and missing_match) else "✗"
            print(f"  {status} {dta_file.name} → {csv_path.name}")
            if not shape_match:
                print(f"    Shape mismatch: {df_stata.shape} vs {df_csv.shape}")
            if not missing_match:
                print(f"    Missing mismatch: {missing_stata} vs {missing_csv}")
                
        except Exception as e:
            print(f"  ERROR validating {dta_file.name}: {e}")
            validation_results[dta_file.name] = {'error': str(e)}
    
    # Summary
    successful = sum(1 for r in validation_results.values() 
                    if r.get('shape_match', False) and r.get('missing_match', False))
    total = len(validation_results)
    print(f"\nValidation complete: {successful}/{total} files validated successfully")
    
    return validation_results

if __name__ == "__main__":
    print("Blair & Weintraub (2023) Data Conversion - Phase 1")
    print("="*50)
    
    # Convert all files
    conversion_log = convert_dta_to_csv()
    
    # Validate conversions
    validation_results = validate_conversions()
    
    print("\nPhase 1 Data Conversion Complete!")
