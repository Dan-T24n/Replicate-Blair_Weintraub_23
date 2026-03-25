"""
Test suite for data conversion validation
Purpose: Validate .dta to .csv conversion preserves data integrity
"""

import pytest
import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config import PATHS, DATA_FILES, VALIDATION_TOLERANCE

class TestDataConversion:
    """Test data conversion from .dta to .csv files"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.tolerance = VALIDATION_TOLERANCE
        self.stata_dir = PATHS['replication_package']
        self.csv_dir = PATHS['data_raw']
        self.rand_dir = PATHS['data_rand']
    
    def load_stata_file(self, filename):
        """Load original .dta file"""
        stata_path = self.stata_dir / filename
        if not stata_path.exists():
            pytest.skip(f"Original file {filename} not found")
        
        df, meta = pyreadstat.read_dta(str(stata_path), 
                                     apply_value_formats=False,
                                     formats_as_category=False)
        return df, meta
    
    def load_csv_file(self, filename, data_dir='raw'):
        """Load converted .csv file"""
        if data_dir == 'raw':
            csv_path = self.csv_dir / filename
        elif data_dir == 'rand':
            csv_path = self.rand_dir / filename
        else:
            raise ValueError("data_dir must be 'raw' or 'rand'")
            
        if not csv_path.exists():
            pytest.skip(f"Converted file {filename} not found")
        
        return pd.read_csv(csv_path)
    
    def validate_basic_structure(self, df_stata, df_csv, filename):
        """Validate basic data structure matches"""
        # Shape validation
        assert df_stata.shape == df_csv.shape, \
            f"{filename}: Shape mismatch - Stata: {df_stata.shape}, CSV: {df_csv.shape}"
        
        # Column names validation (exact match)
        stata_cols = set(df_stata.columns)
        csv_cols = set(df_csv.columns)
        
        assert stata_cols == csv_cols, \
            f"{filename}: Column mismatch - Missing in CSV: {stata_cols - csv_cols}, Extra in CSV: {csv_cols - stata_cols}"
        
        print(f"✓ {filename}: Basic structure validated")
    
    def validate_missing_patterns(self, df_stata, df_csv, filename):
        """Validate missing value patterns match exactly"""
        for col in df_stata.columns:
            stata_missing = df_stata[col].isnull().sum()
            csv_missing = df_csv[col].isnull().sum()
            
            assert stata_missing == csv_missing, \
                f"{filename}, {col}: Missing count mismatch - Stata: {stata_missing}, CSV: {csv_missing}"
        
        print(f"✓ {filename}: Missing patterns validated")
    
    def validate_numeric_precision(self, df_stata, df_csv, filename):
        """Validate numeric values within tolerance"""
        numeric_cols = df_stata.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Skip if all values are missing
            if df_stata[col].isnull().all():
                continue
                
            # Compare non-missing values
            stata_vals = df_stata[col].dropna()
            csv_vals = df_csv[col].dropna()
            
            if len(stata_vals) != len(csv_vals):
                pytest.fail(f"{filename}, {col}: Different number of non-missing values")
            
            # Check numerical precision
            if len(stata_vals) > 0:
                max_diff = np.abs(stata_vals.values - csv_vals.values).max()
                assert max_diff <= self.tolerance, \
                    f"{filename}, {col}: Precision error {max_diff} > {self.tolerance}"
        
        print(f"✓ {filename}: Numeric precision validated (tolerance: {self.tolerance})")
    
    def validate_categorical_preservation(self, df_stata, df_csv, filename):
        """Validate categorical/string variables preserved exactly"""
        # String and categorical columns
        string_cols = df_stata.select_dtypes(include=['object', 'category']).columns
        
        for col in string_cols:
            # Compare unique values (excluding NaN)
            stata_unique = set(df_stata[col].dropna().unique())
            csv_unique = set(df_csv[col].dropna().unique())
            
            assert stata_unique == csv_unique, \
                f"{filename}, {col}: Unique values mismatch - Stata only: {stata_unique - csv_unique}, CSV only: {csv_unique - stata_unique}"
        
        print(f"✓ {filename}: Categorical data validated")
    
    def run_full_validation(self, dta_filename, csv_filename, data_dir='raw'):
        """Run complete validation suite for a file pair"""
        print(f"\n=== Validating: {dta_filename} → {csv_filename} ===")
        
        # Load files
        df_stata, meta_stata = self.load_stata_file(dta_filename)
        df_csv = self.load_csv_file(csv_filename, data_dir)
        
        # Run all validations
        self.validate_basic_structure(df_stata, df_csv, dta_filename)
        self.validate_missing_patterns(df_stata, df_csv, dta_filename)
        self.validate_numeric_precision(df_stata, df_csv, dta_filename)
        self.validate_categorical_preservation(df_stata, df_csv, dta_filename)
        
        print(f"✅ {dta_filename}: All validations passed")
        return True

    # Individual test methods for each data file
    def test_admin_data_during(self):
        """Test administrative data during intervention"""
        self.run_full_validation('admin_data_during.dta', 'admin_data_during.csv')
    
    def test_admin_data_after(self):
        """Test administrative data after intervention"""
        self.run_full_validation('admin_data_after.dta', 'admin_data_after.csv')
    
    def test_admin_data_prior(self):
        """Test administrative data prior to intervention"""
        self.run_full_validation('admin_data_prior.dta', 'admin_data_prior.csv')
    
    def test_survey_endline(self):
        """Test endline survey data"""
        self.run_full_validation('survey_endline.dta', 'survey_endline.csv')
    
    def test_survey_monitoring(self):
        """Test monitoring survey data"""
        self.run_full_validation('survey_monitoring.dta', 'survey_monitoring.csv')
    
    def test_manzanas_restricted(self):
        """Test geographic data"""
        self.run_full_validation('manzanas_restricted.dta', 'manzanas_restricted.csv')
    
    def test_patrols_block_data(self):
        """Test patrol block data"""
        self.run_full_validation('patrols_block_data.dta', 'patrols_block_data.csv')
    
    def test_patrols_data(self):
        """Test individual patrol data"""
        self.run_full_validation('patrols_data.dta', 'patrols_data.csv')
    
    # Randomization inference data tests (10 files)
    def test_randomization_inference_p1(self):
        """Test randomization inference part 1"""
        self.run_full_validation('block_simulate_randomizations_p1.dta', 
                                'block_simulate_randomizations_p1.csv', 'rand')
    
    def test_randomization_inference_p2(self):
        """Test randomization inference part 2"""
        self.run_full_validation('block_simulate_randomizations_p2.dta', 
                                'block_simulate_randomizations_p2.csv', 'rand')
    
    def test_randomization_inference_p3(self):
        """Test randomization inference part 3"""
        self.run_full_validation('block_simulate_randomizations_p3.dta', 
                                'block_simulate_randomizations_p3.csv', 'rand')
    
    def test_randomization_inference_p4(self):
        """Test randomization inference part 4"""
        self.run_full_validation('block_simulate_randomizations_p4.dta', 
                                'block_simulate_randomizations_p4.csv', 'rand')
    
    def test_randomization_inference_p5(self):
        """Test randomization inference part 5"""
        self.run_full_validation('block_simulate_randomizations_p5.dta', 
                                'block_simulate_randomizations_p5.csv', 'rand')
    
    def test_randomization_inference_p6(self):
        """Test randomization inference part 6"""
        self.run_full_validation('block_simulate_randomizations_p6.dta', 
                                'block_simulate_randomizations_p6.csv', 'rand')
    
    def test_randomization_inference_p7(self):
        """Test randomization inference part 7"""
        self.run_full_validation('block_simulate_randomizations_p7.dta', 
                                'block_simulate_randomizations_p7.csv', 'rand')
    
    def test_randomization_inference_p8(self):
        """Test randomization inference part 8"""
        self.run_full_validation('block_simulate_randomizations_p8.dta', 
                                'block_simulate_randomizations_p8.csv', 'rand')
    
    def test_randomization_inference_p9(self):
        """Test randomization inference part 9"""
        self.run_full_validation('block_simulate_randomizations_p9.dta', 
                                'block_simulate_randomizations_p9.csv', 'rand')
    
    def test_randomization_inference_p10(self):
        """Test randomization inference part 10"""
        self.run_full_validation('block_simulate_randomizations_p10.dta', 
                                'block_simulate_randomizations_p10.csv', 'rand')

# Utility function for running tests outside pytest
def run_validation_suite():
    """Run all validation tests programmatically"""
    test_instance = TestDataConversion()
    test_instance.setup_method()
    
    # List of all test methods
    test_methods = [
        'test_admin_data_during', 'test_admin_data_after', 'test_admin_data_prior',
        'test_survey_endline', 'test_survey_monitoring', 
        'test_manzanas_restricted', 'test_patrols_block_data', 'test_patrols_data',
        'test_randomization_inference_p1', 'test_randomization_inference_p2',
        'test_randomization_inference_p3', 'test_randomization_inference_p4',
        'test_randomization_inference_p5', 'test_randomization_inference_p6',
        'test_randomization_inference_p7', 'test_randomization_inference_p8',
        'test_randomization_inference_p9', 'test_randomization_inference_p10'
    ]
    
    results = {}
    for method_name in test_methods:
        try:
            method = getattr(test_instance, method_name)
            method()
            results[method_name] = "PASSED"
        except Exception as e:
            results[method_name] = f"FAILED: {str(e)}"
    
    return results

if __name__ == "__main__":
    print("Running data conversion validation suite...")
    results = run_validation_suite()
    
    print(f"\n=== VALIDATION RESULTS ===")
    passed = sum(1 for result in results.values() if result == "PASSED")
    total = len(results)
    
    for test, result in results.items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"{status} {test}: {result}")
    
    print(f"\nSummary: {passed}/{total} tests passed")
