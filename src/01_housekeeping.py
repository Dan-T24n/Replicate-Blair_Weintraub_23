"""
01_housekeeping.py : equivalent to 01_Housekeeping.do

Purpose: Verify paths, data files, and packages are properly set up
- Check that all required paths exist
- Verify all data files are present and accessible
- Test that required packages can be imported

"""

#%%
# Preliminaries (mirrors: clear all, set more off, graph set window fontface "Times New Roman")
import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output (equivalent to "set more off")
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

#%%
# Import project configuration (mirrors: global path definition and cd)
sys.path.append(str(Path(__file__).parent.parent))
from config import PATHS, DATA_FILES, ensure_directories

#%%
def verify_paths():
    """Verify all required paths exist (mirrors: global data/figures/tables setup)"""
    print("Verifying project paths...")
    
    missing_paths = []
    for path_name, path_obj in PATHS.items():
        if path_obj.exists():
            print(f"  OK {path_name}: {path_obj}")
        else:
            print(f"  MISSING {path_name}: {path_obj}")
            missing_paths.append(path_name)
    
    if missing_paths:
        print(f"\nCreating missing directories: {missing_paths}")
        ensure_directories()
        print("  All directories created")
    
    return len(missing_paths) == 0

#%%
def verify_data_files():
    """Verify all required data files exist and are accessible"""
    print("\nVerifying data files...")
    
    missing_files = []
    for file_key, filename in DATA_FILES.items():
        raw_path = PATHS['data_raw'] / filename
        rand_path = PATHS['data_rand'] / filename
        
        if raw_path.exists():
            print(f"  OK {file_key}: {raw_path}")
        elif rand_path.exists():
            print(f"  OK {file_key}: {rand_path}")
        else:
            print(f"  MISSING {file_key}: {filename}")
            missing_files.append(file_key)
    
    if missing_files:
        print(f"\nMissing data files: {missing_files}")
        print("Run data conversion scripts first!")
        return False
    
    return True

#%%
def verify_packages():
    """Verify all required packages can be imported"""
    print("\nVerifying required packages...")
    
    required_packages = {
        'pandas': 'data manipulation',
        'numpy': 'numerical computing', 
        'matplotlib': 'plotting',
        'seaborn': 'statistical plotting',
        'scipy.stats': 'statistical functions',
        'statsmodels': 'regression analysis',
        'pathlib': 'path handling'
    }
    
    failed_imports = []
    for package, description in required_packages.items():
        try:
            if '.' in package:
                main_package = package.split('.')[0]
                __import__(main_package)
                exec(f"import {package}")
            else:
                __import__(package)
            print(f"  OK {package}: {description}")
        except ImportError:
            print(f"  FAILED {package}: {description}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed package imports: {failed_imports}")
        print("Run: uv sync")
        return False
    
    return True

#%%
def verify_matplotlib_setup():
    """Verify matplotlib configuration"""
    print("\nVerifying matplotlib setup...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Set font (equivalent to Stata graph settings)
        plt.rcParams['font.family'] = 'Times New Roman'
        
        # Test basic plotting capability
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.plot([1, 2], [1, 2])
        plt.close(fig)
        
        print("  OK matplotlib: plotting and font configuration working")
        return True
        
    except Exception as e:
        print(f"  FAILED matplotlib: configuration failed ({e})")
        return False

#%%
def run_housekeeping_verification():
    """Run complete housekeeping verification"""
    print("="*60)
    print("RUNNING HOUSEKEEPING CHECKS")
    print("="*60)
    
    # Run all verification checks
    paths_ok = verify_paths()
    data_ok = verify_data_files()
    packages_ok = verify_packages()
    matplotlib_ok = verify_matplotlib_setup()
    
    # Summary
    print("\n" + "="*60)
    if all([paths_ok, data_ok, packages_ok, matplotlib_ok]):
        print("ALL HOUSEKEEPING CHECKS PASSED")
        print("Ready to run pipeline!")
        return True
    else:
        print("SOME HOUSEKEEPING CHECKS FAILED")
        print("Fix issues above before running analysis!")
        return False

#%%
# Main execution
if __name__ == "__main__":
    success = run_housekeeping_verification()
    sys.exit(0 if success else 1)
