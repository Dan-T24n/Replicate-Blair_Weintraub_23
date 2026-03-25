"""
pipeline_random_inference_full.py

Blair & Weintraub (2023) Military Policing Replication

Complete pipeline coordinator: 1-2-2b-3-4 
Pipeline: Housekeeping → Generate New Files → Precompute RI Coefficients → Tables → Figures


Authors: Robert Blair and Michael Weintraub (original)
Pipeline coordination: 2024
"""

import time
import logging
import importlib
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_input_files(step_name, required_files):
    """Validate that required input files exist before each step"""
    logger.info(f"Validating input files for {step_name}...")
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
            logger.error(f"  MISSING: {file_path}")
        else:
            logger.info(f"  OK: {file_path}")
    
    if missing_files:
        logger.error(f"Step {step_name} validation FAILED - missing {len(missing_files)} files")
        return False
    
    logger.info(f"Step {step_name} input validation PASSED - all {len(required_files)} files found")
    return True

def validate_output_files(step_name, expected_files, timeout_seconds=30):
    """Validate that expected output files are created after each step"""
    logger.info(f"Validating output files for {step_name}...")
    
    # Wait a bit for files to be written
    time.sleep(2)
    
    missing_files = []
    for file_path in expected_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
            logger.error(f"  MISSING: {file_path}")
        else:
            file_size = file_path.stat().st_size
            logger.info(f"  OK: {file_path} ({file_size:,} bytes)")
    
    if missing_files:
        logger.error(f"Step {step_name} output validation FAILED - missing {len(missing_files)} files")
        return False
    
    logger.info(f"Step {step_name} output validation PASSED - all {len(expected_files)} files created")
    return True

def validate_final_pipeline_outputs():
    """Validate all final pipeline outputs exist and have reasonable sizes"""
    logger.info("Validating final pipeline outputs...")
    
    from config import PATHS
    
    final_outputs = [
        # Main results table
        PATHS['output_tables'] / 'table1_complete.tex',
        # Main figures
        PATHS['output_figures'] / 'figure_2.pdf',
        PATHS['output_figures'] / 'figure_3.pdf',
        PATHS['output_figures'] / 'figure_4.pdf',
        # Coefficient cache (sample files)
        PATHS['data_rand_new'] / 'coefs' / 'RI_table1_col1_p1.csv',
        PATHS['data_rand_new'] / 'coefs' / 'RI_table1_col5_p10.csv',
        # Simulation data (sample files)
        PATHS['data_rand_new'] / 'block_simulate_randomizations_p1.csv',
        PATHS['data_rand_new'] / 'block_simulate_randomizations_p10.csv'
    ]
    
    validation_results = {}
    total_size = 0
    
    for file_path in final_outputs:
        if file_path.exists():
            file_size = file_path.stat().st_size
            total_size += file_size
            validation_results[str(file_path)] = {
                'status': 'OK',
                'size': file_size,
                'size_mb': file_size / (1024 * 1024)
            }
            logger.info(f"  ✅ {file_path.name}: {file_size:,} bytes ({file_size/(1024*1024):.1f} MB)")
        else:
            validation_results[str(file_path)] = {
                'status': 'MISSING',
                'size': 0,
                'size_mb': 0
            }
            logger.error(f"  ❌ {file_path.name}: MISSING")
    
    # Summary statistics
    successful = sum(1 for r in validation_results.values() if r['status'] == 'OK')
    total = len(validation_results)
    total_mb = total_size / (1024 * 1024)
    
    logger.info(f"\nFinal Pipeline Validation Summary:")
    logger.info(f"  Files: {successful}/{total} successfully created")
    logger.info(f"  Total output size: {total_size:,} bytes ({total_mb:.1f} MB)")
    
    if successful == total:
        logger.info("🎉 All pipeline outputs validated successfully!")
        return True
    else:
        logger.error(f"⚠️  Pipeline validation incomplete - {total - successful} files missing")
        return False

def run_housekeeping():
    """Step 1: Setup paths, imports, verify environment"""
    logger.info("Starting Step 1: Housekeeping")
    start_time = time.time()
    
    try:
        housekeeping = importlib.import_module('01_housekeeping')
        
        # Run verification steps
        paths_ok = housekeeping.verify_paths()
        data_ok = housekeeping.verify_data_files()
        packages_ok = housekeeping.verify_packages()
        
        if all([paths_ok, data_ok, packages_ok]):
            logger.info(f"Step 1 completed successfully in {time.time() - start_time:.2f}s")
            return True
        else:
            logger.error("Step 1 failed - environment verification failed")
            return False
            
    except Exception as e:
        logger.error(f"Step 1 failed with error: {e}")
        return False

def run_randomization_inference():
    """Step 2: Generate new randomization inference simulations"""
    logger.info("Starting Step 2: Randomization Inference")
    start_time = time.time()
    
    # Validate input files
    from config import PATHS
    required_inputs = [
        PATHS['data_raw'] / 'manzanas_restricted.csv'
    ]
    
    if not validate_input_files("Randomization Inference", required_inputs):
        return False
    
    try:
        ri_module = importlib.import_module('02_run_randomization_inference')
        
        # Run RI generation
        ri_module.main()
        
        # Validate output files
        expected_outputs = [
            PATHS['data_rand_new'] / f'block_simulate_randomizations_p{i}.csv' 
            for i in range(1, 11)
        ]
        
        if not validate_output_files("Randomization Inference", expected_outputs):
            return False
        
        logger.info(f"Step 2 completed successfully in {time.time() - start_time:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"Step 2 failed with error: {e}")
        return False

def run_coefficient_precomputation():
    """Step 2b: Pre-compute RI coefficients for all simulations"""
    logger.info("Starting Step 2b: Coefficient Pre-computation")
    start_time = time.time()
    
    # Validate input files (simulation data from Step 2)
    from config import PATHS
    required_inputs = [
        PATHS['data_rand_new'] / f'block_simulate_randomizations_p{i}.csv' 
        for i in range(1, 11)
    ]
    
    if not validate_input_files("Coefficient Pre-computation", required_inputs):
        return False
    
    try:
        precomp_module = importlib.import_module('02b_precompute_ri_coefficients')
        
        # Run coefficient pre-computation
        precomp_module.main()
        
        # Validate output files (coefficient cache)
        expected_outputs = [
            PATHS['data_rand_new'] / 'coefs' / f'RI_table1_col{col}_p{batch}.csv'
            for col in range(1, 6)  # 5 columns
            for batch in range(1, 11)  # 10 batches
        ]
        
        if not validate_output_files("Coefficient Pre-computation", expected_outputs):
            return False
        
        logger.info(f"Step 2b completed successfully in {time.time() - start_time:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"Step 2b failed with error: {e}")
        return False

def run_table_generation():
    """Step 3: Generate tables using pre-computed coefficients"""
    logger.info("Starting Step 3: Table Generation")
    start_time = time.time()
    
    # Validate input files (coefficients from Step 2b)
    from config import PATHS
    required_inputs = [
        PATHS['data_rand_new'] / 'coefs' / f'RI_table1_col{col}_p1.csv'
        for col in range(1, 6)  # Just check first batch of each column
    ]
    
    if not validate_input_files("Table Generation", required_inputs):
        return False
    
    try:
        tables_module = importlib.import_module('03_tables')
        
        # Run table generation
        tables_module.main()
        
        # Validate output files (LaTeX tables)
        expected_outputs = [
            PATHS['output_tables'] / 'table1_complete.tex'
        ]
        
        if not validate_output_files("Table Generation", expected_outputs):
            return False
        
        logger.info(f"Step 3 completed successfully in {time.time() - start_time:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"Step 3 failed with error: {e}")
        return False

def run_figure_generation():
    """Step 4: Generate figures with RI p-values"""
    logger.info("Starting Step 4: Figure Generation")
    start_time = time.time()
    
    # Validate input files (raw data for regressions)
    from config import PATHS
    required_inputs = [
        PATHS['data_raw'] / 'admin_data_during.csv',
        PATHS['data_raw'] / 'survey_endline.csv',
        PATHS['data_raw'] / 'survey_monitoring.csv'
    ]
    
    if not validate_input_files("Figure Generation", required_inputs):
        return False
    
    try:
        figures_module = importlib.import_module('04_figures')
        
        # Run figure generation
        figures_module.main()
        
        # Validate output files (PDF figures)
        expected_outputs = [
            PATHS['output_figures'] / 'figure_2.pdf',
            PATHS['output_figures'] / 'figure_3.pdf',
            PATHS['output_figures'] / 'figure_4.pdf'
        ]
        
        if not validate_output_files("Figure Generation", expected_outputs):
            return False
        
        logger.info(f"Step 4 completed successfully in {time.time() - start_time:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"Step 4 failed with error: {e}")
        return False

def run_full_pipeline():
    """Execute complete pipeline: 1-2-2b-3-4"""
    logger.info("Starting full pipeline execution")
    pipeline_start = time.time()
    
    steps = [
        ("Housekeeping", run_housekeeping),
        ("Randomization Inference", run_randomization_inference),
        ("Coefficient Pre-computation", run_coefficient_precomputation),
        ("Table Generation", run_table_generation),
        ("Figure Generation", run_figure_generation)
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        logger.info(f"Executing: {step_name}")
        success = step_func()
        results[step_name] = success
        
        if not success:
            logger.error(f"Pipeline failed at step: {step_name}")
            return False
    
    # Final validation of all outputs
    final_validation = validate_final_pipeline_outputs()
    
    total_time = time.time() - pipeline_start
    logger.info(f"Full pipeline completed successfully in {total_time:.2f}s")
    
    # Log summary
    logger.info(f"\nPipeline Execution Summary:")
    for step_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {step_name}: {status}")
    
    logger.info(f"Final Validation: {'PASSED' if final_validation else 'FAILED'}")
    
    return final_validation

if __name__ == "__main__":
    run_full_pipeline()
