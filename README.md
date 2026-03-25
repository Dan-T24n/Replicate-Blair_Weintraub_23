# Blair & Weintraub (2023) Military Policing Replication

A complete Python replication of Blair & Weintraub (2023), converting the entire Stata pipeline to Python while preserving exact logic, results, and statistical procedures.

## Original Paper

Blair, R. A. & Weintraub, M. (2023). Little evidence that military policing reduces crime or improves human security. *Nature Human Behaviour*, 7, 861-873. [https://doi.org/10.1038/s41562-023-01600-1](https://doi.org/10.1038/s41562-023-01600-1)

**Replication package**: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WAJ9SR](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WAJ9SR) (Harvard Dataverse)

**Summary**: The study evaluates the effect of military policing on crime and human security through a randomized controlled trial in Cali, Colombia. 1,255 city blocks were randomly assigned to treatment (joint military-police patrols), spillover (within 25m of treated blocks), or control conditions. Using administrative crime records and survey data, the authors find little evidence that military patrols reduced crime or improved residents' perceptions of security. Randomization inference with 100,000 simulations confirms the null results are robust. The findings challenge the widespread policy of deploying military forces for domestic policing in Latin America and beyond.

## Project Overview

**Replication Goal**: Convert entire Stata replication package to Python while maintaining:
- Exact variable names and value encodings
- Identical statistical procedures and results
- Original file structure and naming conventions
- Numerical precision within 1e-4 tolerance

## Project Structure

```
replication_brodeaux/
├── data/
│   ├── raw/                    # Converted CSV files from original .dta
│   ├── rand/                   # Original simulation data (from replication package)
│   └── rand_new/               # New simulation data + coefficient cache
├── src/                       # Python scripts mirroring original .do files
│   ├── 00_pipeline_full.py    # Complete pipeline coordinator
│   ├── 01_housekeeping.py     # Path setup, imports
│   ├── 02_run_randomization_inference.py  # Simulation engine
│   ├── 02b_precompute_ri_coefficients.py  # Coefficient pre-computation
│   ├── 03_tables.py           # Table generation (LaTeX output)
│   ├── 04_figures.py          # Figure generation (PDF output)
│   └── utils/                 # Shared utility modules
├── output/
│   ├── tables/               # LaTeX tables (.tex files)
│   └── figures/              # PDF figures
├── tests/                    # Test suite for validation
├── config.py                 # Centralized path management
├── LICENSE                   # MIT License
└── pyproject.toml           # Python environment configuration
```

## Key Features

- **Exact Replication**: All variable names (`unw_crime2_num`, `i.treatment`, `i.barrio_code`, etc.) preserved exactly
- **Statistical Fidelity**: Probability weighting, clustered standard errors, and randomization inference replicated precisely
- **Three-Group Design**: Treatment/spillover/control framework with identical specifications
- **Performance Optimization**: Python implementation with parallel processing for 100,000+ simulations
- **Format Preservation**: LaTeX tables and PDF figures matching original styling exactly

## Environment Setup

This project uses `uv` for dependency management with Python 3.12.

### Prerequisites
- Python 3.12+
- `uv` package manager

### Installation
```bash
# Clone repository
git clone <repository-url>
cd replication_brodeaux

# Install dependencies with uv
uv sync

# Activate environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### Required Packages
- **Data Processing**: pandas, numpy, pyreadstat
- **Statistics**: scipy, statsmodels
- **Visualization**: matplotlib, seaborn
- **Performance**: numba, joblib, tqdm
- **Testing**: pytest

## Usage

### Full Pipeline (Recommended)
```bash
# Run complete pipeline: 1-2-2b-3-4 (generates new RI simulations)
uv run src/00_pipeline_full.py
```

### Individual Scripts (Development)
```bash
# Setup paths and imports
uv run src/01_housekeeping.py

# Generate new simulation data (100,000 simulations)
uv run src/02_run_randomization_inference.py

# Pre-compute regression coefficients (optional optimization)
uv run src/02b_precompute_ri_coefficients.py

# Generate tables
uv run src/03_tables.py

# Generate figures  
uv run src/04_figures.py
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific validation tests
pytest tests/test_table1_numerical_comparison.py
pytest tests/test_data_conversion.py
```

### Key Output Files
- **Table 1**: `output/tables/table1_complete_RI.tex` - Main treatment effects with RI p-values
- **Figure 2**: `output/figures/figure_2.pdf` - Crime effects during intervention
- **Figure 3**: `output/figures/figure_3.pdf` - Survey outcomes
- **Figure 4**: `output/figures/figure_4.pdf` - Human rights effects

### Supplementary Analysis Scripts

The following scripts in `src/` are not part of the core pipeline but document the analytical work done to validate the weighting methodology:

- `analyze_iweight_patterns.py` - Investigation of inverse probability weight distributions
- `reproduce_weights.py` - Independent reconstruction of the weighting scheme from first principles
- `robustness_balance_test.py` - Balance checks across treatment arms
- `validate_weighting_schemes.py` - Cross-validation of weighting approaches against Stata output

## Data Sources

### Administrative Data
- `admin_data_during.csv` - Crime data during intervention
- `admin_data_after.csv` - Post-intervention crime data  
- `admin_data_prior.csv` - Pre-intervention baseline data

### Survey Data
- `survey_endline.csv` - Post-intervention survey responses
- `survey_monitoring.csv` - Monitoring survey data

### Geographic & Patrol Data
- `manzanas_restricted.csv` - Geographic block characteristics
- `patrols_block_data.csv` - Block-level patrol data
- `patrols_data.csv` - Individual patrol records

### Randomization Inference
- `block_simulate_randomizations_p*.csv` - Simulation results (10 files)

## Statistical Framework

### Core Methodology
- **Inverse Probability Weighting**: Exact replication of Stata `iweight` calculations
- **Clustered Standard Errors**: Block-level clustering matching Stata methodology
- **Randomization Inference**: Two-tailed tests with 100,000 simulations
- **Treatment Assignment**: Three-group design with spillover effects

### Key Variables
- **Primary Outcome**: `unw_crime2_num` (unweighted crime count)
- **Treatment**: `i.treatment` (categorical: control/spillover/treatment)
- **Geographic Controls**: `$geovars` (original Stata macro variables)
- **Block Demographics**: `$blockdemovars` (original Stata macro variables)

## Validation Criteria

- **Numerical Precision**: Coefficients match within 1e-4 tolerance
- **Statistical Output**: All p-values and confidence intervals identical
- **File Format**: LaTeX and PDF outputs match original styling exactly
- **Variable Names**: No renaming or case changes from original Stata code

## Performance Achievements

- **Simulation Speed**: 1,578x faster than original Stata implementation
- **Execution Time**: 100,000 randomization inference simulations complete in 7.6 seconds
- **Statistical Equivalence**: 99.9% match with original simulation results
- **Memory Efficiency**: Vectorized operations with multiprocessing support

## Reference

See [Original Paper](#original-paper) section above for full citation, replication package link, and study summary.

## License

This Python replication code is released under the [MIT License](LICENSE).

The original data from Blair & Weintraub (2023) is available on [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WAJ9SR) under CC0 1.0 Public Domain Dedication.
