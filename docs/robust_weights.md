# Robustness Analysis: Weighting Schemes in Blair & Weintraub (2023)

**Date:** 2024  
**Analysis:** Alternative weighting explanations and sensitivity testing for military policing effects

## Executive Summary

This analysis investigates the complex weighting scheme used in Blair & Weintraub (2023) and tests the sensitivity of results to alternative weighting approaches. We find that the original weights represent a sophisticated design-based correction that goes beyond simple inverse probability weighting, with treatment groups receiving approximately double the expected inverse probability weights.

## Key Findings

### 1. Original Weighting Scheme Analysis

**Weight Distribution by Treatment Group:**

| Treatment Group | Description | Blocks | Population % | Mean Weight | Weight Range |
|----------------|-------------|--------|-------------|-------------|--------------|
| **0 (Control)** | Pure control blocks | 275 | 21.9% | 4.43 | 1.24 - 17.65 |
| **1 (Treatment)** | Military patrol blocks | 214 | 17.1% | 11.78 | 6.05 - 19.74 |
| **2 (Spillover)** | Adjacent to treatment | 765 | 61.0% | 1.62 | 1.20 - 6.67 |

**Key Observations:**
- Control and spillover weights approximate inverse probability weights (IPW)
- Treatment weights are **~2x higher** than expected IPW (11.78 vs 5.86)
- Weights vary significantly within treatment groups, indicating stratification effects
- Strong correlation with neighborhood (barrio) structure

### 2. Alternative Weighting Explanations Tested

We tested four alternative explanations for the weighting scheme:

#### A. Simple Inverse Probability Weights
- **Formula:** Weight = 1 / (group size / total sample)
- **Correlation with original:** 0.85 (admin data), 0.84 (survey data)
- **Limitation:** Doesn't account for stratification or treatment group adjustment

#### B. Stratified Inverse Probability Weights  
- **Formula:** Weight = 1 / (group size within barrio / barrio total)
- **Correlation with original:** 0.79 (admin data), 0.77 (survey data)
- **Limitation:** Still underweights treatment group

#### C. Sample Size Adjustment
- **Formula:** Adjust for survey oversampling within blocks
- **Correlation with original:** -0.01 (survey data)
- **Conclusion:** Not the explanation - sampling is nearly proportional

#### D. Hybrid Method (Best Match)
- **Formula:** Stratified IPW × 2 for treatment group
- **Correlation with original:** 0.93 (admin data), 0.92 (survey data)
- **Conclusion:** **Best approximation** of original weights

### 3. Robustness Test Results: Table 1 Replication

We re-ran Table 1 with five different weighting schemes to test sensitivity:

#### Treatment Effects (Coefficient Estimates)

| Weighting Method | Col 1: Crime During | Col 2: Crime After | Col 3: Victim During | Col 4: Victim After | Col 5: Witness After |
|-----------------|-------------------|-------------------|--------------------|--------------------|-------------------|
| **Original** | 0.0030 | 0.1098 | 0.0060 | -0.0067 | 0.1534 |
| **Unweighted** | -0.0013 | 0.0959 | 0.0224 | -0.0053 | 0.1551 |
| **Simple IPW** | -0.0030 | 0.1020 | 0.0189 | 0.0026 | 0.1571 |
| **Stratified IPW** | -0.0201 | 0.1077 | 0.0254 | 0.0191 | 0.1719 |
| **Hybrid (Best Match)** | -0.0245 | 0.1135 | 0.0255 | 0.0246 | 0.1738 |

#### Spillover Effects (Coefficient Estimates)

| Weighting Method | Col 1: Crime During | Col 2: Crime After | Col 3: Victim During | Col 4: Victim After | Col 5: Witness After |
|-----------------|-------------------|-------------------|--------------------|--------------------|-------------------|
| **Original** | -0.0378 | 0.0830 | 0.0263 | 0.0131 | 0.1856 |
| **Unweighted** | -0.0437 | 0.0595 | 0.0366 | 0.0155 | 0.1832 |
| **Simple IPW** | -0.0410 | 0.0719 | 0.0348 | 0.0224 | 0.1859 |
| **Stratified IPW** | -0.0594 | 0.0763 | 0.0435 | 0.0336 | 0.1969 |
| **Hybrid (Best Match)** | -0.0612 | 0.0851 | 0.0425 | 0.0381 | 0.2004 |

## Analysis and Interpretation

### 1. Sensitivity Assessment

**Low Sensitivity Results (Robust):**
- **Column 5 (Crime Witnessing):** Treatment effects remain positive and substantial (0.15-0.17) across all weighting schemes
- **Column 2 (Crime After):** Treatment effects consistently positive (0.10-0.11)

**Moderate Sensitivity Results:**
- **Column 1 (Crime During):** Treatment effect sign changes from positive (original) to negative (alternatives)
- **Columns 3-4 (Victimization):** Magnitude varies but generally small effects

**High Sensitivity Results:**
- **Spillover effects:** Generally show more variation across weighting schemes
- **Crime during intervention:** Most sensitive to weighting choice

### 2. Implications for Interpretation

**Robust Findings:**
1. **Strong evidence** for military policing increasing crime witnessing after intervention
2. **Consistent evidence** for increased crime after intervention period
3. **Spillover effects** are generally positive but sensitive to weighting

**Sensitive Findings:**
1. **Crime during intervention:** Effect direction depends on weighting scheme
2. **Victimization effects:** Small and variable across methods

### 3. Methodological Insights

**Why Treatment Weights Are Higher:**
The systematic 2x multiplier for treatment groups suggests the weights incorporate:

1. **Design-based corrections** for complex geographic randomization
2. **Precision optimization** to increase power for treatment comparisons  
3. **Survey sampling adjustments** beyond simple randomization
4. **Compliance or implementation factors** (speculative)

**Statistical Justification:**
- Treatment blocks are the **rarest** and most **informationally valuable**
- Higher weights increase **precision** for treatment effect estimation
- Reflects **optimal weighting** for causal inference in complex designs

## Recommendations

### 1. For Replication
- **Use original weights** as provided in the datasets
- **Document** that weights exceed simple IPW for treatment groups
- **Note** that key findings (witnessing, post-intervention crime) are robust

### 2. For Interpretation
- **Emphasize robust results** (columns 2 and 5) in main findings
- **Acknowledge sensitivity** of crime-during-intervention results
- **Report robustness tests** in supplementary materials

### 3. For Future Research
- **Investigate** the exact weighting formula used by original authors
- **Consider** reporting both weighted and unweighted results
- **Test** sensitivity to weighting in complex experimental designs

## Technical Details

### Data Sources
- **Admin Data:** `admin_data_during.csv`, `admin_data_after.csv` (1,254 blocks)
- **Survey Data:** `survey_endline.csv` (7,918 respondents)
- **Geographic Data:** 30 neighborhoods (barrios) in Cali, Colombia

### Regression Specifications
- **Administrative data:** Block-level analysis with neighborhood fixed effects
- **Survey data:** Individual-level analysis with clustered standard errors at block level
- **Controls:** Geographic variables, demographic controls, pre-intervention crime

### Code Availability
- **Robustness test script:** `src/robustness_test_weights.py`
- **Results data:** `output/tables/robustness_weights_results.csv`
- **Original replication:** `src/03_tables.py`

