# MS&E 226 Project Part 2: Inference and Causality

This directory contains the R code for Part 2 of the MS&E 226 project on predicting recidivism.

## Files

- `part2_analysis.R` - Main R script containing all analyses for Part 2
- `RearrestPrediction/` - Directory containing data files and Part 1 code

## Required R Packages

Install the following packages before running the script:

```r
install.packages(c("tidyverse", "boot", "MASS", "pROC"))
```

## Running the Analysis

1. Make sure you're in the project directory (`226ProjectPart2`)
2. Open R or RStudio
3. Run the script:
   ```r
   source("part2_analysis.R")
   ```

## Output Files

The script generates several output files:

1. **part2_train_inference_summary.csv** - Summary of coefficients, p-values, and significance on training data
2. **part2_test_inference_summary.csv** - Summary of coefficients, p-values, and significance on test data
3. **part2_bootstrap_ci_comparison.csv** - Comparison of bootstrap vs standard confidence intervals
4. **part2_multiple_testing_results.csv** - Results of Bonferroni and Benjamini-Hochberg corrections
5. **part2_test_predictions.csv** - Predictions on the test set
6. **part2_models.RData** - Saved R model objects for later use

## Analysis Sections

The script performs the following analyses:

### 1. Prediction on Holdout Set
- Applies logistic regression model to test set
- Calculates test error (log loss), accuracy, and AUC
- **Note**: Your best model from Part 1 was a neural network. You may want to:
  - Load predictions from your Python model, or
  - Use `reticulate` to call your Python model from R

### 2. Inference Analysis

#### (a) Statistical Significance
- Reports which coefficients are statistically significant (α = 0.05)
- Highlights most practically meaningful coefficients

#### (b) Test vs Training Comparison
- Fits model on test data
- Compares which coefficients remain significant
- Identifies differences and potential reasons

#### (c) Bootstrap Confidence Intervals
- Computes 1000 bootstrap samples
- Compares bootstrap CIs with standard regression CIs
- Discusses which to report to stakeholders

#### (d) Multiple Hypothesis Testing
- Applies Bonferroni correction
- Applies Benjamini-Hochberg procedure
- Compares results and discusses which to use

#### (e) Post-Selection Inference
- Provides framework for discussion
- Notes that model selection affects inference validity

## Notes

- The preprocessing matches Part 1 (Python notebook) preprocessing
- Scaling parameters from training data are applied to test data
- Bootstrap analysis may take a few minutes (1000 iterations)
- All results are saved to CSV files for easy inclusion in your report

## Next Steps

1. Review the output CSV files
2. Write up interpretations for your report
3. Address the stakeholder guidance questions:
   - How do inference results inform stakeholder decisions?
   - Which relationships require causal interpretation?
   - What confounding variables might exist?
   - What additional data would strengthen causal claims?


