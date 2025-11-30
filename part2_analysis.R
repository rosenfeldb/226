# ============================================================================
# MS&E 226 Project Part 2: Inference and Causality
# Predicting Recidivism - R Code
# Authors: Ben Rosenfeld, Ness Arikan
# ============================================================================

# Load required libraries
library(tidyverse)
library(boot)
library(stats)
library(MASS)

# Set seed for reproducibility
set.seed(42)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

cat("Loading data...\n")

# Load training and test data
train_data <- read.csv("RearrestPrediction/RearrestPrediction_data_train.csv")
test_data <- read.csv("RearrestPrediction/RearrestPrediction_data_test.csv")

cat("Training data: ", nrow(train_data), " rows\n")
cat("Test data: ", nrow(test_data), " rows\n")

# Replicate preprocessing from Part 1 (matching Python notebook)
preprocess_data <- function(df) {
  # Remove rows with missing rearrest, male, or charge_weight
  df_clean <- df %>%
    filter(!is.na(rearrest), !is.na(male), !is.na(charge_weight)) %>%
    filter(charge_severity != " ", charge_severity != "", 
           !is.na(charge_severity))
  
  # Filter to only released defendants
  df_clean <- df_clean %>% filter(release == 1)
  
  # Handle missing values for request variables
  na_vars <- c('remand_requested', 'ror_requested', 'nmr_requested', 
               'no_release_requested', 'prior_vfo_cnt', 'prior_nonvfo_cnt', 
               'prior_misd_cnt', 'pending_vfo', 'pending_nonvfo', 'pending_misd')
  
  for (col in na_vars) {
    if (col %in% names(df_clean)) {
      df_clean[[paste0(col, '_missing')]] <- as.integer(is.na(df_clean[[col]]))
      df_clean[[col]][is.na(df_clean[[col]])] <- -1
    }
  }
  
  # Create charge_score from charge_weight
  charge_weight_order <- c('BM' = 1, 'UM' = 2, 'AM' = 3, 'EF' = 4, 
                          'DF' = 5, 'CF' = 6, 'BF' = 7, 'AF' = 8)
  df_clean$charge_score <- charge_weight_order[df_clean$charge_weight]
  
  # Create interaction term
  df_clean$black_nyc <- df_clean$black * df_clean$nyc
  
  # Rename missing indicators to match Part 1
  if ('no_release_requested_missing' %in% names(df_clean)) {
    df_clean <- df_clean %>%
      rename(requests_missing = no_release_requested_missing)
  }
  
  if ('pending_nonvfo_missing' %in% names(df_clean)) {
    df_clean <- df_clean %>%
      rename(pend_priors_missing = pending_nonvfo_missing)
  }
  
  # Drop unnecessary columns
  cols_to_drop <- c('judge_name', 'charge', 'charge_severity', 'charge_weight', 
                    'release', 'remand_requested_missing', 'ror_requested_missing', 
                    'nmr_requested_missing', 'pending_vfo_missing', 
                    'prior_misd_cnt_missing', 'prior_vfo_cnt_missing', 
                    'prior_nonvfo_cnt_missing', 'pending_misd_missing')
  
  # Remove columns that exist
  existing_cols_to_drop <- intersect(cols_to_drop, names(df_clean))
  if (length(existing_cols_to_drop) > 0) {
    df_clean <- df_clean %>%
      dplyr::select(-all_of(existing_cols_to_drop))
  }
  
  return(df_clean)
}

# Preprocess training data
train_clean <- preprocess_data(train_data)
cat("Cleaned training data: ", nrow(train_clean), " rows\n")

# Calculate scaling parameters from training data
age_mean_train <- mean(train_clean$age, na.rm = TRUE)
age_sd_train <- sd(train_clean$age, na.rm = TRUE)

charge_score_mean_train <- mean(train_clean$charge_score, na.rm = TRUE)
charge_score_sd_train <- sd(train_clean$charge_score, na.rm = TRUE)

# Scale training data
train_clean$age <- (train_clean$age - age_mean_train) / age_sd_train
train_clean$charge_score <- (train_clean$charge_score - charge_score_mean_train) / charge_score_sd_train

# Preprocess and scale test data using training parameters
test_clean <- preprocess_data(test_data)
cat("Cleaned test data: ", nrow(test_clean), " rows\n")

# Apply same scaling from training to test
test_clean$age <- (test_clean$age - age_mean_train) / age_sd_train
test_clean$charge_score <- (test_clean$charge_score - charge_score_mean_train) / charge_score_sd_train

# Ensure both datasets have the same columns
common_cols <- intersect(names(train_clean), names(test_clean))
common_cols <- common_cols[common_cols != "rearrest"]
train_clean <- train_clean %>% dplyr::select(all_of(common_cols), rearrest)
test_clean <- test_clean %>% dplyr::select(all_of(common_cols), rearrest)

cat("Final feature count: ", length(common_cols), "\n")

# ============================================================================
# 2. PREDICTION ON HOLDOUT SET
# ============================================================================

cat("\n=== SECTION 2: PREDICTION ON HOLDOUT SET ===\n")

# Note: The best model from Part 1 was a neural network (AUC = 0.70)
# For inference purposes, we'll fit a logistic regression model
# You can also load predictions from your Python neural network model if needed

# Fit logistic regression on training data (for comparison with Part 1)
model_lr_train <- glm(rearrest ~ ., data = train_clean, family = binomial)

# Predict on test set
test_pred_probs <- predict(model_lr_train, newdata = test_clean, type = "response")
test_pred_class <- ifelse(test_pred_probs > 0.5, 1, 0)

# Calculate test error metrics
test_log_loss <- -mean(test_clean$rearrest * log(test_pred_probs + 1e-15) + 
                        (1 - test_clean$rearrest) * log(1 - test_pred_probs + 1e-15))

test_accuracy <- mean(test_pred_class == test_clean$rearrest)

# Calculate AUC
library(pROC)
test_auc <- auc(test_clean$rearrest, test_pred_probs)

cat("Test Set Performance:\n")
cat("  Log Loss: ", round(test_log_loss, 4), "\n")
cat("  Accuracy: ", round(test_accuracy, 4), "\n")
cat("  AUC: ", round(as.numeric(test_auc), 4), "\n")

# Save predictions
test_results <- data.frame(
  actual = test_clean$rearrest,
  predicted_prob = test_pred_probs,
  predicted_class = test_pred_class
)
write.csv(test_results, "part2_test_predictions.csv", row.names = FALSE)

# ============================================================================
# 3. INFERENCE ANALYSIS
# ============================================================================

cat("\n=== SECTION 3: INFERENCE ANALYSIS ===\n")

# (a) Statistical significance of coefficients
cat("\n--- (a) Statistical Significance on Training Data ---\n")

summary_train <- summary(model_lr_train)
coef_summary <- summary_train$coefficients
alpha <- 0.05

significant_coefs <- coef_summary[coef_summary[, "Pr(>|z|)"] < alpha, ]

cat("Significance threshold: alpha =", alpha, "\n")
cat("Number of significant coefficients:", nrow(significant_coefs), "\n\n")

cat("Significant coefficients:\n")
print(significant_coefs)

# Focus on most practically meaningful coefficients
# (excluding intercept, focusing on key predictors)
key_vars <- c("age", "male", "black", "white", "nyc", "violent", 
              "pending_misd", "pending_nonvfo", "pending_vfo",
              "prior_misd_cnt", "prior_nonvfo_cnt", "charge_score")

key_coefs <- significant_coefs[rownames(significant_coefs) %in% key_vars, ]
if (nrow(key_coefs) > 0) {
  cat("\nKey significant coefficients:\n")
  print(key_coefs)
}

# (b) Fit model on test data and compare significance
cat("\n--- (b) Statistical Significance on Test Data ---\n")

model_lr_test <- glm(rearrest ~ ., data = test_clean, family = binomial)
summary_test <- summary(model_lr_test)
coef_summary_test <- summary_test$coefficients
significant_test <- coef_summary_test[coef_summary_test[, "Pr(>|z|)"] < alpha, ]

cat("Number of significant coefficients on TEST data:", nrow(significant_test), "\n\n")

# Compare which coefficients changed significance
train_sig_vars <- rownames(significant_coefs)
test_sig_vars <- rownames(significant_test)

only_train_sig <- setdiff(train_sig_vars, test_sig_vars)
only_test_sig <- setdiff(test_sig_vars, train_sig_vars)
both_sig <- intersect(train_sig_vars, test_sig_vars)

cat("Variables significant on BOTH training and test:", length(both_sig), "\n")
if (length(both_sig) > 0) {
  cat("  ", paste(both_sig, collapse = ", "), "\n")
}

cat("\nVariables significant ONLY on TRAINING data:", length(only_train_sig), "\n")
if (length(only_train_sig) > 0) {
  cat("  ", paste(only_train_sig, collapse = ", "), "\n")
  # Show p-values for these
  for (var in only_train_sig) {
    if (var %in% rownames(coef_summary)) {
      cat("    ", var, ": p =", round(coef_summary[var, "Pr(>|z|)"], 4), "\n")
    }
  }
}

cat("\nVariables significant ONLY on TEST data:", length(only_test_sig), "\n")
if (length(only_test_sig) > 0) {
  cat("  ", paste(only_test_sig, collapse = ", "), "\n")
  # Show p-values for these
  for (var in only_test_sig) {
    if (var %in% rownames(coef_summary_test)) {
      cat("    ", var, ": p =", round(coef_summary_test[var, "Pr(>|z|)"], 4), "\n")
    }
  }
}

# (c) Bootstrap confidence intervals
cat("\n--- (c) Bootstrap Confidence Intervals ---\n")

bootstrap_ci <- function(data, formula, n_bootstrap = 1000, alpha = 0.05) {
  n <- nrow(data)
  coef_names <- names(coef(glm(formula, data = data, family = binomial)))
  n_coefs <- length(coef_names)
  
  bootstrap_coefs <- matrix(NA, nrow = n_bootstrap, ncol = n_coefs)
  colnames(bootstrap_coefs) <- coef_names
  
  cat("Running", n_bootstrap, "bootstrap iterations...\n")
  pb <- txtProgressBar(min = 0, max = n_bootstrap, style = 3)
  
  set.seed(42)
  for (i in 1:n_bootstrap) {
    # Resample with replacement
    boot_sample <- data[sample(1:n, n, replace = TRUE), ]
    
    # Fit model
    boot_model <- tryCatch({
      glm(formula, data = boot_sample, family = binomial)
    }, error = function(e) NULL)
    
    if (!is.null(boot_model)) {
      boot_coefs <- coef(boot_model)
      # Handle case where some coefficients might be missing
      for (j in 1:n_coefs) {
        if (coef_names[j] %in% names(boot_coefs)) {
          bootstrap_coefs[i, j] <- boot_coefs[coef_names[j]]
        }
      }
    }
    setTxtProgressBar(pb, i)
  }
  close(pb)
  
  # Calculate confidence intervals
  ci_lower <- apply(bootstrap_coefs, 2, quantile, probs = alpha/2, na.rm = TRUE)
  ci_upper <- apply(bootstrap_coefs, 2, quantile, probs = 1 - alpha/2, na.rm = TRUE)
  
  return(data.frame(
    coefficient = coef_names,
    bootstrap_lower = ci_lower,
    bootstrap_upper = ci_upper,
    bootstrap_mean = apply(bootstrap_coefs, 2, mean, na.rm = TRUE)
  ))
}

# Get bootstrap CIs for training data
formula_lr <- as.formula(paste("rearrest ~", 
                               paste(setdiff(names(train_clean), "rearrest"), 
                                     collapse = " + ")))

cat("Computing bootstrap confidence intervals...\n")
bootstrap_cis <- bootstrap_ci(train_clean, formula_lr, n_bootstrap = 1000)

# Compare with standard regression CIs
standard_cis <- confint(model_lr_train, level = 0.95)
standard_cis_df <- data.frame(
  coefficient = rownames(standard_cis),
  standard_lower = standard_cis[, 1],
  standard_upper = standard_cis[, 2]
)

# Merge for comparison
ci_comparison <- merge(bootstrap_cis, standard_cis_df, by = "coefficient", all = TRUE)
ci_comparison$bootstrap_width <- ci_comparison$bootstrap_upper - ci_comparison$bootstrap_lower
ci_comparison$standard_width <- ci_comparison$standard_upper - ci_comparison$standard_lower
ci_comparison$width_difference <- ci_comparison$bootstrap_width - ci_comparison$standard_width

cat("\nBootstrap vs Standard Confidence Intervals (first 10 coefficients):\n")
print(head(ci_comparison, 10))

# Summary statistics
cat("\nSummary of CI width differences:\n")
cat("  Mean difference (bootstrap - standard):", 
    round(mean(ci_comparison$width_difference, na.rm = TRUE), 4), "\n")
cat("  Median difference:", 
    round(median(ci_comparison$width_difference, na.rm = TRUE), 4), "\n")

# (d) Multiple hypothesis testing correction
cat("\n--- (d) Multiple Hypothesis Testing Correction ---\n")

# Count number of tests (excluding intercept)
n_tests <- nrow(coef_summary) - 1
cat("Number of hypothesis tests:", n_tests, "\n")

# Bonferroni correction
bonferroni_alpha <- alpha / n_tests
cat("Bonferroni corrected alpha:", bonferroni_alpha, "\n")

significant_bonferroni <- coef_summary[coef_summary[, "Pr(>|z|)"] < bonferroni_alpha, ]
cat("Significant coefficients after Bonferroni correction:", 
    nrow(significant_bonferroni), "\n")

if (nrow(significant_bonferroni) > 1) {
  cat("\nBonferroni-significant coefficients:\n")
  print(significant_bonferroni)
}

# Benjamini-Hochberg procedure
p_values <- coef_summary[-1, "Pr(>|z|)"]  # Exclude intercept
p_adjusted_bh <- p.adjust(p_values, method = "BH")
names(p_adjusted_bh) <- names(p_values)

significant_bh <- names(p_adjusted_bh)[p_adjusted_bh < alpha]

cat("\nBenjamini-Hochberg corrected alpha: 0.05\n")
cat("Significant coefficients after Benjamini-Hochberg correction:", 
    length(significant_bh), "\n")

if (length(significant_bh) > 0) {
  cat("\nBH-significant coefficients:\n")
  bh_results <- data.frame(
    Variable = significant_bh,
    P_value = p_values[significant_bh],
    Adjusted_P_value = p_adjusted_bh[significant_bh]
  )
  print(bh_results)
}

# Compare results
cat("\nComparison of multiple testing corrections:\n")
cat("  Original significant (alpha = 0.05):", length(train_sig_vars) - 1, "coefficients\n")
cat("  After Bonferroni:", nrow(significant_bonferroni) - 1, "coefficients\n")
cat("  After Benjamini-Hochberg:", length(significant_bh), "coefficients\n")

# (e) Post-selection inference discussion
cat("\n--- (e) Post-Selection Inference ---\n")
cat("Note: This section requires written discussion in your report.\n")
cat("Key points to address:\n")
cat("  1. Model selection was done on training data\n")
cat("  2. Feature engineering and preprocessing choices were made\n")
cat("  3. Multiple models were compared (Lasso, Ridge, Neural Network)\n")
cat("  4. Hyperparameter tuning was performed\n")
cat("  5. These steps affect the validity of p-values and confidence intervals\n")

# ============================================================================
# 4. HELPER FUNCTIONS FOR REPORTING
# ============================================================================

# Function to create a comprehensive summary table
create_inference_summary <- function(model, alpha = 0.05) {
  coef_summary <- summary(model)$coefficients
  coef_df <- data.frame(
    Variable = rownames(coef_summary),
    Coefficient = coef_summary[, "Estimate"],
    Std_Error = coef_summary[, "Std. Error"],
    Z_value = coef_summary[, "z value"],
    P_value = coef_summary[, "Pr(>|z|)"],
    Significant = coef_summary[, "Pr(>|z|)"] < alpha
  )
  
  # Add confidence intervals - handle case where CI might have different rows
  tryCatch({
    ci <- confint(model, level = 1 - alpha)
    # Match CI rows to coefficient rows by name
    coef_df$CI_Lower <- NA
    coef_df$CI_Upper <- NA
    for (i in 1:nrow(coef_df)) {
      var_name <- coef_df$Variable[i]
      if (var_name %in% rownames(ci)) {
        coef_df$CI_Lower[i] <- ci[var_name, 1]
        coef_df$CI_Upper[i] <- ci[var_name, 2]
      }
    }
  }, error = function(e) {
    # If confint fails, use asymptotic CIs
    cat("Warning: Using asymptotic confidence intervals\n")
    z_crit <- qnorm(1 - alpha/2)
    coef_df$CI_Lower <- coef_df$Coefficient - z_crit * coef_df$Std_Error
    coef_df$CI_Upper <- coef_df$Coefficient + z_crit * coef_df$Std_Error
  })
  
  return(coef_df)
}

# Create summary tables
train_summary <- create_inference_summary(model_lr_train)
test_summary <- create_inference_summary(model_lr_test)

# Add multiple testing corrections to summary
train_summary$Bonferroni_Sig <- train_summary$P_value < (alpha / (nrow(train_summary) - 1))
train_summary$BH_Adjusted_P <- NA
train_summary$BH_Sig <- FALSE

p_vals_train <- train_summary$P_value[-1]  # Exclude intercept
p_adj_bh_train <- p.adjust(p_vals_train, method = "BH")
train_summary$BH_Adjusted_P[-1] <- p_adj_bh_train
train_summary$BH_Sig[-1] <- p_adj_bh_train < alpha

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================

cat("\n=== SAVING RESULTS ===\n")

# Save summary tables
write.csv(train_summary, "part2_train_inference_summary.csv", row.names = FALSE)
write.csv(test_summary, "part2_test_inference_summary.csv", row.names = FALSE)
write.csv(ci_comparison, "part2_bootstrap_ci_comparison.csv", row.names = FALSE)

# Save multiple testing results
multiple_testing_results <- data.frame(
  Variable = names(p_values),
  P_value = p_values,
  Bonferroni_Sig = p_values < bonferroni_alpha,
  BH_Adjusted_P = p_adjusted_bh,
  BH_Sig = p_adjusted_bh < alpha
)
write.csv(multiple_testing_results, "part2_multiple_testing_results.csv", row.names = FALSE)

# Save model objects
save(model_lr_train, model_lr_test, train_clean, test_clean, 
     train_summary, test_summary, ci_comparison,
     file = "part2_models.RData")

cat("Results saved to:\n")
cat("  - part2_train_inference_summary.csv\n")
cat("  - part2_test_inference_summary.csv\n")
cat("  - part2_bootstrap_ci_comparison.csv\n")
cat("  - part2_multiple_testing_results.csv\n")
cat("  - part2_test_predictions.csv\n")
cat("  - part2_models.RData\n")

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Next steps:\n")
cat("  1. Review the saved CSV files for your report\n")
cat("  2. Write up the interpretation of results\n")
cat("  3. Address post-selection inference in your discussion\n")
cat("  4. Complete the stakeholder guidance section\n")


