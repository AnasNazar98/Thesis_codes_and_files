


rm(list = ls())
library(tidyverse)
library(skimr)
library(magrittr)
library(readxl)
library(writexl)
library(corrplot)
library(glmnet)
library(caret)
library(pROC)
library(xgboost)
library(PRROC)
library(tidymodels)
library(vip)
library(dials)
library(purrr)
library(tibble)
library(yardstick)
library(recipes)
library(finetune)
library(future)
library(themis)

library(lightgbm)
library(bonsai)  

################################################################################
# tidymodels xgboost falling_1
################################################################################
rm(list = ls())
seed <- 42

cross <- read_excel('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/cross_processed.xlsx')

gender <- read_xlsx('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/sex.xlsx')

cross$gender <- gender$gender

cross <- cross %>%
  mutate(across(everything(), ~ as.numeric(as.character(.))))


for (col in names(cross)) {
  unique_vals <- length(unique(na.omit(cross[[col]])))
  if (unique_vals <= 5) {
    cross[[col]] <- as.factor(cross[[col]])
  }
}

cross <- cross %>%
  mutate(across(
    where(is.factor),
    ~ if (all(levels(.) %in% c("1", "2"))) {
      factor(ifelse(. == "2", "0", "1"), levels = c("0", "1"))
    } else {
      .
    }
  ))

outcome <- factor(ifelse(cross$falling_1 == '1', 'Yes', 'No'), levels = c('Yes', 'No'))

cross <- cross %>% dplyr::select(-participant_id, -starts_with("falling_1"))

cross$falling_1 <- outcome

cross <- cross %>% mutate(case_wts = ifelse(falling_1 == "Yes", 21, 1), # work is 50 for thesis last report
                          case_wts = importance_weights(case_wts))

model <- 'LightGBM'
label <- 'Falling'


################################################################################
set.seed(seed)
data_split <- initial_split(cross, strata = falling_1, prop = 0.7)
data_train <- training(data_split)
data_test <- testing(data_split)

################################################################################

spec_default <- boost_tree() %>%
  set_engine("lightgbm") %>%
  set_mode("classification")


rec_default <- recipe(falling_1 ~ ., data = data_train) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) 

wf_default <- workflow() %>%
  add_recipe(rec_default) %>%
  add_model(spec_default) %>% add_case_weights(case_wts)

default_res <- last_fit(
  wf_default,
  split = data_split,
  metrics = metric_set(
    yardstick::f_meas,
    yardstick::precision,
    yardstick::recall,
    yardstick::spec,
    yardstick::accuracy,
    yardstick::bal_accuracy,
    yardstick::pr_auc
  )
)


collect_metrics(default_res)

preds <- collect_predictions(default_res) %>%
  mutate(.pred_class = factor(if_else(.pred_Yes >= 0.5, "Yes", "No"), levels = c("Yes", "No")))

collect_metrics(default_res)
conf_mat(preds, truth = falling_1, estimate = .pred_class)


fitted_model <- extract_fit_parsnip(default_res)

vip(fitted_model$fit, num_features = 10) +
  ggtitle(paste('Most predictive features for\n', label, 'using', model))

# this is the results for the thesis, based on the default parameters, 
# tuning did not yield better metrics for this outcome. 


################################################################################
set.seed(seed)
spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  learn_rate = tune()
) %>%
  set_engine("lightgbm", 
             lambda_l1 = tune(), 
             lambda_l2 = tune()
             , num_leaves = tune()) %>%
  set_mode("classification")


library(dials)
set.seed(seed)
params <- parameters(
  trees(),
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  learn_rate(),

  lambda_l1 = penalty(range = c(-5, 1)),  
  lambda_l2 = penalty(range = c(-5, 1))
  , num_leaves()
)

#################################################
# recipe
rec <- recipe(falling_1 ~ ., data = data_train) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) 

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(spec) %>% add_case_weights(case_wts)


set.seed(seed)
data_folds <- vfold_cv(data_train, strata = falling_1
                       , v = 5)

data_folds


library(future)
plan(multisession, workers = parallel::detectCores() - 4)


# Bayesian tuning for LightGBM
set.seed(seed)
(start_time <- Sys.time())
res <- tune_bayes(
  wf,
  resamples = data_folds,
  param_info = params,
  initial = 50,
  iter = 20,
  metrics = metric_set(
    yardstick::f_meas,
    yardstick::precision
    
  ),
  control = control_bayes(
    verbose = TRUE,
    no_improve = 10,
    seed = 123,
    save_pred = TRUE,
    allow_par = TRUE
  )
)
end_time <- Sys.time()
(parallel_time <- end_time - start_time)

falling_lgbm_res <- res



cross <- cross %>% 
  mutate(case_wts = ifelse(falling_1 == "Yes", 21, 1), 
         case_wts = importance_weights(case_wts))

set.seed(seed)
data_split <- initial_split(cross, strata = falling_1, prop = 0.70)
data_train <- training(data_split)
data_test <- testing(data_split)


collect_metrics(res)

best_parms <- best_auc <- select_best(res, metric = "precision")

spec <- boost_tree(
  trees = best_auc$trees,
  tree_depth = best_auc$tree_depth,
  min_n = best_auc$min_n,
  loss_reduction = best_auc$loss_reduction,
  sample_size = best_auc$sample_size,
  learn_rate = best_auc$learn_rate
) %>%
  set_engine("lightgbm",
             lambda_l1 = best_auc$lambda_l1,
             lambda_l2 = best_auc$lambda_l2,
             num_leaves = best_auc$num_leaves) %>%
  set_mode("classification")



#################################################
# recipe
rec <- recipe(falling_1 ~ ., data = data_train) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) 

final <- workflow() %>%
  add_recipe(rec) %>%
  add_model(spec) %>% add_case_weights(case_wts)

#################################################


set.seed(seed)
final_fit <- fit(final, data = data_train)

final_res <- last_fit(final, data_split, metrics = metric_set(
  yardstick::f_meas,
  yardstick::precision,
  yardstick::recall,
  yardstick::spec,
  yardstick::accuracy,
  yardstick::bal_accuracy,
  yardstick::pr_auc
))

collect_metrics(final_res)



preds <- collect_predictions(final_res) %>%
  mutate(.pred_class = factor(if_else(.pred_Yes >= 0.5, "Yes", "No"), levels = c("Yes", "No")))

conf_mat(preds, truth = falling_1, estimate = .pred_class)

label <- 'Falling'
model <- 'LightGBM'

vip(final_fit, num_features = 10) +
  ggtitle(paste('Most predictive features for\n', label, 'using', model))



