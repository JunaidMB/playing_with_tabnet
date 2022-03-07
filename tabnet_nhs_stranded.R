# Install development version of tabnet package
remotes::install_github("mlverse/tabnet")

library(tidymodels)
library(parameters)
library(skimr)
library(remotes)
library(tidyverse)
library(parallel)
library(doParallel)
library(vip)
library(themis)
library(lme4)
library(BradleyTerry2)
library(finetune)
library(butcher)
library(lobstr)
library(lubridate)
library(NHSRdatasets)
library(torch)
library(tabnet)
library(yardstick)

set.seed(777)
torch_manual_seed(777)

# Read in data ----
##  a stranded patient is a patient that has been in hospital for longer than 7 days and we also call these Long Waiters.
strand_pat <- NHSRdatasets::stranded_data %>% 
  setNames(c("stranded_class", "age", "care_home_ref_flag", "medically_safe_flag", 
             "hcop_flag", "needs_mental_health_support_flag", "previous_care_in_last_12_month", "admit_date", "frail_descrip")) %>% 
  mutate(stranded_class = factor(stranded_class),
         admit_date = as.Date(admit_date, format = "%d/%m/%Y")) %>% 
  drop_na()

# Partition into training and test data splits ----
split <- initial_split(strand_pat)
train_data <- training(split)
test_data <- testing(split)  

# Create Recipe ----
## Define Recipe to be applied to the dataset
stranded_rec <- 
  recipe(stranded_class ~ ., data = train_data) %>% 
  # Make a day of week and month feature from admit date and remove raw admit date
  step_date(admit_date, features = c("dow", "month")) %>% 
  step_rm(admit_date) %>% 
  # Upsample minority (positive) class
  themis::step_upsample(stranded_class, over_ratio = as.numeric(upsample_ratio)) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

## Prepare and Bake recipe on training and test data
stranded_recipe_prep <- prep(stranded_rec, training = train_data)

stranded_train_bake <- bake(stranded_recipe_prep, new_data = NULL)
stranded_test_bake <- bake(stranded_recipe_prep, new_data = test_data)

# hyperparameter settings (apart from epochs) as per the TabNet paper (TabNet-S)
tabnet_model <- tabnet(epochs = 5, batch_size = 256, decision_width = tune(), attention_width = tune(),
              num_steps = tune(), penalty = 0.000001, virtual_batch_size = 256, momentum = 0.6,
              feature_reusage = 1.5, learn_rate = tune()) %>%
  set_engine("torch", verbose = TRUE) %>%
  set_mode("classification")

# Create Workflow to connect recipe and model
tabnet_workflow <- workflow() %>%
  add_model(tabnet_model) %>%
  add_recipe(stranded_rec)

# Specify parameter tuning grid
grid <-
  tabnet_workflow %>%
  tune::parameters() %>%
  update(
    decision_width = decision_width(range = c(20, 40)),
    attention_width = attention_width(range = c(20, 40)),
    num_steps = num_steps(range = c(4, 6)),
    learn_rate = learn_rate(range = c(-2.5, -1))
  ) %>%
  grid_max_entropy(size = 8)

# Parameter Tuning ----
## Make Cross Validation folds
folds <- vfold_cv(train_data, v = 5)
set.seed(777)

## Apply win/loss tuning method
res <- tabnet_workflow %>% 
  tune_race_win_loss(
    resamples = folds,
    grid = grid,
    metrics = metric_set(roc_auc, accuracy),
    control = control_race()
  )

## View performance metrics across all hyperparameter permutations
res %>% 
  collect_metrics()


## Select the best model according to AUC
tabnet_best_model <- res %>% 
  select_best(metric = "roc_auc")

# Finalise the Model: Select best model ----

## Update the workflow with the model with the best hyperparameters (obtained from select_best())
final_tabnet_workflow <- tabnet_workflow %>% 
  finalize_workflow(res %>% 
                      select_best(metric = "roc_auc"))

## Fit the final model to the training data
final_tabnet_model <- final_tabnet_workflow %>% 
  fit(data = train_data)


## Pull model from the workflow
final_tabnet_model %>% 
  extract_fit_parsnip()

## Predict from final model
final_tabnet_model %>% 
  predict(train_data, type = "prob")


# Fit the model to the test data ----
## Use last_fit() this function fits the finalised model on the full training dataset and evaluates the finalised model on the testing data
tabnet_fit_final <- final_tabnet_model %>% 
  last_fit(split)

## Metrics on test set
tabnet_fit_final %>% 
  collect_metrics()

## Predictions on test set
tabnet_fit_final %>% 
  collect_predictions() %>%
  dplyr::select(starts_with(".pred")) %>% 
  bind_cols(test_data) 

## Confusion Matrix on test set
tabnet_fit_final %>% 
  collect_predictions() %>% 
  conf_mat(stranded_class, .pred_class)

## Generate ROC Curve
roc_plot <- 
  tabnet_fit_final %>% 
  collect_predictions() %>% 
  roc_curve(stranded_class, '.pred_Not Stranded') %>% 
  autoplot()


# Variable Importance - Tabnet Default

## Extract final fitted workflow
tabnet_wf_model <- tabnet_fit_final$.workflow[[1]]

tabnet_explain <- tabnet_explain(
  tabnet_wf_model %>% extract_fit_engine(),
  new_data = stranded_test_bake
)

autoplot(tabnet_explain)

# Save Model and Metrics ----

## Save Model
### Measure object size of workflow
obj_size(tabnet_wf_model)

### Weigh in the workflow, the objects that are taking up the most memory
weigh(tabnet_wf_model)

### Butcher workflow to take up less space
tabnet_wf_model_reduced <- butcher::butcher(tabnet_wf_model)

### Check size difference
print(obj_size(tabnet_wf_model))
print(obj_size(tabnet_wf_model_reduced))
obj_size(tabnet_wf_model) - obj_size(tabnet_wf_model_reduced) 

### Save model object as an RDS object
saveRDS(tabnet_wf_model_reduced, file = "./saved_models/tabnet_stranded.rds")

# Reading in workflow and predicting ----
## rm(rf_wf_model)
tabnet_wf_model <- readRDS(file = "./saved_models/tabnet_stranded.rds")

## Predict on test data with loaded workflow ----
test_sample <- test_data %>% 
  slice_sample(n = 50)

tabnet_wf_model %>% 
  predict(test_sample) %>% 
  cbind(stranded_class = test_sample$stranded_class)









