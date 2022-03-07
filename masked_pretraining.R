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

# Masked Pretraining. Process involves Unsupervised learning on a similar dataset first, then supervised training. We're creating our own pretrained model.
data("lending_club", package = "modeldata")

split <- initial_split(lending_club, strata = Class, prop = 9/10)

unsupervised <- training(split) %>% mutate(Class=NA) ## No class labels
supervised <- testing(split)

# recipe, prep and baking
prep_unsup <- recipe(Class ~ ., unsupervised) %>% 
  step_normalize(all_numeric()) %>% 
  prep()

unsupervised_baked_df <- prep_unsup %>% 
  bake(new_data=NULL)

## Unsupervised training first
pretrained_model <- tabnet_pretrain(x = unsupervised_baked_df %>%  select(-Class), y = NULL, epochs = 25, valid_split = 0.2, verbose = TRUE)

# Now apply to supervised set
split_s <- initial_split(supervised, strate = Class)
train <- training(split_s)

supervised_train_df <- prep_unsup %>% 
  bake(new_data = train)

model_fit <- tabnet_fit(x = supervised_train_df %>% select(-Class),
                        y = supervised_train_df$Class,
                        tabnet_model = pretrained_model,
                        valid_split = 0.2,
                        epochs = 10, 
                        verbose = TRUE)

# Explainability
# Unsupervised
pretrain_explain <- tabnet_explain(
  pretrained_model,
  new_data = unsupervised_baked_df
)
autoplot(pretrain_explain)

# Supervised
model_explain <- tabnet_explain(
  model_fit,
  new_data = supervised_train_df
)
autoplot(model_explain)


