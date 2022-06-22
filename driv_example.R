 library("haven")
library("fastDummies")
library(dplyr)
library(nnls)
library(randomForest)
library(ranger)
library(xgboost)
library(glmnet)
library(SuperLearner)
library(tuneRanger)
library(mlr)
library(caret)

source(driv.R)
set.seed(123456)

data <- read_dta("data/ml_data/boy_haz.dta")

data <- dummy_cols(data, select_columns = "wave")
Y <- data$haz
X <- data %>%
  select(c("f_height", "f_weight", "m_height", "m_weight",        
           "f_edu_yr", "m_edu_yr", "c_age", "m_age_at_fb", "m_age_at_fb2",    
           "d_m_fhei", "d_m_mhei", "d_m_fwei", "d_m_mwei", "sex_ratio",        
           "birth_rate", "share_non_agr", "share_pri_ind", "share_sec_ind",  
           "lg_gdp_per_capita", "wave_1991", "wave_1993", "wave_1997",  
           "wave_2000", "wave_2004", "wave_2006",  "wave_2009",  "wave_2011"))
T <- data$nsib
Z <- data$coverage_percent
result <- driv.final(Y,X,T,Z,k=5, "rf")
haz_boy_cate <-  result$cate
haz_boy_rf <- result$forest

