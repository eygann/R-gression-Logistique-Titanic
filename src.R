#-------------------------------------------------------------------------------
# PROJECT: TITANIC SURVIVAL PREDICTION WITH LOGISTIC REGRESSION
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# 1. SETUP: LOADING LIBRARIES AND DATA
#-------------------------------------------------------------------------------

# Load necessary libraries
library(foreign)     # To read .arff files
library(tidyverse)   # A suite of packages for data science (dplyr, ggplot2, etc.)
library(gtsummary)   # To create publication-ready summary tables
library(caret)       # For machine learning workflows, including data splitting
library(pROC)        # For Receiver Operating Characteristic (ROC) curve analysis
library(questionr)   # For miscellaneous data analysis functions
library(guideR)      # For guided data analysis (used for plot_proportions)


# Load the dataset
tita <- read.arff("phpMYEkMl.arff")

# Display a basic summary of the initial data
summary(tita)


#-------------------------------------------------------------------------------
# 2. DATA CLEANING AND FEATURE ENGINEERING
#-------------------------------------------------------------------------------

### 2.1. Renaming and Recoding Variables ###

# Rename the passenger class factor levels for better readability
tita$pclass <- factor(tita$pclass,
                      levels = c(1, 2, 3),
                      labels = c("1st class", "2nd class", "3rd class"))

# Rename all columns for clarity and consistency
new_names <- c("pclass", "survived", "name", "sex", "age", "sibsp",
               "parch", "ticket", "fare" ,"cabin", "embarked", 
               "boat", "body", "home.dest")
names(tita) <- new_names


### 2.2. Discretizing Continuous and High-Cardinality Variables ###

# Discretize 'age' into five groups based on quintiles
# Note: include.lowest=TRUE is important to include the minimum value in a bin
age_quantiles <- quantile(tita$age, c(0, 0.2, 0.4, 0.6, 0.8, 1), na.rm = TRUE)
tita <- mutate(tita, age_group = cut(tita$age, age_quantiles, include.lowest = TRUE))

# Discretize 'sibsp' (number of siblings/spouses)
# First, ensure the variable is numeric before cutting
tita$sibsp <- as.numeric(as.character(tita$sibsp))
tita$sibsp <- cut(tita$sibsp, 
                  breaks = c(-Inf, 0, 1, Inf),
                  labels = c("0", "1", "2+"))

# Discretize 'parch' (number of parents/children)
tita$parch <- as.numeric(as.character(tita$parch))
tita$parch <- cut(tita$parch,
                  breaks = c(-Inf, 0, 1, Inf),
                  labels = c("0", "1", "2+"))

# Discretize 'fare' into four groups based on quartiles
fare_quantiles <- quantile(tita$fare, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE)
tita$fare_group <- cut(tita$fare,
                       breaks = fare_quantiles,
                       include.lowest = TRUE,
                       labels = c("Q1 (<=$8)", "Q2 ($8-$14.5)", "Q3 ($14.5-$31)", "Q4 (>$31)"))


#-------------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
#-------------------------------------------------------------------------------

### 3.1. Summary Table ###

# Create a summary table of key predictors, stratified by survival status
tita %>%
  select(survived, pclass, age_group, sex, fare_group) %>%
  tbl_summary(
    by = "survived",
    percent = "row"
  )

### 3.2. Bivariate Visualization ###

# Plot the proportion of survivors for each predictor category
# guideR's plot_proportions is used here as in the original script
# We convert 'survived' to a numeric (0/1) variable for the function to work correctly
tita$survived_num <- as.numeric(as.character(tita$survived))

tita %>%
  plot_proportions(
    survived_num == 1, # Proportion where survived is 1
    by = c("pclass", "age_group", "sex", "fare_group"),
    fill = "lightblue",
    flip = TRUE)


#-------------------------------------------------------------------------------
# 4. MODEL BUILDING
#-------------------------------------------------------------------------------

### 4.1. Data Splitting ###

# Ensure reproducible results
set.seed(123)

# Create an 80/20 stratified split based on the 'survived' outcome
index <- createDataPartition(tita$survived, p = 0.8, list = FALSE)
train_data <- tita[index, ]
valid_data <- tita[-index, ]


### 4.2. Training the Logistic Regression Model ###

# Build the model using the training data
model <- glm(data = train_data, 
             survived ~ pclass + age_group + sex + fare_group,
             family = binomial)

# Display the model coefficients and summary
summary(model)


#-------------------------------------------------------------------------------
# 5. MODEL EVALUATION
#-------------------------------------------------------------------------------

### 5.1. Making Predictions ###

# Predict survival probabilities on the validation set
probabilities <- predict(model, newdata = valid_data, type = "response")

# Convert probabilities to binary predictions using a 0.5 threshold
predictions <- ifelse(probabilities > 0.5, 1, 0)


### 5.2. ROC Curve and AUC Score ###

# Generate the ROC curve data
# Ensure the response variable is correctly formatted for pROC
roc_curve <- roc(response = valid_data$survived, predictor = probabilities)

# Calculate the AUC score
auc_score <- auc(roc_curve)
print(paste("AUC Score:", round(auc_score, 3)))

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for Titanic Survival Prediction", col = "blue", lwd = 2)
