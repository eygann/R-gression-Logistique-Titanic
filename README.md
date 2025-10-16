# Project: Titanic Survival Prediction with Logistic Regression

### **Objective**

This document provides a step-by-step walkthrough of an R script designed to predict passenger survival on the RMS Titanic. We will use a **logistic regression** model, a standard statistical method for binary classification.

The process involves these key stages:

1. **Setup**: Loading necessary libraries and the dataset.
  
2. **Data Cleaning & Feature Engineering**: Preparing the data for analysis by renaming variables and transforming features.
  
3. **Exploratory Data Analysis (EDA)**: Understanding the relationships between variables and survival.
  
4. **Model Building**: Training a logistic regression model.
  
5. **Model Evaluation**: Assessing the model's predictive performance using a validation set.
  

---

## 1. Setup: Loading Libraries and Data

First, we load the R packages required for data manipulation, visualization, and modeling.

```
# Load necessary libraries
library(foreign) # To read .arff files
library(questionr) # For data exploration
library(labelled) # To work with variable labels
library(forcats) # For handling categorical variables (factors)
library(tidyverse) # A suite of packages for data science (dplyr, ggplot2, etc.)
library(gtsummary) # To create publication-ready summary tables
library(guideR) # For guided data analysis
library(breakDown) # For model interpretability
library(caret) # For machine learning workflows, including data splitting
library(pROC) # For Receiver Operating Characteristic (ROC) curve analysis
```

Next, we load the Titanic dataset, which is stored in the `.arff` format.

```
# Load the dataset
tita <- read.arff("phpMYEkMl.arff")

# Display a basic summary of the initial data
summary(tita)
```

---

## 2. Data Cleaning and Feature Engineering

This data isn't ready for analysis, I will cover the essential preprocessing steps to make the data more meaningful and suitable for our model.

### 2.1. Renaming and Recoding Variables

The original variable names are not descriptive, and some categorical variables need clearer labels.

```
# Rename the passenger class factor levels for better readability
tita$pclass <- factor(tita$pclass,
                      levels = c(1, 2, 3),
                      labels = c("1st class", "2nd class", "3rd class"))

# Rename all columns for clarity and consistency
new_names <- c("pclass", "survived", "name", "sex", "age", "sibsp",
               "parch", "ticket", "fare" ,"cabin", "embarked", 
               "boat", "body", "home.dest")
names(tita) <- new_names
```

### 2.2. Discretizing Continuous and High-Cardinality Variables

To improve model performance and interpretability, we convert several numerical variables into categorical ones (a process called **discretization** or **binning**).

**Age:** We transform the continuous `age` variable into five distinct groups based on quintiles. This helps the model capture non-linear relationships between age and survival.

```
# Create age groups based on quintiles
q <- quantile(tita$age, c(0, 0.2, 0.4, 0.6, 0.8, 1), na.rm = TRUE)
tita <- mutate(tita, age_group = cut(tita$age, q, include.lowest = TRUE))
```

**Family Size:** The `sibsp` (siblings/spouses) and `parch` (parents/children) variables have many distinct values. We group them into smaller, more meaningful categories to prevent overfitting.

```
# Group the number of siblings/spouses
tita$sibsp <- as.numeric(as.character(tita$sibsp)) # Ensure numeric type
tita$sibsp <- cut(tita$sibsp, 
                  breaks = c(-Inf, 0, 1, Inf),
                  labels = c("0", "1", "2+"))

# Group the number of parents/children
tita$parch <- as.numeric(as.character(tita$parch)) # Ensure numeric type
tita$parch <- cut(tita$parch,
                  breaks = c(-Inf, 0, 1, Inf),
                  labels = c("0", "1", "2+"))
```

**Fare:** The ticket fare is a heavily skewed continuous variable. We bin it into quartiles to create four price categories.

```
# Create fare categories based on quartiles
q_fare <- quantile(tita$fare, na.rm = TRUE)
tita$fare_group <- cut(tita$fare,
                       breaks = q_fare,
                       include.lowest = TRUE,
                       labels = c("Q1 (<=$8)", "Q2 ($8-$14.5)", "Q3 ($14.5-$31)", "Q4 (>$31)"))
```

---

## 3. Exploratory Data Analysis

Before building the model, we explore the data to identify which variables are most strongly associated with survival.

### 3.1. Summary Table

The `tbl_summary` function from the `gtsummary` package creates a concise table comparing the characteristics of survivors and non-survivors.

```
# Create a summary table of key predictors, stratified by survival status
tita %>%
  tbl_summary(
    by = "survived",
    include = c("pclass", "age_group", "sex", "fare_group"),
    percent = "row"
  )
```

This table provides initial evidence that `pclass`, `age`, `sex`, and `fare` are strong candidates for our predictive model.

### 3.2. Bivariate Visualization

A visual inspection confirms these relationships. We plot the proportion of survivors for each category of our selected predictors.

```
# Plot the proportion of survivors for each predictor category
tita %>%
  plot_proportions(
    survived == 1,
    by = c("pclass", "age_group", "sex", "fare_group"),
    fill = "lightblue",
    flip = TRUE)
```

The plot visually demonstrates higher survival rates for passengers in 1st class, females, and those who paid a higher fare.

---

## 4. Model Building

Now we proceed with building the logistic regression model.

### 4.1. Data Splitting

To evaluate our model's performance on unseen data, we must split our dataset into a **training set** (to build the model) and a **validation set** (to test it). We use an 80/20 split.

Setting a `seed` ensures that this random split is **reproducible**.

```
# Ensure reproducible results
set.seed(123)

# Create an 80/20 stratified split based on the 'survived' outcome
index <- createDataPartition(tita$survived, p = 0.8, list = FALSE)
train_data <- tita[index, ]
valid_data <- tita[-index, ]
```

### 4.2. Training the Logistic Regression Model

We train a **Generalized Linear Model (GLM)** using the `glm()` function. By specifying `family = binomial`, we instruct R to build a logistic regression model, which is appropriate for a binary outcome (survived vs. not survived).

The model is trained *only* on the `train_data`.

```
# Build the model using the training data
model <- glm(data = train_data, 
             survived ~ pclass + age_group + sex + fare_group,
             family = binomial)

# Display the model coefficients and summary
summary(model)
```

---

## 5. Model Evaluation

The final step is to evaluate how well our trained model predicts survival on the `valid_data`, which it has never seen before.

### 5.1. Making Predictions

We use the `predict()` function to generate survival probabilities for the validation set. A probability greater than 0.5 is classified as "Survived" (1), and less than or equal to 0.5 as "Not Survived" (0).

```
# Predict survival probabilities on the validation set
probabilities <- predict(model, newdata = valid_data, type = "response")

# Convert probabilities to binary predictions using a 0.5 threshold
predictions <- ifelse(probabilities > 0.5, 1, 0)
```

### 5.2. ROC Curve and AUC Score

A powerful tool for evaluating a binary classifier is the **Receiver Operating Characteristic (ROC) curve**. It plots the model's true positive rate against its false positive rate across all possible thresholds.

The **Area Under the Curve (AUC)** provides a single metric to summarize the model's performance.

- **AUC = 1.0**: Perfect model.
  
- **AUC = 0.5**: No better than random chance.
  

```
# Generate the ROC curve data
roc_curve <- roc(valid_data$survived, probabilities)

# Calculate the AUC score
auc_score <- auc(roc_curve)
print(paste("AUC Score:", round(auc_score, 3)))

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for Titanic Survival Prediction", col = "blue", lwd = 2)
```

A high AUC score (typically > 0.75) indicates that the model has strong predictive power.
