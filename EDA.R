# Load Libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(corrplot)
library(reshape2)
library(caret)
library(randomForest)
library(e1071)
library(gbm)
library(MLmetrics)

# Load Data
df <- read.csv("7_data.csv")

# Quick Summary
cat("Shape of dataset:", paste(dim(df), collapse = " x "), "\n\n")
cat("Sample data:\n")
print(head(df))
cat("\nData types:\n")
print(str(df))

# Check for Missing Values
missing <- colSums(is.na(df))
cat("\nMissing values in each column:\n")
if (any(missing > 0)) {
  print(missing[missing > 0])
} else {
  cat("No missing values found!\n")
}

# Target Variable Check
target <- "HeartDiseaseorAttack"
cat("\nClass distribution:\n")
df %>%
  count(!!sym(target)) %>%
  mutate(proportion = n / sum(n)) %>%
  print()

# Visualizing Class Balance
ggplot(df, aes_string(x = target)) +
  geom_bar(fill = "skyblue") +
  ggtitle("Class Distribution (Target: Heart Disease or Attack)") +
  xlab("Heart Disease (1 = Yes, 0 = No)") +
  ylab("Count")

# Drop ID Column if Present
if ("id" %in% names(df)) {
  df <- df %>% select(-id)
  cat("'id' column dropped.\n")
}

# Summary Statistics
cat("\nDescriptive statistics:\n")
print(summary(df))

# Correlation Heatmap
numeric_df <- df %>% select(where(is.numeric))
corr_matrix <- cor(numeric_df, use = "complete.obs")
corrplot(corr_matrix, method = "color", tl.col = "black", addCoef.col = "black", number.cex = 0.7)

# Distribution of Top Features
top_features <- c("HighBP", "HighChol", "Age", "Diabetes", "BMI", "Smoker", "PhysActivity", "DiffWalk")
df %>%
  select(all_of(top_features)) %>%
  pivot_longer(cols = everything()) %>%
  ggplot(aes(value)) +
  facet_wrap(~name, scales = "free", ncol = 4) +
  geom_histogram(bins = 20, fill = "steelblue", color = "black") +
  ggtitle("Distributions of Key Features") +
  theme_minimal()

# Compare Means for Target Classes
cat("\nMean values grouped by target class:\n")
df %>%
  group_by(across(all_of(target))) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE)) %>%
  pivot_longer(-all_of(target), names_to = "Feature", values_to = "Mean") %>%
  arrange(desc(Mean)) %>%
  head(10) %>%
  print()

ggplot(df, aes(x = HeartDiseaseorAttack)) +
  geom_bar(fill = "skyblue") +
  ggtitle("Target Class Distribution") +
  xlab("HeartDiseaseorAttack") +
  ylab("Count") +
  theme_minimal()

# Print class proportions (percentage)
df %>%
  count(HeartDiseaseorAttack) %>%
  mutate(percentage = round((n / sum(n)) * 100, 2)) %>%
  print()

# Boxplot: Age distribution by Heart Disease
ggplot(df, aes(x = as.factor(HeartDiseaseorAttack), y = Age)) +
  geom_boxplot(fill = "tomato", color = "black") +
  ggtitle("Age Distribution by Heart Disease") +
  xlab("Heart Disease (0 = No, 1 = Yes)") +
  ylab("Age") +
  theme_minimal()

# Histogram with density curve and class hue
ggplot(df, aes(x = BMI, fill = as.factor(HeartDiseaseorAttack))) +
  geom_histogram(aes(y = ..density..), bins = 30, alpha = 0.5, position = "identity", color = "black") +
  geom_density(aes(color = as.factor(HeartDiseaseorAttack)), size = 1.2) +
  scale_fill_manual(values = c("#00BFC4", "#F8766D"), name = "Heart Disease") +
  scale_color_manual(values = c("#00BFC4", "#F8766D"), name = "Heart Disease") +
  ggtitle("BMI Distribution by Class") +
  xlab("BMI") +
  ylab("Density") +
  theme_minimal()

# Count plot: Smoker status by Heart Disease
ggplot(df, aes(x = as.factor(Smoker), fill = as.factor(HeartDiseaseorAttack))) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("#00BFC4", "#F8766D"), name = "Heart Disease") +
  xlab("Smoker") +
  ylab("Count") +
  ggtitle("Smoker Status by Heart Disease") +
  theme_minimal()

# Compute correlation matrix (for numeric columns only)
numeric_df <- df %>% select(where(is.numeric))
corr_matrix <- cor(numeric_df, use = "complete.obs")

# Heatmap: Correlation matrix (no annotations)
corrplot(corr_matrix,
         method = "color",
         col = colorRampPalette(c("blue", "white", "red"))(200),
         tl.col = "black",
         tl.cex = 0.8,
         cl.cex = 0.8,
         addCoef.col = NA,  # No annotations
         mar = c(0, 0, 2, 0),
         title = "Feature Correlation Heatmap")

df <- df %>% select(-any_of("id"))

# 3. Define target and features
target <- "HeartDiseaseorAttack"
X <- df %>% select(-all_of(target))
y <- df[[target]] %>% as.factor()  # Ensure it's a factor for classification

# 4. Train/test split
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# 5. Train a basic Random Forest model
rf_model <- randomForest(x = X_train, y = y_train,
                         ntree = 100, importance = TRUE,
                         classwt = c("0" = 1, "1" = 1),  # balanced weights
                         random_state = 42)

# 6. Predictions
y_pred <- predict(rf_model, X_test)

# 7. Evaluation metrics
conf_mat <- confusionMatrix(y_pred, y_test, positive = "1")

accuracy <- conf_mat$overall["Accuracy"]
precision <- conf_mat$byClass["Precision"]
recall <- conf_mat$byClass["Recall"]
f1 <- conf_mat$byClass["F1"]

# 8. Detailed report
cat("Random Forest Classifier Evaluation:\n")
cat("Overall Accuracy:", round(accuracy * 100, 2), "%\n")
cat("Precision (Class 1):", round(precision * 100, 2), "%\n")
cat("Recall (Class 1):", round(recall * 100, 2), "%\n")
cat("F1 Score (Class 1):", round(f1 * 100, 2), "%\n\n")

cat("Classification Report:\n")
print(conf_mat)

# 2. Drop 'id' column if present
df <- df %>% select(-any_of("id"))

# 3. Define features and target
target <- "HeartDiseaseorAttack"
X <- df %>% select(-all_of(target))
y <- as.factor(df[[target]])

# 4. Train/test split
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# 5. Train Random Forest (class weight balanced)
# Estimate class weights manually for balancing
class_weights <- table(y_train)
total <- sum(class_weights)
weights <- total / (length(class_weights) * class_weights)

rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 100,
  importance = TRUE,
  classwt = weights,
  random_state = 42
)

# 6. Predictions
y_pred <- predict(rf_model, X_test)

# 7. Evaluation metrics
conf_mat <- confusionMatrix(y_pred, y_test, positive = "1")

accuracy <- conf_mat$overall["Accuracy"]
precision <- conf_mat$byClass["Precision"]
recall <- conf_mat$byClass["Recall"]
f1 <- conf_mat$byClass["F1"]

# 8. Classification Report
cat("Random Forest (class weight = 'balanced') Evaluation:\n")
cat("Overall Accuracy:", round(accuracy * 100, 2), "%\n")
cat("Precision (Class 1):", round(precision * 100, 2), "%\n")
cat("Recall (Class 1):", round(recall * 100, 2), "%\n")
cat("F1 Score (Class 1):", round(f1 * 100, 2), "%\n\n")

cat("Classification Report:\n")
print(conf_mat)

# 1. Drop 'id' column if present
df <- df %>% select(-any_of("id"))

# 2. Define target and features
target <- "HeartDiseaseorAttack"
X <- df %>% select(-all_of(target))
y <- as.factor(df[[target]])

# 3. Train/test split
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# 4. Train Gradient Boosting Model
# Combine features and target for gbm training
train_data <- X_train
train_data$target <- y_train

# Convert target to numeric for gbm (required)
train_data$target <- as.numeric(as.character(train_data$target))

gb_model <- gbm(
  formula = target ~ .,
  distribution = "bernoulli",
  data = train_data,
  n.trees = 100,
  interaction.depth = 3,
  shrinkage = 0.1,
  n.minobsinnode = 10,
  verbose = FALSE
)

# 5. Predictions
pred_probs <- predict(gb_model, X_test, n.trees = 100, type = "response")
y_pred <- ifelse(pred_probs > 0.5, 1, 0)
y_pred <- as.factor(y_pred)
y_test <- as.factor(as.numeric(as.character(y_test)))

# 6. Evaluation metrics
conf_mat <- confusionMatrix(y_pred, y_test, positive = "1")

accuracy <- conf_mat$overall["Accuracy"]
precision <- conf_mat$byClass["Precision"]
recall <- conf_mat$byClass["Recall"]
f1 <- conf_mat$byClass["F1"]

# 7. Classification report
cat("Gradient Boosting Classifier Evaluation:\n")
cat("Overall Accuracy:", round(accuracy * 100, 2), "%\n")
cat("Precision (Class 1):", round(precision * 100, 2), "%\n")
cat("Recall (Class 1):", round(recall * 100, 2), "%\n")
cat("F1 Score (Class 1):", round(f1 * 100, 2), "%\n\n")

cat("Classification Report:\n")
print(conf_mat)

# 1. Logistic Regression with class weights
# Create class weights
class_weights <- table(y_train)
total <- sum(class_weights)
weights <- total / (length(class_weights) * class_weights)
wts <- ifelse(y_train == "1", weights["1"], weights["0"])

# Train weighted logistic regression
logit_model <- glm(y_train ~ ., data = X_train, family = binomial(), weights = wts)

# Predict on test set
logit_probs <- predict(logit_model, X_test, type = "response")
logit_preds <- ifelse(logit_probs > 0.5, "1", "0") %>% as.factor()

# 2. Naive Bayes
nb_model <- naiveBayes(x = X_train, y = y_train)
nb_preds <- predict(nb_model, X_test)

# 3. Evaluation function
evaluate_model <- function(name, y_true, y_pred) {
  cm <- confusionMatrix(y_pred, y_true, positive = "1")
  
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  f1 <- cm$byClass["F1"]
  
  cat("\n", name, "Results\n")
  cat("Accuracy:", round(accuracy * 100, 2), "%\n")
  cat("Class 1 Precision:", round(precision * 100, 2), "%\n")
  cat("Class 1 Recall:", round(recall * 100, 2), "%\n")
  cat("Class 1 F1-Score:", round(f1 * 100, 2), "%\n")
  cat("\nClassification Report:\n")
  print(cm)
  
  return(tibble(
    Model = name,
    Accuracy = round(accuracy * 100, 2),
    Precision = round(precision * 100, 2),
    Recall = round(recall * 100, 2),
    F1 = round(f1 * 100, 2)
  ))
}

# 4. Evaluate both models
lr_results <- evaluate_model("Logistic Regression (weighted)", y_test, logit_preds)
nb_results <- evaluate_model("Naive Bayes", y_test, nb_preds)

# Assuming you already have the model and predictions
# Set custom threshold
custom_threshold <- 0.35
y_pred_custom <- as.integer(y_probs >= custom_threshold)

# Create classification report equivalent in R
conf_matrix <- confusionMatrix(factor(y_pred_custom, levels = c(0, 1)), factor(y_test, levels = c(0, 1)))

# Print confusion matrix and other metrics
print(paste("Classification Report (Threshold =", custom_threshold, "):"))
print(conf_matrix)

# For additional metrics like precision, recall, F1 Score, you can use:
precision <- Precision(y_test, y_pred_custom)
recall <- Recall(y_test, y_pred_custom)
f1 <- F1_Score(y_test, y_pred_custom)

# Print precision, recall, and F1 Score
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1, "\n")

# Assuming you already have the y_pred_custom and y_test values
accuracy <- Accuracy(y_test, y_pred_custom)
f1_macro <- F1_Score(y_test, y_pred_custom, average = "macro")
f1_weighted <- F1_Score(y_test, y_pred_custom, average = "weighted")

# Print the results
cat(sprintf("Accuracy: %.2f%%\n", accuracy * 100))
cat(sprintf("Macro Avg F1 Score: %.4f\n", f1_macro))
cat(sprintf("Weighted Avg F1 Score: %.4f\n", f1_weighted))

# Set theme for ggplot
theme_set(theme_bw())

# Features to analyze
top_features <- c('HighBP', 'HighChol', 'Age', 'Stroke', 'PhysActivity', 'Diabetes', 
                  'BMI', 'DiffWalk', 'Smoker', 'Education', 'Income', 
                  'AnyHealthcare', 'NoDocbcCost', 'MentalHealth')

# Target variable
target <- 'HeartDiseaseorAttack'

# 1. Barplots for binary/categorical features
binary_features <- c('HighBP', 'HighChol', 'Stroke', 'PhysActivity', 'Diabetes', 
                     'DiffWalk', 'Smoker', 'AnyHealthcare', 'NoDocbcCost')

for (col in binary_features) {
  p <- ggplot(df, aes_string(x = col, fill = factor(target))) + 
    geom_bar(position = "fill") + 
    labs(title = paste("Heart Disease Rate by", col), 
         y = "Proportion with Heart Disease", 
         x = col) + 
    scale_fill_manual(values = c("blue", "red"), labels = c("No Heart Disease", "Heart Disease")) + 
    theme_minimal()
  
  print(p)
}

# 2. Boxplots for numeric features
numeric_features <- c('Age', 'BMI', 'Education', 'Income', 'MentalHealth')

for (col in numeric_features) {
  p <- ggplot(df, aes_string(x = target, y = col)) + 
    geom_boxplot() + 
    labs(title = paste(col, "by Heart Disease Status"),
         x = "Heart Disease (0 = No, 1 = Yes)", 
         y = col) +
    theme_minimal()
  
  print(p)
}

# 3. Correlation Heatmap (optional)
cor_matrix <- cor(df[, c(top_features, target)], use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45, addCoef.col = "black", 
         title = "Correlation Heatmap of Top Influential Features")


