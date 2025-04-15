library(readr)
library(dplyr)
library(caret)

# Load the dataset
data <- read_csv("7_data.csv")

# Drop any ID column if it exists (adjust if needed)
if ("id" %in% colnames(data)) {
  data <- data %>% select(-id)
}

# Ensure the target variable exists
if (!"HeartDiseaseorAttack" %in% colnames(data)) {
  stop("Target column 'HeartDiseaseorAttack' not found. Please check column names.")
}

# Convert target to factor
data$HeartDiseaseorAttack <- as.factor(data$HeartDiseaseorAttack)

# Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(data$HeartDiseaseorAttack, p = 0.8, list = FALSE)
train <- data[train_index, ]

# Train logistic regression model
model <- glm(HeartDiseaseorAttack ~ ., data = train, family = "binomial")

# Save the trained model
saveRDS(model, "logistic_model.rds")

cat("Model trained and saved as logistic_model.rds\n")
