# Credit Risk Modeling in Loans
# ls() - LIsts all the objects
# getwd() - Where the path
# setwd() - Set the path where the data is

# Load the Home Loan Data
loan_data <- readRDS("Loan_Data.rds")

# Explore the data
# 11948 rows and 12 columns in this credit risk loan dataset 
rowSums(loan_data)
colSums(loan_data)
names(loan_data)

View(loan_data)
names(loan_data)
dim(loan_data)
str(loan_data)
summary(loan_data)

# Convert the Data to CSV
library('rio')
# Create file to convert
export(loan_data, "Loan_Data.rds")
# convert RDS to CSV
convert("Loan_Data.rds", "Loan_Data.csv")


# Tools for model fitting
install.packages("gmodels")
library(gmodels)

CrossTable(loan_data$loan_status)
# ll.1% is loan defaults

# Call CrossTable() on grade and loan_status
CrossTable(loan_data$grade, loan_data$loan_status, prop.r = TRUE, prop.c = FALSE,
           prop.t = FALSE, prop.chisq = FALSE)
# The prportion of defaults increases when the credit rating moves from A to G

# Plot histogram for Loan 
hist_1 <- hist(loan_data$loan_amnt)
# Prints out locations of the breaks in hist_1
hist_1$breaks # shows Right_skewed distribution


# Change number of breaks and add labels of x-axis and title: hist_2
hist_2 <- hist(loan_data$loan_amnt, breaks = 200, xlab = "Loan amount", main = "Histogram of the loan amount")

# Plot the age variable
plot(loan_data$age, ylab = "Age")

# Save the outlier's index to index_highage
index_highage <- which(loan_data$age > 122)

# Create data set new_data with outlier deleted
new_data <- loan_data[-index_highage, ]

# Make bivariate scatterplot of age and annual income with x and y label
plot(loan_data$age, loan_data$annual_inc, xlab = "Age", ylab = "Annual Income")

hist(loan_data$int_rate, main = "Histogram of Interest Rate", xlab = "Interest rate")

# ggplot2 package - Color Graphics
library(ggplot2)
ggplot(data = loan_data, aes(x = int_rate)) + geom_histogram(binwidth = 1, na.rm = TRUE)

# Annual income
hist(loan_data$annual_inc, main = "Histogram of Annual Income", xlab = "Annual Income")
# Histogram breaks
hist_income <- hist(loan_data$annual_inc, main = "Histogram of Annual Income", xlab = "Annual Income")


# Plot Boxplot
boxplot(age ~ annual_inc, data = loan_data, xlab = "Age", ylab = "Annual Income")
boxplot(annual_inc ~ loan_status, data = loan_data, xlab = "Loan Status", ylab = "Annual Income")




# Missing data in "Loan Data"
# Observe the data
# Check the data set for any missing values 
any(is.na(loan_data))
summary(loan_data$int_rate) # 2776 Missing data

# Get indices of missing interest rates: na_index
na_index <- which(is.na(loan_data$int_rate))

# Remove NA in rows
loan_data_delrow_na <- loan_data[-na_index, ]

# Make copy of Loan Data
loan_data_delcol_na <- loan_data

# Delete interest rate column from loan_data_delcol_na
loan_data_delcol_na$int_rate <- NULL

# Replacing missing data
median_ir <- median(loan_data$int_rate, na.rm = TRUE)
# Make copy of loan_data
loan_data_replace <- loan_data

loan_data_replace$int_rate[na_index] <- median_ir

# Check if the NAs are gone
summary(loan_data_replace$int_rate)


# Keeping missing data
# Keep NA and separate the NA in missing category
# Make the necessary replacements in the coarse classification example below
loan_data$ir_cat <- rep(NA, length(loan_data$int_rate))

loan_data$ir_cat[which(loan_data$int_rate <= 8)] <- "0-8"
loan_data$ir_cat[which(loan_data$int_rate > 8 & loan_data$int_rate <= 11)] <- "8-11"
loan_data$ir_cat[which(loan_data$int_rate > 11 & loan_data$int_rate <= 13.5)] <- "11-13.5"
loan_data$ir_cat[which(loan_data$int_rate > 13.5)] <- "13.5+"
loan_data$ir_cat[which(is.na(loan_data$int_rate))] <- "Missing"

loan_data$ir_cat <- as.factor(loan_data$ir_cat)

# Look at your new variable using plot()
plot(loan_data$ir_cat)


# To prepare the binary response variable for the logistic regression model, 
# create a new variable, charge_off, derived from the loan_status, wherein 1 denotes 
# a loan charge off (loan cannot be recovered by the bank) and 0 denotes FALSE (loan still outstanding) 
loan_data <- loan_data %>% mutate(charge_off = ifelse(loan_status == "Charged Off", 1, 0))

# Examine the output of the newly created variable using a Cross Table
CrossTable(loan_data$charge_off, prop.r = T, prop.c = FALSE, prop.chisq = F, prop.t = F)
# The table shows that 12% of loans issued by the credit company defaulted and have been charged-off 
# There are 1428 cases of charge-off loans out of 11948 data points in the data set 



# Data splitting and confusion matrices
# Train Set - Run the Model
# Test Set - Evaluate the Result

# Splitting the data set
# Set seed of 567
set.seed(567)

# Store row numbers for training set: index_train
index_train <- sample(1:nrow(loan_data), 2/3 * nrow(loan_data))

# Training set: run the model (2/3 of the original data in Loan Data)
# Test set: evaluate the result (1/3 of the original data in Loan Data)
# Create training set: training_set
training_set <- loan_data[index_train, ]

# Create test set: test_set
test_set <- loan_data[-index_train, ]


# Logistic Regression #
# Fit the Logistic Regression Model with the charge_off column as our binary response variable 
# Use the training_set to build the model 
logistic.model <- glm(charge_off ~ home_ownership + annual_inc + loan_amnt + term + int_rate + grade + fico_score + inq_last_6mths, family = "binomial", data = training_set)

# Obtain the significance levels of the variables using the summary() function 
# The statistically significant variables based on the p-value are annual_inc, term, grade, inq_last_6_mths 
# The variables home_ownership, loan_amnt, int_rate and fico_score are not statistically significant 
summary(logistic.model) 


# Fit another Logistic Regression Model using only statistically significant predictor variables 
logistic.model.sig <- glm(charge_off ~ annual_inc + term + grade + inq_last_6mths, family = "binomial", data = training_set) 


# Predicting Probability of a Loan Charge-Off
# Make predictions for the test set elements using the created logistic regression model
predictions.logistic <- predict(logistic.model, newdata = test_set, type = "response") 

# Make predictions for the test set elements using the logistic regression model only with significant variables 
predictions.logistic.sig <- predict(logistic.model.sig, newdata = test_set, type = "response") 

# Take a look at the range of the probability predictions 
range(predictions.logistic)
range(predictions.logistic.sig)

# The range of predictions for both models is wide which is a good indicator; a small range means that the test set 
# cases do not lie far apart, therefore the model might not be good in discriminating good and bad loans 



# Evaluating the Result of Logistic Regression Model
# To compare our predictions with the binary test_set$charge_off column, we must 
# transform the prediction vector to binary values of 1 and 0 indicating the status of the loan. 
# A cut-off or threshold must be set in this case. 
# If the predicted probability lies above the cutoff value then the prediction is set to 1, indicating
# a loan that charged off, otherwise it is set to 0, indicating the loan is still active;  
# A confusion matrix can be created afterwards to calculate Accuracy and compare cut-offs 
# The cut-off is basically a measure of risk tolerance of the financial institution 
# If we set a lower cutoff value, it means that we will classify a loan as a charge-off if it exceeds 
# a certain level of probabilistic risk. 

# USING A CUT-OFF VALUE OF 25% 
# Make a binary predictions-vector using a cut-off of 25%
pred_cutoff_25 <- ifelse(predictions > 0.25, 1, 0)
# Construct a confusion matrix using a cut-off of 25% 
conf_matrix_25 <- table(test_set$charge_off, pred_cutoff_25)
# Calculate for Accuracy 
accuracy.25 <- sum(diag(conf_matrix_25)) / nrow(test_set)
# The accuracy for the model is 86.32% 

# USING A CUT-OFF VALUE OF 50% 
# Make a binary predictions vector using a cut-off of 50% 
pred_cutoff_50 <- ifelse(predictions > 0.50, 1, 0) 
# Construct a confusion matrix using a cut-off of 50% 
conf_matrix_50 <- table(test_set$charge_off, pred_cutoff_50) 
# Calculate for Accuracy 
accuracy.50 <- sum(diag(conf_matrix_50)) / nrow(test_set) 
# The accuracy for the model is 89.02% 

# COMPARING THE TWO CUT-OFFS 
# Moving from a cut-off of 25% to 50% increases overall Accuracy of the model 



# Decision Tree
# Load the rpart package for the construction of the decision tree 
library(rpart)

# Construct a Decision Tree 
tree.model <- rpart(charge_off ~ home_ownership + annual_inc + loan_amnt + term + int_rate +
                      grade + fico_score + inq_last_6mths, method = "class",
                    data = training_set, control = rpart.control(cp = 0.001))

# Construct a Decision Tree with changed prior probabilities 
# The original data set is imbalanced in the sense that the cases of non charge-off loans outnumber
# loans that are charged-off. To fix this, we can change the prior probabilities in the rpart function 
# By default, the prior probabilities of charge-off and non charge-off are set equal to their proportions 
# in the training set. By making the prior probabilities of charge-off larger, we place a greater importance 
# on charge-offs, leading to a better decision tree 

tree.model.modified <- rpart(charge_off ~ home_ownership + annual_inc + loan_amnt + term + int_rate +
                      grade + fico_score + inq_last_6mths, method = "class",
                    data = training_set, parms = list(prior = c(0.60, 0.40)),
                    control = rpart.control(cp = 0.001))

# Plot the decision trees 
plot(tree.model, uniform = TRUE)
plot(tree.model.modified, uniform = TRUE) 

# Add labels to the decision trees 
text(tree.model) 
text(tree.model.modified)


# Pruning the Decision Tree
# Pruning a large tree is necessary to prevent overfitting, which can lead to inaccurate predictions 

# Use plotcp() to visualize cross-vaidated error (X-val Relative Error) in relation 
# to the complexity parameter for the tree.model 
plotcp(tree.model.modified)

# Use printcp() to print a table of information about CP, splits, and errors. The goal is to identify 
# which split has the minimum cross-validated error in tree.model 
printcp(tree.model.modified) 

# Create an index for the row with the minimum xerror
index <- which.min(tree.model.modified$cptable[, "xerror"])

# Create tree_min
tree_min <- tree.model.modified$cptable[index, "CP"]

#  Prune the tree using tree_min
pruned.tree <- prune(tree.model.modified, cp = tree_min)

# Use prp() to plot the pruned tree
prp(pruned.tree)


# Evaluating the Decision Tree
# Make predictions for the probability of default using the 2 decision trees created 
predictions.tree <- predict(tree.model, newdata = test_set)[ ,2]
predictions.ptree <- predict(pruned.tree, newdata = test_set)[, 2]

# Make binary predictions for the pruned decision tree and original tree using the test set 
predictions.binary.tree <- predict(tree.model, newdata = test_set, type = "class")
predictions.binary.ptree <- predict(pruned.tree, newdata = test_set, type = "class")

# Construct confusion matrices using the predictions.
confmatrix_tree <- table(test_set$charge_off, predictions.tree)

# Calculate for the model accuracy 
accuracy.tree <- sum(diag(confmatrix_tree)) / nrow(test_set)
# The accuracy of the decision tree model is 72% 


# The ROC Curve
# ROC Curves for the comparison of the 2 Logistic Regression Models 
ROC_logistic <- roc(test_set$charge_off, predictions.logistic)
ROC_logistic.significant <- roc(test_set$charge_off, predictions.logistic.sig)

# Use the previously created objects to construct ROC-curves all in one plot 
plot(ROC_logistic, col = "green") 
lines(ROC_logistic.significant, col = "red")

# Compute for the Area Under the Curve (AUC) 
# The logistic regression model with only significant variables has a higher AUC and should be preferred 
auc(ROC_logistic) # 0.6690
auc(ROC_logistic.significant) # 0.6709 

# ROC Curves for the comparison of the 2 Decision Tree Models 
ROC_tree <- roc(test_set$charge_off, predictions.tree)
ROC_tree.pruned <- roc(test_set$charge_off, predictions.ptree)

# Use the previously created objects to construct ROC-curves all in one plot 
plot(ROC_tree, col = "black") 
lines(ROC_tree.pruned, col = "blue")

# Compute for the Area Under the Curve (AUC) 
# The pruned decision tree model has a higher AUC and should be preferred 
auc(ROC_tree) # 0.6222 
auc(ROC_tree.pruned) # 0.6539 

# Compare the ROC of the logistic regression and decision tree models 
plot(ROC_logistic.significant, col = "red")
lines(ROC_tree.pruned, col = "blue")

# COMPARE THE best logistic vs. the best decision tree 
auc(ROC_logistic.significant) # 0.6709 
auc(ROC_tree.pruned) # 0.6539 
