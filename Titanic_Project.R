#Loading the Titanic dataset

train <- read.csv(file.choose(), stringsAsFactors = F, na.strings = c("","NA",""))
test <- read.csv(file.choose(), stringsAsFactors = F, na.strings = c("","NA",""))

str(train)
str(test)

head(train)
head(test)

##################################################
# Lazy Predictor
##################################################

#Set survived column to 0
test$Survived <- 0
my_solution <- data.frame(PassengerID = test$PassengerId, Survived = test$Survived)

nrow(my_solution)

#Write to a csv file
write.csv(my_solution, "LazyPredictor.csv", row.names = FALSE)


###################################################
# Data Preparation
###################################################

# Combine train and test data for Data Cleaning and Preparation
Full <- rbind(train,test)

# Structure of the Full data
str(Full)
summary(Full)


# Survival rates in absolute numbers
table(Full$Survived)

# Survival rates in proportions
prop.table(table(Full$Survived))


#Data Type conversion
Full$Pclass = as.factor(Full$Pclass)

#Look at missing values
library(Amelia)
missmap(Full, main = "Missing Map")

# Imputing Missing Value

# Missing Value Imputation - Age
Full$Age[is.na(Full$Age)] <- mean(Full$Age,na.rm=T)
sum(is.na(Full$Age))

# Missing Value Imputation - Embarked
table(Full$Embarked, useNA = "always")

# Substitute the missing values with the mode value
Full$Embarked[is.na(Full$Embarked)] <- 'S'
sum(is.na(Full$Embarked))
table(Full$Embarked, useNA = "always")


# Missing Value Imputation - Fare
# Substitute the missing values with the average value
Full$Fare[is.na(Full$Fare)] <- mean(Full$Fare,na.rm=T)
sum(is.na(Full$Fare))


# Missing Value Imputation - Cabin
#Drop the variable as the missing value is more than 20%
Full <- Full[-11]


#Check again for NA
sapply(Full, function(df)
{
  sum(is.na(df)==T)/length(df)
})

#Splitting data set back into train and test

train_cleaned <- Full[1:891,]
test_cleaned <- Full[892:1309,]

#######################################################
# Data Exploration
#######################################################

##univariate EDA
library(ggplot2)

#categorical variables
xtabs(~Survived,train_cleaned)
ggplot(train_cleaned) + geom_bar(aes(x=Survived))
ggplot(train_cleaned) + geom_bar(aes(x=Sex))
ggplot(train_cleaned) + geom_bar(aes(x=Pclass))

#numerical variables
ggplot(train_cleaned) + geom_histogram(aes(x=Fare),fill = "white", colour = "black")
ggplot(train_cleaned) + geom_boxplot(aes(x=factor(0),y=Fare)) + coord_flip()
ggplot(train_cleaned) + geom_histogram(aes(x=Age),fill = "white", colour = "black")
ggplot(train_cleaned) + geom_boxplot(aes(x=factor(0),y=Age)) + coord_flip()

#####################################################################################
##bivariate EDA
#Categorical-Categorical relationships
xtabs(~Survived+Sex,train_cleaned)
ggplot(train_cleaned) + geom_bar(aes(x=Sex, fill=factor(Survived)))

xtabs(~Survived+Pclass,train_cleaned)
ggplot(train_cleaned) + geom_bar(aes(x=Pclass, fill=factor(Survived)) )

xtabs(~Survived+Embarked,train_cleaned)
ggplot(train_cleaned) + geom_bar(aes(x=Embarked, fill=factor(Survived)) )

#Numerical-Categorical relationships
ggplot(train_cleaned) + geom_boxplot(aes(x = factor(Survived), y = Age))
ggplot(train_cleaned) + geom_histogram(aes(x = Age),fill = "white", colour = "black") + facet_grid(factor(Survived) ~ .)

ggplot(train_cleaned) + geom_boxplot(aes(x = factor(Survived), y = Fare))
ggplot(train_cleaned) + geom_histogram(aes(x = Fare),fill = "white", colour = "black") + facet_grid(factor(Survived) ~ .)

#####################################################################################
##multivariate EDA
xtabs(~factor(Survived)+Pclass+Sex,train_cleaned)
ggplot(train_cleaned) + geom_bar(aes(x=Sex, fill=factor(Survived))) + facet_grid(Pclass ~ .)


xtabs(~Survived+Embarked+Sex,train_cleaned)
ggplot(train_cleaned) + geom_bar(aes(x=Sex, fill=factor(Survived))) + facet_grid(Embarked ~ .)
#####################################################################################

###############################################
# Feature Engineering
###############################################

# Engineered variable 1: Child
Full$Child <- NA
Full$Child[Full$Age < 18] <- 1
Full$Child[Full$Age >= 18] <- 0
str(Full$Child)
ggplot(Full) + geom_bar(aes(x=Child))


# Engineered variable 2: Title
Full$Title <- sapply(Full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
Full$Title <- sub(' ', '', Full$Title)  # Remove the white space or blank
table(Full$Title)
ggplot(Full) + geom_bar(aes(x=Title))


# Combine small title groups
Full$Title[Full$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
Full$Title[Full$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
Full$Title[Full$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
# Convert to a factor
Full$Title <- factor(Full$Title)
table(Full$Title)
ggplot(Full) + geom_bar(aes(x=Title))

# Engineered variable 3: Family size
Full$FamilySize <- Full$SibSp + Full$Parch + 1
table(Full$FamilySize)
ggplot(Full) + geom_bar(aes(x=FamilySize))


# Split back into test and train sets
train_Featured <- Full[1:891,]
test_Featured <- Full[892:1309,]

train_Featured$Survived <- as.factor(train_Featured$Survived)
train_Featured$Sex <- as.factor(train_Featured$Sex)
train_Featured$Embarked <- as.factor(train_Featured$Embarked)

test_Featured$Sex <- as.factor(test_Featured$Sex)
test_Featured$Embarked <- as.factor(test_Featured$Embarked)

##################################################################
# Model Building : Logistic Regression
##################################################################

#Split the data set
library(caTools)
set.seed(123)

split <- sample.split(train_Featured, SplitRatio = 0.8)

train.data <- subset(train_Featured, split == TRUE)
test.data <- subset(train_Featured, split == FALSE)

#Train the model
#Model 1: Without feature Engineering
# After removing Passenger ID, Name and Ticket and Engineered Variable

logit_model1 <- glm(Survived ~ ., family = binomial (link = "logit"), 
                    train.data[-c(1,4,9,12,13,14)])
summary(logit_model1) #AIC : 617.02

#Model 2 : With feature engineering
#Remove only Passenger ID, Name and Ticket
logit_model2 <- glm(Survived ~ . , family = binomial (link = "logit"),
                    train.data[-c(1,4,9,13)])
summary(logit_model2)   #AIC: 614.05                 

# Check if the two models are significantly different

anova(logit_model1, logit_model2, test = "Chisq") # It is different

anova(logit_model2, test = "Chisq")

# Predict using test.data
fitted.results <- predict(logit_model2, test.data, type = "response")

#Convert probabilities to factor
fitted.results <- ifelse(fitted.results >0.5, 1, 0)

#Model Evaluation
library(caret)
confusionMatrix(table(test.data$Survived, fitted.results)) # Accuracy = 0.7853

# Make predictions on the test set
my_predictions <- predict(logit_model2, test_Featured, type = "response")

#Convert Probabilities
my_predictions <- ifelse(my_predictions >0.5, 1, 0)

#Write it to a csv file
my_solution <- data.frame(PassengerID = test_Featured$PassengerId, 
                          Survived = my_predictions)

write.csv(my_solution, "Logistic Regression Prediction.csv", row.names = FALSE)

###############################################################################

# Build Model and Predict for full train_Featured dataset

logit_model3 <- glm(Survived ~ . , family = binomial (link = "logit"),
                    train_Featured[-c(1,4,9)])

Y_pred <- predict(logit_model3, test_Featured, type = "response")
Y_pred <- ifelse(Y_pred > 0.5, 1, 0)

my_solution <- data.frame(PassengerID = test_Featured$PassengerId, 
                          Survived = Y_pred)

write.csv(my_solution, "Logistic Regression Prediction 2.csv", row.names = FALSE)

###############################################################################
# Model Building : Decision Tree
###############################################################################

#Model 1 : Without feature engineering
library(rpart)
library(rpart.plot)
dtree <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
               train_Featured, method = "class") 

rpart.plot(dtree)

Y_pred <- predict(dtree, test_Featured, type = "class")

my_solution <- data.frame(PassengerID = test_Featured$PassengerId, 
                          Survived = Y_pred)

write.csv(my_solution, "Decision Tree 1.csv", row.names = FALSE)

#Model 2 : With feature engineering
dtree2 <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked
                + Child + FamilySize + Title,
                train_Featured, method = "class") 

rpart.plot(dtree2)

Y_pred <- predict(dtree2, test_Featured, type = "class")

my_solution <- data.frame(PassengerID = test_Featured$PassengerId, 
                          Survived = Y_pred)

write.csv(my_solution, "Decision Tree 2.csv", row.names = FALSE)

#Pruning the tree
printcp(dtree2)

prune_tree <- prune(dtree2, cp = 0.01)

Y_pred <- predict(prune_tree, test_Featured, type = "class")

my_solution <- data.frame(PassengerID = test_Featured$PassengerId, 
                          Survived = Y_pred)

write.csv(my_solution, "Decision Tree 3.csv", row.names = FALSE)

##############################################################################
# Model Building : Random Forest
##############################################################################
set.seed(123)
install.packages("randomForest")
library("randomForest")
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Child + Title + FamilySize,
                    data=train_Featured, importance=TRUE, ntree=2000)
# Look at variable importance
varImpPlot(fit)

# Prediction
my_Prediction <- predict(fit, test_Featured)
my_solution <- data.frame(PassengerId = test_Featured$PassengerId, Survived = my_Prediction)
write.csv(my_solution, file = "Featuredfirstforest.csv", row.names = FALSE)
