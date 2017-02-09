setwd("D:/Personal/Coursea/8. Machine Learning/Assignment1")

library(caret);library(randomForest); library(e1071)
library(rpart);library(rattle);library(rpart.plot)

TrainingDataUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
TestingDataUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(TrainingDataUrl, "training.csv")
download.file(TestingDataUrl, "testing.csv")

#Loading datasets
training <- read.csv("training.csv", header = TRUE, 
                     na.strings = c("",NA,"#DIV/0!"), 
                     stringsAsFactors = FALSE)
testing <- read.csv("testing.csv", header = TRUE,
                    na.strings = c("",NA,"#DIV/0!"),
                    stringsAsFactors = FALSE)

#Split training data into a validation set, now that the data is processed
inTrain <- createDataPartition(y = training$classe, p = 0.7, list = FALSE)
training <- training[inTrain,]
validation <- training[-inTrain,]


set.seed(197811)

#Exploratory Data Analysis (on training set)
dim(training)
str(training)
summary(training)
View(training)


#Convert classe to a factor variable
training$classe <- as.factor(training$classe)

##Selecting the Predictors to use in the model
#When viewed the data, it looks like there were summation lines, (when new window = yes)
#Eliminate those
training <- training[!training$new_window == "yes",]

#Remove unneeded variables (X, time stamps, name variables)
training <- subset(training, select = -c(X, user_name, raw_timestamp_part_1,
                                         raw_timestamp_part_2, cvtd_timestamp, new_window))


#Find Remove Zero Covariates (make sure you don't remove outcome variable (column 154))
nsv <- nearZeroVar(training[,-154], saveMetrics = TRUE)
nsv
nsv <- nearZeroVar(training)
training <- training[,-nsv]

#53 variables is still quite alot, and makes it hard to explain the model.  
#Going to use random forest to figure out which variable are most important
fit.forest <- randomForest(classe ~., data = training, importance = TRUE, ntree = 250)
varImpPlot(fit.forest, type = 1)
varImp <- importance(fit.forest)

#Looks like we can include the top 25 variables and still maintain model accuracy, so.
#remove the extra variables
selVars <- names(sort(varImp[,1], decreasing = TRUE))[1:25 & 54]

#Fit a random forest model
#In random forests, there is no need for cross-validation or a separate test set 
#to get an unbiased estimate of the test set error. It is estimated internally , during the run...
modRF <- randomForest(x = training[,selVars], y = training$classe,
                      ntree = 100,
                      nodesize = 7,
                      importance = TRUE)

#Do a prediction on the training data
predictTRF <- predict(modRF, newdata = training[,selVars])

#Check to see how well it works, not surprisingly, it did well on the training data
confusionMatrix(predictTRF, training$classe)

#Lets validate on the validation data set
#First, we need to process the validation the same way as the training set
validation$classe <- as.factor(training$classe)
validation <- validation[!validation$new_window == "yes",]
validation <- subset(validation, select = -c(X, user_name, raw_timestamp_part_1,
                                         raw_timestamp_part_2, cvtd_timestamp, new_window))

#Predict using validation data
predictVRF <- predict(modRF, newdata = validation[,selVars])
confusionMatrix(predictVRF, validation$classe)

#Still got a 99.99% accuracy rating

#Process test data same way as training data
testing$classe <- as.factor(testing$classe)
testing <- testing[!testing$new_window == "yes",]
testing <- subset(testing, select = -c(X, user_name, raw_timestamp_part_1,
                                             raw_timestamp_part_2, cvtd_timestamp, new_window))

#Predict using testing data
predictTRF <- predict(modRF, newdata = testing[,selVars])
table(predictTRF)