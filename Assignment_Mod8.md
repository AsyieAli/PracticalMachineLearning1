--
title: "Assignment Module 8"
author: "Asyie Ien"
date: "January 29, 2016"
output: html_document
---

## GitHub Link

The repository of this assignment in Github located here:
https://github.com/AsyieAli/PracticalMachineLearning1.git



## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 
In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 
They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data Prep  
```{r, cache = T}
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(corrplot)
```

### Data Download
```{r, cache = T}
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./data/pml-training.csv"
testFile  <- "./data/pml-testing.csv"
if (!file.exists("./data")) {
  dir.create("./data")
}
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile=trainFile, method="curl")
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile=testFile, method="curl")
}
```  
### Data Read
Read the two csv files into two data frames after download.  
```{r, cache = T}
trainData <- read.csv("./data/pml-training.csv")
testData <- read.csv("./data/pml-testing.csv")
dim(trainData)
dim(testData)
```
The testing data set contains 20 observations and 160 variables.
The training data set contains 19622 observations and 160 variables.
The "classe" field in the training set is the outcome to predict. 

### Data Clean
The data is cleaned and remove missing values and other meaningless variables.
```{r, cache = T}
sum(complete.cases(trainData))
```
To remove columns that contain NA missing values.
```{r, cache = T}
trainData <- trainData[, colSums(is.na(trainData)) == 0] 
testData <- testData[, colSums(is.na(testData)) == 0] 
```  
To remove columns with not much contribution to the accelerometer measurements.
```{r, cache = T}
classe <- trainData$classe
trainGetRid <- grepl("^X|timestamp|window", names(trainData))
trainData <- trainData[, !trainGetRid]
trainCleaned <- trainData[, sapply(trainData, is.numeric)]
trainCleaned$classe <- classe
testGetRid <- grepl("^X|timestamp|window", names(testData))
testData <- testData[, !testGetRid]
testCleaned <- testData[, sapply(testData, is.numeric)]
```
The testing data set contains 20 observations and 53 variables.
The training data set contains 19622 observations and 53 variables.
The "classe" variable/field is still in the cleaned training set.

### Data Dicing
Split the cleaned training set into a training data set (70%) and a validation data set (30%). 
The validation data set to conduct cross validation.  
```{r, cache = T}
set.seed(22519) 
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData1 <- trainCleaned[inTrain, ]
testData1 <- trainCleaned[-inTrain, ]
```

## Data Modeling

Use Random Forest algorithm for activity recognition.Use 5-fold cross validation when applying the algorithm.  
```{r, cache = T}
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData1, method="rf", trControl=controlRf, ntree=250)
modelRf
```
Estimate the performance of the model on the validation data set.  
```{r, cache = T}
predictRf <- predict(modelRf, testData1)
confusionMatrix(testData1$classe, predictRf)
```
```{r, cache = T}
accuracy <- postResample(predictRf, testData1$classe)
accuracy
oose <- 1 - as.numeric(confusionMatrix(testData1$classe, predictRf)$overall[1])
oose
```
The estimated accuracy is 99.42% and the estimated out-of-sample error is 0.58%.

## Test Data Set Prediction
Apply the model to the original testing data. Remove the `problem_id` column first.  
```{r, cache = T}
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result
```  

## Output: Plots

1. Decision Tree Visualization
```{r, cache = T}
DecTreeVis <- rpart(classe ~ ., data=trainData1, method="class")
prp(DecTreeVis) 
```

2. Correlation Matrix Visualization  
```{r, cache = T}
CorMatVis <- cor(trainData1[, -length(names(trainData1))])
corrplot(CorMatVis, method="color")

```