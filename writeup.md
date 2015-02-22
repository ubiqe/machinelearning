# Machine Learning Course Project
ubiqe  
Sunday, February 22, 2015  
##Data loading and cleaning
A quick glance at the dataset shows there are a lot of missing values, some of which are coded NA, some are simply missing, some are divisions by 0 (in excel! ;) ). To tidy up we can code them all in the same way while reading in the data (the dataset should be in the working directory):

```r
library(plyr)
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
train<-read.csv("pml-training.csv", stringsAsFactors=FALSE, na.string=c("", "NA", "#DIV/0!") )
```

The first 7 columns are observation ID, subject name and timestamps - they are useless for prediction (that is, if you predict on them the interpretation would be meaningless). The ratio of between and within subject variance should be taken care of (i.e. by specifying the error SS properly), but this is out of scope here. The last variable is the response, we can change it into factor.

```r
trainA<-train[,-c(1,2,3,4,5,6,7)]
trainA<-mutate(trainA, classe=factor(classe))
```
Most of the columns consist of NAs - they cannont be uses for prediction. We can select the non-NA columns by counting the sum of NA values.


```r
goodCols<-colSums(is.na(trainA))==0
trainA<-trainA[,goodCols]
```

Now we split the data into training and test partitions - for cross validation and computing estimates of out-of-sample error.


```r
inTrain<-createDataPartition(y=trainA$classe, p=0.75, list=FALSE)
trainB<-trainA[inTrain,]
testCV<-trainA[-inTrain,]
```

We can verify if all the selected variables are useful for prediction by checking if any of them has near zero variance.


```r
nzv<-nearZeroVar(trainB)
length(nzv)
```

```
## [1] 0
```

To reduce intercorrelation between predictors (and speed up training process) we can reduce the number of predictors via PCA. Then we predict the values for each observation and each principal component for both training and test partitions.


```r
pca<-preProcess(trainB[,-53], method="pca", na.remove=TRUE)
pca$numComp
```

```
## [1] 26
```

```r
trainC<-predict(pca, trainB[,-53])
trainC$classe<-trainB$classe
testPCA<-predict(pca, testCV[,-53])
```

Now comes the action - we build a model using random forests algorithm. The in-sample accuracy is very high.

```r
model<-randomForest(classe~., data=trainC)
model
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = trainC) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 5
## 
##         OOB estimate of  error rate: 2.21%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4166    8    7    3    1     0.00454
## B   43 2762   35    4    4     0.03020
## C    5   37 2502   19    4     0.02532
## D    3    1  103 2301    4     0.04602
## E    1   11   21   12 2661     0.01663
```

Finally, we validate the model against the testing partition. The out-of-sample accuracy can be estimated at 98%


```r
predRF<-predict(model, newdata=testPCA)
confusionMatrix(testCV$classe, predRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1387    2    5    0    1
##          B   10  927    9    2    1
##          C    1   13  835    6    0
##          D    0    0   31  770    3
##          E    0    6   11    5  879
## 
## Overall Statistics
##                                         
##                Accuracy : 0.978         
##                  95% CI : (0.974, 0.982)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.973         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.992    0.978    0.937    0.983    0.994
## Specificity             0.998    0.994    0.995    0.992    0.995
## Pos Pred Value          0.994    0.977    0.977    0.958    0.976
## Neg Pred Value          0.997    0.995    0.986    0.997    0.999
## Prevalence              0.285    0.193    0.182    0.160    0.180
## Detection Rate          0.283    0.189    0.170    0.157    0.179
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.995    0.986    0.966    0.988    0.994
```

The last part is the actual test with out-of-sample observations.


```r
testing<-read.csv("pml-testing.csv")
testingA<-testing[,-c(1,2,3,4,5,6,7)]
testingA<-testingA[,goodCols]
testingPCA<-predict(pca, testingA[,-53])
predTest<-predict(model, newdata=testingPCA)
answers<-as.character(predTest)
```
Well, here the accuracy was 100% so beyond expectations ;)
