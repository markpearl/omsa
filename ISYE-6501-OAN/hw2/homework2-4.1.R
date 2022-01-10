library(kknn)
library(caret)
library(tidyverse)

credit_data <- read.table("C:/Users/mjpearl/Downloads/data 2.2/credit_card_data.txt", stringsAsFactors = FALSE, header = FALSE)

#Number of rows in credit card table data
rows = nrow(credit_data) 

#Randomly selecting 1 5th of the rows indexes among 654 indexes
sample = sample(1:rows, size = round(rows/5), replace = FALSE)

#training dataset selected by excluding the 1/5th of the sample
training = credit_data[-sample,]
#testing will cover the remaining portion and include the 1/5th previously exlcuded in training
testing = credit_data[sample,]  

#training of kknn method via leave-one-out crossvalidation, we want to find the optimal value of 'k'. 
#The training function will determine the optimal hyperparameter values required for the best model performance
kknn_fit=train.kknn(formula = V11 ~ .,
                data = training, 
                kmax = 100, 
                kernel = c("optimal","rectangular", "inv", "gaussian", "triangular"),
                scale = TRUE) 
kknn_fit

#Call:
#  train.kknn(formula = V11 ~ ., data = d.train, kmax = 100, kernel = c("optimal",     "rectangular", "inv", "gaussian", "triangular"), scale = TRUE)#
#
#Type of response variable: continuous
#minimal mean absolute error: 0.2006881
#Minimal mean squared error: 0.1062738
#Best kernel: inv
#Best k: 31

#Testing the model on test data
pred<-predict(kknn_fit, testing)
#This to ensure we're receiving a binary response for classifcation instead of a rounded result
pred_bin<-round(pred)
pred_accuracy<-table(pred_bin,testing$V11)
pred_accuracy

#KKNN accuracy for the prediction 
sum(pred_bin==testing$V11)/length(testing$V11)

#[1] 0.8549618

#The most effective train/test split was 1/5. Multiple combinations were tried including 1/3, 1/4 and 1/2.