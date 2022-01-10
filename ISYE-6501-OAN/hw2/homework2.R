library(kknn)
library(caret)
library(tidyverse)

credit_data <- read.table("C:/Users/mjpearl/Downloads/data 2.2/credit_card_data.txt", stringsAsFactors = FALSE, header = FALSE)

head(credit_data)

#Utilize cross validation to 

# Split the data into training and test set
set.seed(123)

#Initialize the training dataset with a sample of an 80% partition using random split
sample <- sample.int(n = nrow(credit_data), size = floor(.75*nrow(credit_data)), replace = F)
train <- credit_data[sample, ]
test  <- credit_data[-sample, ]

#Build the knn model using cross validation
model <- lm(Fertility ~., data = train.data)
# Make predictions and compute the R2, RMSE and MAE
predictions <- model %>% predict(test.data)


data.frame( R2 = R2(predictions, test.data$Fertility),
            RMSE = RMSE(predictions, test.data$Fertility),
            MAE = MAE(predictions, test.data$Fertility))

