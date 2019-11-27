library(reticulate)
library(keras)
library(dplyr)
library(mlbench)
library(neuralnet)
library(magrittr)
library(randomForest)
library(rfUtilities)
#####Estabile your own dataframe, the first column is your target variable
#the following columns are your predictor variables. Do not include any other
#columns in the dataframe
setwd("C:/Columbia")
df<-read.table("combined_dengue.csv", header=TRUE, sep=",")
drop<-c("X","ID","city")
data_df<-df[,!(names(df) %in% drop)]
used_df<-data_df[complete.cases(data_df),]
used_df[used_df<0]=0
ind<-sample(2,nrow(used_df),replace=T,prob=c(0.8,0.2))
#Random Forests################
training_RF<-used_df[ind==1,1:19]
test_RF<-used_df[ind==2,1:19]
rf<-randomForest(number_dengue~.,data=training_RF,importance=T)
pred<- predict(rf, test_RF)
RF_MAE<-mean(abs(test_RF$number_dengue-pred))
RF_RMSE<-sqrt(mean((test_RF$number_dengue-pred)^2))
RF_R2 <- 1 - (sum((test_RF$number_dengue-pred)^2)/sum((test_RF$number_dengue-mean(test_RF$number_dengue))^2))
#deep nerual network###############
my_data<-as.matrix(used_df)
dimnames(my_data)<-NULL
training<-my_data[ind==1,2:19]
test<-my_data[ind==2,2:19]
trainingtarget<-my_data[ind==1,1]
testtarget<-my_data[ind==2,1]
m <- colMeans(training)
s <- apply(training, 2, sd)
training<-scale(training, center=m, scale=s)
test<-scale(test,center=m,scale=s)
#carefully adjust hyperparameters in the DNN especially the number of
#hidden layers (based on my experience 3 hidden layers), the number of neurons,
#and dropout.
model<-keras_model_sequential()
model %>%
  layer_dense(units = 150, activation='relu',input_shape = c(18)) %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units = 120, activation='relu') %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units = 50, activation='relu') %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units = 30, activation='relu') %>%
  layer_dropout(rate=0.5) %>%
  layer_dense(units=1)
model %>% compile(loss='mse',
                  optimizer=optimizer_adam(lr=0.001),
                  metrics='mae')
mymodel<-model %>% fit(training,trainingtarget,
                       epochs = 100,
                       batch_size = 32,
                       validation_split = 0.2)
model %>% evaluate(test,testtarget)
pred<- model %>% predict(test)
DNN_MAE<-mean(abs(testtarget-pred))
DNN_RMSE<-sqrt(mean((testtarget-pred)^2))
DNN_R2 <- 1 - (sum((testtarget-pred)^2)/sum((testtarget-mean(testtarget))^2))


