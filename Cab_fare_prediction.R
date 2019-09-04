#Remove all existing environment
rm(list=ls(all=T))
#set the working directory
setwd("C:/Data science/Project/Cab_fare_prediction")
#check the working directory
getwd()
#load libraries
X=c("ggplot2","corrgram","DMwR","caret","randomForest","unbalanced","C50","dummies","MASS","rpart",
    "gbm","ROSE","e1071","Information","sampling","DataCombine","inTrees","readxl")
#Installing packages
install.packages(c("randomForest","unbalanced","c50","dummies","MASS","rpart","gbm","ROSE","e1071","Information","DataCombine"))
lapply(X,require,character.only=TRUE)
rm(X)

# loading Cab fare data set
cab_train=read.csv("train_cab.csv",header = TRUE,na.strings = c(" ", "", "NA"))
str(cab_train)
# Loading test data
cab_test=read.csv("test.csv",header = TRUE)
str(cab_test)

#####Explorataory Data Analysis##########

str(cab_train)
str(cab_test)
head(cab_train)
head(cab_test)
summary(cab_train)
summary(cab_test)

#Necessary Type conversions
cab_train$fare_amount=as.numeric(as.character(cab_train$fare_amount))

cab_train$pickup_datetime=as.POSIXct(cab_train$pickup_datetime,format="%Y-%m-%d %H:%M:%S")

#Necessary Type conversions in test data
cab_test$pickup_datetime=as.POSIXct(cab_test$pickup_datetime,format="%Y-%m-%d %H:%M:%S")
cab_test$passenger_count=as.numeric(cab_test$passenger_count)
### Removing values which are not within desired range(outlier) depending upon basic understanding of dataset##

##Fare amount has a negative value, which doesn't make sense. A price amount cannot be -ve and also cannot be 0.
#So we will remove these fields.
cab_train[which(cab_train$fare_amount < 1 ),]
nrow(cab_train[which(cab_train$fare_amount < 1 ),])
cab_train = cab_train[-which(cab_train$fare_amount < 1 ),]

##### Passenger_count variable #########
for (i in seq(4,11,by=1)){
  print(paste('passenger_count above ' ,i,nrow(cab_train[which(cab_train$passenger_count > i ),])))
}
# so 20 observations of passenger_count is consistenly above from 6,7,8,9,10 passenger_counts, let's check them.
cab_train[which(cab_train$passenger_count > 6 ),]
nrow(cab_train[which(cab_train$passenger_count > 6 ),])
# Also we need to see if there are any passenger_count==0
cab_train[which(cab_train$passenger_count <1 ),]
nrow(cab_train[which(cab_train$passenger_count <1 ),])
# We will remove these 58 observations and 20 observation which are above 6 value because a cab cannot hold these number of passengers.
cab_train =cab_train[-which(cab_train$passenger_count < 1 ),]
cab_train = cab_train[-which(cab_train$passenger_count > 6),]
### Let us look at Geographical Features
print(paste('pickup_longitude above 180=',nrow(cab_train[which(cab_train$pickup_longitude >180 ),])))
print(paste('pickup_longitude above -180=',nrow(cab_train[which(cab_train$pickup_longitude < -180 ),])))
print(paste('pickup_latitude above 90=',nrow(cab_train[which(cab_train$pickup_latitude > 90 ),])))
print(paste('pickup_latitude above -90=',nrow(cab_train[which(cab_train$pickup_latitude < -90 ),])))
print(paste('dropoff_longitude above 180=',nrow(cab_train[which(cab_train$dropoff_longitude > 180 ),])))
print(paste('dropoff_longitude above -180=',nrow(cab_train[which(cab_train$dropoff_longitude < -180 ),])))
print(paste('dropoff_latitude above -90=',nrow(cab_train[which(cab_train$dropoff_latitude < -90 ),])))
print(paste('dropoff_latitude above 90=',nrow(cab_train[which(cab_train$dropoff_latitude > 90 ),])))
# There's only one outlier which is in variable pickup_latitude.So we will remove it with nan.
# Also we will see if there are any values equal to 0.
nrow(cab_train[which(cab_train$pickup_longitude == 0 ),])
nrow(cab_train[which(cab_train$pickup_latitude == 0 ),])
nrow(cab_train[which(cab_train$dropoff_longitude == 0 ),])
nrow(cab_train[which(cab_train$pickup_latitude == 0 ),])
# there are values which are equal to 0. we will remove them.
df=cab_train
cab_train = cab_train[-which(cab_train$pickup_latitude > 90),]
cab_train = cab_train[-which(cab_train$pickup_longitude == 0),]
cab_train = cab_train[-which(cab_train$dropoff_longitude == 0),]

############ Missing Value Analysis #############
missing_val = data.frame(apply(cab_train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(cab_train)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
missing_val
# we decided to drop missing values from data set
cab_train=na.omit(cab_train)
summary(cab_train)
df1=cab_train
dfT=cab_test
#cab_train=df1
############## Outlier Analysis ###########
#Box plots-Distributaion and outlier check
Numeric_index=sapply(cab_train,is.numeric)#selecting only numeric
Numeric_data=cab_train[,Numeric_index]
cnames=colnames(Numeric_data)
cnames
for(i in 1:length(cnames)){
  assign(paste0("AB",i),ggplot(aes_string(y=(cnames[i]),x="fare_amount"),
                               d=subset(cab_train))
         +geom_boxplot(outlier.colour = "Red",outlier.shape = 18,outlier.size = 2,
                       fill="skyblue4")+theme_gray()
         +stat_boxplot(geom = "errorbar", width=0.5)
         +labs(y=cnames[i],x="fare_amount")
         +ggtitle("Box Plot of cab_fare",cnames[i]))
}
#ploting plots together
gridExtra::grid.arrange(AB1,AB2,ncol=2)
gridExtra::grid.arrange(AB3,AB4,ncol=2)
gridExtra::grid.arrange(AB5,AB6,ncol=2)

#remove Outliers using boxplotmethod
#Loop to remove from all variables

for(i in cnames){
  print(i)
  val=cab_train[,i][cab_train[,i]%in% boxplot.stats(cab_train[,i])$out]
  cab_train=cab_train[which(!cab_train[,i] %in% val),]
}
####### Feature Engineering ############
df3=cab_train
# 1.Feature Engineering for timestamp variable
# we will derive new features from pickup_datetime variable
# new features will be year,month,day_of_week,hour
#Convert pickup_datetime from factor to date time
cab_train$pickup_date = as.Date(as.character(cab_train$pickup_datetime))
cab_train$pickup_weekday = as.factor(format(cab_train$pickup_date,"%u"))# Monday = 1
cab_train$pickup_mnth = as.factor(format(cab_train$pickup_date,"%m"))
cab_train$pickup_yr = as.factor(format(cab_train$pickup_date,"%Y"))
pickup_time = strptime(cab_train$pickup_datetime,"%Y-%m-%d %H:%M:%S")
cab_train$pickup_hour = as.factor(format(pickup_time,"%H"))

#Add same features to test set
cab_test$pickup_date = as.Date(as.character(cab_test$pickup_datetime))
cab_test$pickup_weekday = as.factor(format(cab_test$pickup_date,"%u"))# Monday = 1
cab_test$pickup_mnth = as.factor(format(cab_test$pickup_date,"%m"))
cab_test$pickup_yr = as.factor(format(cab_test$pickup_date,"%Y"))
pickup_time = strptime(cab_test$pickup_datetime,"%Y-%m-%d %H:%M:%S")
cab_test$pickup_hour = as.factor(format(pickup_time,"%H"))
#remove unwanted variables 
cab_train = subset(cab_train,select = -c(pickup_datetime,pickup_date))
cab_test = subset(cab_test,select = -c(pickup_datetime,pickup_date))

##Calculate the distance travelled using longitude and latitude
deg_to_rad = function(deg){
  (deg * pi) / 180
}
haversine = function(long1,lat1,long2,lat2){
  long1rad = deg_to_rad(long1)
  phi1 = deg_to_rad(lat1)
  long2rad = deg_to_rad(long2)
  phi2 = deg_to_rad(lat2)
  delphi = deg_to_rad(lat2 - lat1)
  dellamda = deg_to_rad(long2 - long1)
  
  a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
    sin(dellamda/2) * sin(dellamda/2)
  
  c = 2 * atan2(sqrt(a),sqrt(1-a))
  R = 6371e3
  R * c / 1000 #1000 is used to convert to meters
}
# Using haversine formula to calculate distance fr both train and test
cab_train$dist = haversine(cab_train$pickup_longitude,cab_train$pickup_latitude,cab_train$dropoff_longitude,cab_train$dropoff_latitude)
cab_test$dist = haversine(cab_test$pickup_longitude,cab_test$pickup_latitude,cab_test$dropoff_longitude,cab_test$dropoff_latitude)
str(cab_train)
summary(cab_test)
# We will remove the variables which were used to feature engineer new variables
cab_train = subset(cab_train,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
cab_test = subset(cab_test,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))

str(cab_train)
summary(cab_train)
########  Feature Scaling   #########
#Normality check
qqnorm(cab_train$fare_amount)
histogram(cab_train$fare_amount)
library(car)
# dev.off()
par(mfrow=c(1,2))
qqPlot(cab_train$fare_amount)          # qqPlot, it has a x values derived from gaussian distribution, if data is distributed normally then the sorted data points should lie very close to the solid reference line 
truehist(cab_train$fare_amount)       # truehist() scales the counts to give an estimate of the probability density.
lines(density(cab_train$fare_amount))   # Right skewed      # lines() and density() functions to overlay a density plot on histogram

#Normalisation

print('dist')
cab_train[,'dist'] = (cab_train[,'dist'] - min(cab_train[,'dist']))/
  (max(cab_train[,'dist'] - min(cab_train[,'dist'])))
#check multicollearity
library(usdm)
vif(cab_train[,-1])
#################### Splitting train into train and validation subsets ###################
set.seed(1000)
tr.idx = createDataPartition(cab_train$fare_amount,p=0.75,list = FALSE) # 75% in trainin and 25% in Validation Datasets
train_data = cab_train[tr.idx,]
test_data = cab_train[-tr.idx,]

rmExcept(c("Cab_test","cab_train","df",'test_data','train_data'))
###################Model Selection################
#Error metric used to select model is RMSE,rsquare 

#############  Decision Tree  #####################

Dt_model = rpart(fare_amount ~ ., data = train_data, method = "anova")

summary(Dt_model)
#Predict for new test cases
predictions_DT = predict(Dt_model, test_data[-1])

qplot(x = test_data[,1], y = predictions_DT, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],predictions_DT)
#  mae       mse      rmse      mape 
# 1.6358343 5.0547785 2.2482834 0.2025124
#function for r-square-
rsquare=function(y,y1){
  cor(y,y1)^2
}
#r square calculation
rsquare(test_data[,1],predictions_DT)
# rsquare=0.6571
#Visulaization to check the model performance on test data-
plot(test_data$fare_amount,type="l",lty=1.8,col="Green",main="Decision Tree")
lines(predictions_DT,type="l",col="Blue")
summary(cab_train)
############Linear Regression ##########
lm_model = lm(fare_amount ~.,data=train_data)

summary(lm_model)
str(train_data)

lm_predictions = predict(lm_model,test_data[-1])

qplot(x = test_data[,1], y = lm_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],lm_predictions)
#  mae       mse      rmse      mape 
# 1.4534437 4.1357858 2.0336631 0.1815939

#r square calculation
rsquare(test_data[,1],lm_predictions)
#rsquare=0.7195
############# Random Forest ######
rf_model = randomForest(fare_amount ~.,data=train_data)

summary(rf_model)

rf_predictions = predict(rf_model,test_data[-1])

qplot(x = test_data[,1], y = rf_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],rf_predictions)
#  mae       mse      rmse      mape 
# 1.4305602 4.0125912 2.0031453 0.1846496 
#rsquare calculation
rsquare(test_data[,1],rf_predictions)
#rsquare value=0.7408
############  XGBOOST #########33
library("xgboost")
train_data_matrix = as.matrix(sapply(train_data[-1],as.numeric))
test_data_data_matrix = as.matrix(sapply(test_data[-1],as.numeric))

xgboost_model = xgboost(data = train_data_matrix,label = train_data$fare_amount,nrounds = 15,verbose = FALSE)

summary(xgboost_model)
xgb_predictions = predict(xgboost_model,test_data_data_matrix)


regr.eval(test_data[,1],xgb_predictions)
#  mae       mse      rmse      mape 
# 1.3649879 3.8021741 1.9499164 0.1705012
#rsquare calculation
rsquare(test_data[,1],xgb_predictions)

#rsquare value=0.7428

############# Finalizing and Saving Model for later use ####################
# In this step we will train our model on whole training Dataset and save that model for later use
train_data_matrix2 = as.matrix(sapply(cab_train[-1],as.numeric))
test_data_matrix2 = as.matrix(sapply(cab_test,as.numeric))

xgboost_model2 = xgboost(data = train_data_matrix2,label = cab_train$fare_amount,nrounds = 15,verbose = FALSE)

# Saving the trained model
saveRDS(xgboost_model2, "./final_Xgboost_model_using_R.rds")

# loading the saved model
super_model <- readRDS("./final_Xgboost_model_using_R.rds")
print(super_model)

# Lets now predict on test dataset
xgb = predict(super_model,test_data_matrix2)

xgb_pred = data.frame(test_data_matrix2,"predictions" = xgb)


# Now lets write(save) the predicted fare_amount in disk as .csv format 
write.csv(xgb_pred,"xgb_predictions_R.csv",row.names = FALSE)
