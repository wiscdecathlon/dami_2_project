
#DAMI2 Project Report Code

#2017 Nathan Brown

#This code imports a data set, preprocesses, and runs a data mining experiment 
#comparing methods for dealing with imbalanced data using Random Forests, AdaBooat
#Naive Bayes, and Logistic regression for a class prediction task of Admit or Discharge 
#from Emergency Department data. 

library(caret)
library(pROC)

#Loading the data
data <-read.csv("data_mining.csv",header = TRUE, sep = "|",quote = "")

    ## Summary of Data ## 
    # > nrow(data)
    #[1] 25967

    # > summary(data)
    # esi             presenting_problem                     complaint    
    # 3      :12234   Cardiorespiratory            :4129     abd pain :  266  
    # 4      : 7487   Gastrointestinal             :3830     CP       :  193  
    # 2      : 4655   Trauma                       :3705     SI       :  142  
    # 5      : 1106   General medicine             :3001     ETOH     :  136  
    # 0      :  329   Musculoskeletal non-traumatic:2626     back pain:  120  
    # 1      :  138   EENT and dental              :1785     SOB      :  118  
    # (Other):   18   (Other)                      :6891     (Other)  :24992 

    # age_yrs         gender            time            weekday    
    # Min.   : 0.00   F        :14786   18     : 1579   Monday   :3958  
    # 1st Qu.:25.00   M        :11146   12     : 1546   Tuesday  :3885  
    # Median :37.00            :   18   13     : 1493   Wednesday:3775  
    # Mean   :41.28   Monday   :    6   20     : 1472   Thursday :3716  
    # 3rd Qu.:57.00   Sunday   :    3   11     : 1451   Friday   :3593  
    # Max.   :90.00   Wednesday:    3   16     : 1432   Saturday :3509  
    # NA's   :18      (Other)  :    5   (Other):16994   (Other)  :3531  

    # provider_type                      door_to_provider_mins
    # Physician                 :15946   Min.   :-1227.00     
    # Advanced Practice Provider: 9386   1st Qu.:    5.00     
    # Unknown                   :  600   Median :    9.00     
    #                           :   18   Mean   :   16.63     
    # 113                       :    2   3rd Qu.:   19.00     
    # 20                        :    2   Max.   : 2892.00     
    # (Other)                   :   13   NA's   :18  

    # provider_to_decision_mins  los_mins           #### dispo ####     
    # 10     :  215             Min.   :-371384.0        :   18  
    # 9      :  201             1st Qu.:     81.0   0    :   15  
    # 11     :  200             Median :    138.0   1    :    2  
    # 14     :  199             Mean   :    114.9   Admit: 4061  
    # 16     :  199             3rd Qu.:    212.0   DC   :21456  
    # 13     :  194             Max.   : 302773.0   Other:  415  
    # (Other):24759             NA's   :35    

    # order_name                            event_count      stroke_alert    
    #                              :25477   Min.   :0.0000   Min.   :0.00000  
    # 0                            :   17   1st Qu.:0.0000   1st Qu.:0.00000  
    # Alert: Possible Sepsis       :  257   Median :0.0000   Median :0.00000  
    # Alert: Possible Septic Shock :   67   Mean   :0.2185   Mean   :0.01072  
    # Alert: Possible Severe Sepsis:  149   3rd Qu.:0.0000   3rd Qu.:0.00000  
    #                                       Max.   :5.0000   Max.   :1.00000  
    #                                       NA's   :18       NA's   :35 

    # intake     
    # Min.   :0.000  
    # 1st Qu.:0.000  
    # Median :0.000  
    # Mean   :0.191  
    # 3rd Qu.:0.000  
    # Max.   :1.000  
    # NA's   :35 

#########################
## BASIC PREPROCESSING ##
#########################

#To omit Errors and 0s From ESI feature
esi.idx <- data$esi %in% c(1,2,3,4,5)
data.cl<-data[esi.idx,]
nrow(data[!esi.idx,])

#To omit Genderless records
gender.idx<- data.cl$gender %in% c("M","F")
data.cl<-data.cl[gender.idx,]
nrow(data[!gender.idx,])

#To omit "Other" "Dispo
dispo.idx<- data.cl$dispo %in% c("Admit","DC")
data.cl<-data.cl[dispo.idx,]
nrow(data[!dispo.idx,])

#Drop Unused Factors after preprocessing
data.cl<- droplevels(data.cl)
data.cl$esi <- as.integer(data.cl$esi)

############################################################
## Creating Additional Complaint Categories (using regex) ##
############################################################

library(stringr)

target_string <- lapply(data.cl$complaint, as.character)

#EtOH related complaints
target_complaint<- paste(c("[Ee][Tt][Oo][Hh]","[Aa][Ll][Cc][Oo]","[Ww][Ii][Tt][Hh][Dd]","[Ii][Nn][Tt][Oo][Xx]","[Hh][Ee][Rr][Oo][Ii]","[Dd][Rr][Uu][Gg]"),collapse = '|')
string_index <-str_detect(target_string,target_complaint)
# table(data.cl$dispo[string_index],data.cl$esi[string_index])
string_index[FALSE]<-0
data.cl$intox<-string_index

#Headache relatd complaints
target_complaint<- paste(c("^[Hh][Ee][Aa][Dd][Aa]", "[Mm]igrai","[/ ]HA[ /]"),collapse = '|')
string_index <-str_detect(target_string,target_complaint)
# table(data.cl$dispo[string_index],data.cl$esi[string_index])
string_index[FALSE]<-0
data.cl$headache<-string_index

#Shortness of breath 
target_complaint<- paste(c("[Ss][Oo][Bb]","[Dd][Ii][Bb]","[Dd][Yy][Ss][Pp][Nn]","[Ss][Hh][Oo][Rr][Tt]"),collapse = '|')
string_index <-str_detect(target_string,target_complaint)
# table(data.cl$dispo[string_index],data.cl$esi[string_index])
string_index[FALSE]<-0
data.cl$sob<-string_index

#Eye related complaints
target_complaint<- paste(c("[Ee][Yy][Ee]"),collapse = '|')
string_index <-str_detect(target_string,target_complaint)
# table(data.cl$dispo[string_index],data.cl$esi[string_index])
string_index[FALSE]<-0
data.cl$eye<-string_index

#Seizure 
target_complaint<- paste(c("[Ss][Ee][Ii][Zz]"),collapse = '|')
string_index <-str_detect(target_string,target_complaint)
# table(data.cl$dispo[string_index],data.cl$esi[string_index])
string_index[FALSE]<-0
data.cl$seizure<-string_index

#SI/Suicidal 
target_complaint<- paste(c("^[Ss][Ii][^CcNn]","^SI$","[Ss][Uu][Ii][Cc][Ii]"),collapse = '|')
string_index <-str_detect(target_string,target_complaint)
# table(data.cl$dispo[string_index],data.cl$esi[string_index])
string_index[FALSE]<-0
data.cl$si<-string_index

#"Weakness"
target_complaint<- paste(c("[Ww][Ee][Aa][Kk][Nn]"),collapse = '|')
string_index <-str_detect(target_string,target_complaint)
# table(data.cl$dispo[string_index],data.cl$esi[string_index])
string_index[FALSE]<-0
data.cl$weakness<-string_index

#Alerted mental status 
target_complaint<- paste(c("[Aa][Mm][Ss]","[Aa][Ll][Tt][Ee][Rr][Ee]", "[Cc][Oo][Nn][Ff][Uu][Ss]"),collapse = '|')
string_index <-str_detect(target_string,target_complaint)
  # table(data.cl$dispo[string_index],data.cl$esi[string_index])
string_index[FALSE]<-0
data.cl$ams<-string_index

#Distress/Failure/Stroke
target_complaint<- paste(c("[Dd][Ii][Ss][Tt][Rr]","[Ff][Aa][Ii][Ll][Uu]","[Ss][Tt][Rr][Oo][Kk]"),collapse = '|')
string_index <-str_detect(target_string,target_complaint)
  # table(data.cl$dispo[string_index],data.cl$esi[string_index])
string_index[FALSE]<-0
data.cl$distress<-string_index

#Hypertension
target_complaint<- paste(c("[Hh]ypertension","HTN"),collapse = '|')
string_index <-str_detect(target_string,target_complaint)
  # table(data.cl$dispo[string_index],data.cl$esi[string_index])
string_index[FALSE]<-0
data.cl$htn<-string_index

#Anxiety
target_complaint<- paste(c("[Aa][Nn][Xx]","[Pp][Aa][Nn][Ii][Cc]","[Pp][Aa][Ll][Pp][Ii]"),collapse = '|')
string_index <-str_detect(target_string,target_complaint)
  # table(data.cl$dispo[string_index],data.cl$esi[string_index])
string_index[FALSE]<-0
data.cl$anxiety<-string_index

#Minor care complaints
target_complaint<- paste(c("[Cc]ough","[Tt]hroat","[Ss]ore","[Aa]che"),collapse = '|')
string_index <-str_detect(target_string,target_complaint)
  # table(data.cl$dispo[string_index],data.cl$esi[string_index])
string_index[FALSE]<-0
data.cl$uri<-string_index

#GI complaints 
target_complaint<- paste(c("[V]omit","[Nn][Aa][Uu][Ss]","[Nn]/[Vv]"),collapse = '|')
string_index <-str_detect(target_string,target_complaint)
  # table(data.cl$dispo[string_index],data.cl$esi[string_index])
string_index[FALSE]<-0
data.cl$nausea<-string_index

#################################################
### Save a "Reserved" 30% Test Data Partition ###
#################################################

set.seed(279)
train.idx <- createDataPartition(data.cl$dispo, p=0.7,list=FALSE)
train.save <- data.cl[train.idx,]
test.save <- data.cl[-train.idx,]

#Some quick tests to assess test/train partition class distribution
    # table(train.save$dispo)/nrow(train.save)
    # table(test.save$dispo)/nrow(test.save)
    # 
    # table(train.save$esi)/nrow(train.save)
    # table(test.save$esi)/nrow(test.save)
    # 
    # table(train.save$gender)/nrow(train.save)
    # table(test.save$gender)/nrow(test.save)


###############################################
### DATA COLLECTION DEFINITIONS FOR METHODS ###
###############################################

#Defining a Collection Table for Summary of all the Cross Validation Results
row_names <- c("Random_Forest_Original",
               "RF_Downsampled",
               "RF_Mixed",
               "RF_Upsampled",
               "RF_SMOTE_100",
               "RF_SMOTE_200",
               "RF_SMOTE_300",
               "RF_SMOTE_500",
               "AdaBoost_Original",
               "AB_Downsampled",
               "AB_Mixed",
               "AB_Upsampled",
               "AB_SMOTE_100",
               "AB_SMOTE_200",
               "AB_SMOTE_300",
               "AB_SMOTE_500",
               "Naive_Bayes_Original",
               "NB_Downsampled",
               "NB_Mixed",
               "NB_Upsampled",
               "NB_SMOTE_100",
               "NB_SMOTE_200",
               "NB_SMOTE_300",
               "NB_SMOTE_500",
               "Logistic_Original",
               "Log_Downsampled",
               "Log_Mixed",
               "Log_Upsampled",
               "Log_SMOTE_100",
               "Log_SMOTE_200",
               "Log_SMOTE_300",
               "Log_SMOTE_500"
               )
methods<- length(row_names)
result.summary <- data.frame("AUC"=numeric(methods),
                             "pAUC"=numeric(methods),
                             "Accuracy"=numeric(methods),
                             "base_F1"=numeric(methods),
                             "best_F1"=numeric(methods),
                             "best_cut_point"=numeric(methods),
                             "Recall"=numeric(methods),
                             "Precision"=numeric(methods),
                             "Admit_train"=numeric(methods),# of admit class training examples 
                             "DC_train"=numeric(methods), # of DC class training examples 
                             row.names = row_names
                             )
test.summary <- data.frame("AUC"=numeric(methods),
                           "AUC_CI"=numeric(methods),
                           "pAUC"=numeric(methods),
                           "Accuracy"=numeric(methods),
                           "F1"=numeric(methods),
                           "Recall"=numeric(methods),
                           "Precision"=numeric(methods),
                           "TrueAdmit"=numeric(methods),
                           "FalseNeg"=numeric(methods),
                           "FalsePos"=numeric(methods),
                           "DC_correct"=numeric(methods),
                           "Admit_train"=numeric(methods),# of admit class training examples
                           "DC_train"=numeric(methods), # of discharge class training examples 
                            row.names = row_names
                              )


############################################################
### STANDARD CROSS VALIDATION EVALUATION PARAMS /METHODS ###
############################################################

#Defining cross validation fold parameters to all imbalance methods
set.seed(300)
num_folds <- 10
k_fold_train <- createFolds(train.save$dispo, k = num_folds)
cv.stats <- data.frame("AUC"=numeric(num_folds),
                             "pAUC"=numeric(num_folds),
                             "Accuracy"=numeric(num_folds),
                             "base_F1"=numeric(num_folds),
                             "best_F1"=numeric(num_folds),
                             "best_cut_point"=numeric(num_folds),
                             "Recall"=numeric(num_folds),
                             "Precision"=numeric(num_folds),
                             "Admit_train"=numeric(num_folds),# of admit class training examples 
                             "DC_train"=numeric(num_folds) # of DC class training examples 
                              )


#Define the "Best" model classification cutoff based on probability (In terms of F1) 
# during Cross Validation to take to Model Testing
evaluateFoldCutoff <- function(pred.probs,validations,admits,dcs){
  
  #Predicted classifications from predicted class probabilies, with a variable cutoff parameter
  graph_model_params <- data.frame("Cutoff" = numeric(99),
                                   "Recall" = numeric(99),
                                   "Precision" = numeric(99),
                                   "F1" = numeric(99)
                                    )
  for (k in 1:99){
      predictions <-numeric(length(validations))
      cutoff = k*0.01
    for (j in 1:length(validations)){
      predictions[j]<- if (pred.probs[j] > cutoff) "Admit" else "DC"
    }
      
    #calculate fold statistics
    graph_model_params[[k,"Cutoff"]] <- k*0.01
    this.recall<- sum(predictions == "Admit" & validations == "Admit")/sum(validations == "Admit")
    graph_model_params[[k,"Recall"]]<-this.recall
    this.precision<- sum(predictions == "Admit" & validations == "Admit")/sum(predictions == "Admit")
    graph_model_params[[k,"Precision"]]<-this.precision
    graph_model_params[[k,"F1"]]<- 2/(1/this.recall + 1/this.precision)
  }
  #which iteration is best? 
  cutoff <- graph_model_params[which.max(graph_model_params$F1),"Cutoff"]
 
  #define collector object for cross validation reporting 
  this.stats <- data.frame("AUC"=numeric(),
                           "pAUC"=numeric(),
                           "Accuracy"=numeric(),
                           "base_F1"=numeric(),
                           "best_F1"=numeric(),
                           "best_cut_point"=numeric(),
                           "Recall"=numeric(),
                           "Precision"=numeric(),
                           "Admit_train"=numeric(),# of admit class training examples 
                           "DC_train"=numeric() # of DC class training examples 
                            )   
  
  #calculate fold statistics for cross validation reporting 
  this.stats[1,"Accuracy"] <- mean(predictions == validations)
  this.stats[1,"base_F1"] <- graph_model_params[which(graph_model_params$Cutoff == 0.50 ),"F1"]
  this.stats[1,"best_F1"] <- graph_model_params[which.max(graph_model_params$F1),"F1"]
  this.stats[1,"best_cut_point"] <- graph_model_params[which.max(graph_model_params$F1),"Cutoff"]
  this.stats[1,"Recall"]<-graph_model_params[which.max(graph_model_params$F1),"Recall"]
  this.stats[1,"Precision"]<-graph_model_params[which.max(graph_model_params$F1),"Precision"]
  this.stats[1,"Admit_train"]<- admits # of admit class training examples
  this.stats[1,"DC_train"]<- dcs # of discharge class training examples 
  
  return(this.stats)
  
}

#Define the cross validation processing of predicted class probabilities over multiple folds for ROC calculation
rocProcessFold <- function(pred_probs,validation_class,cv.roc){
  this.roc <- data.frame("prob"=pred_probs,"dispo"=validation_class)
  cv.roc <- rbind(cv.roc,this.roc)
  return(cv.roc)
}

################################################
### STANDARD TEST EVALUATION PARAMS /METHODS ###
################################################

#evaluation of the test stats 
evaluateCutoffStats <-function (pred.probs,validations,this.row_name,admits,dcs){
  cutoff <- result.summary[this.row_name,"best_cut_point"] 
  cutoff <- if (cutoff > 0) cutoff else 0.5 

  #define test calculation collector object 
  this.stats <- data.frame("AUC"=numeric(),
                           "AUC_CI"=numeric(),
                           "pAUC"=numeric(),
                           "Accuracy"=numeric(),
                           "F1"=numeric(),
                           "Recall"=numeric(),
                           "Precision"=numeric(),
                           "TrueAdmit"=numeric(),
                           "FalseNeg"=numeric(),
                           "FalsePos"=numeric(),
                           "DC_correct"=numeric(),
                           "Admit_train"=numeric(),# of admit class training examples
                           "DC_train"=numeric() # of discharge class training examples 
  )   
  
  #Predicted classifications from predicted class probabilies, with a variable cutoff parameter
  predictions <- vector(length = length(validations))
  for (j in 1:length(validations)){
    predictions[j]<- if (pred.probs[j] > cutoff) "Admit" else "DC"
  }
  
  #calculate test statistics for each model test 
  this.stats[1,"Accuracy"] <- mean(predictions == validations)
  this.recall<- sum(predictions == "Admit" & validations == "Admit")/sum(validations == "Admit")
  this.precision<- sum(predictions == "Admit" & validations == "Admit")/sum(predictions == "Admit")
  this.stats[1,"F1"] <- 2/(1/this.recall + 1/this.precision) 
  this.stats[1,"Recall"]<-this.recall
  this.stats[1,"Precision"]<-this.precision
  this.stats[1,"TrueAdmit"]<- sum(predictions == "Admit" & validations == "Admit")
  this.stats[1,"FalsePos"]<- sum(predictions == "Admit" & validations == "DC")
  this.stats[1,"FalseNeg"]<- sum(predictions == "DC" & validations == "Admit")
  this.stats[1,"DC_correct"]<- sum(predictions == "DC" & validations == "DC")
  this.stats[1,"Admit_train"]<- admits # of admit class training examples
  this.stats[1,"DC_train"]<- dcs # of discharge class training examples 
  
  return(this.stats)

}


#########################################
##### MAIN Learning Model Function ######
########################################
library(randomForest)
library(ada)
library(e1071)

#expression for model relationship with predictors 
relationship_expr <- dispo ~ esi + age_yrs + 
                            gender + time + 
                            weekday + stroke_alert +
                            order_name + presenting_problem + 
                            provider_type + intake + 
                            door_to_provider_mins + intox + 
                            headache + sob + ams + 
                            htn + eye + seizure + 
                            si + distress + weakness + 
                            anxiety + uri + nausea

#define the RandomForest Params for all later Preprocessing Methods
myModel <- function(train_data,validation.set,model_method){
  if (model_method == "forest"){
      model <- randomForest(relationship_expr,
                        data = train_data, 
                        ntree = 500, 
                        importance = TRUE) 
      pred_probs <- predict(model, newdata = validation.set, type = "prob")
      return_probs <- pred_probs[,1]
      
      
      #Outputing Model importance
      print(importance(model))
      drops<-c("dispo","complaint","los_mins","provider_to_decision_mins","event_count")
      temp <-filterVarImp(validation.set[,!names(validation.set) %in% drops],return_probs)
      temp <-data.frame(row.names(temp),temp[with(temp, order(-temp$Overall)),])                    
      print(temp)
  }else if (model_method == "boost"){
    model <- ada(relationship_expr,
                 data=train_data,
                 iter=50,
                 type="discrete")
    pred_probs <- predict(model, newdata = validation.set,type="probs")
    return_probs <- pred_probs[,1]
    
    #Outputing Model importance
    drops<-c("dispo","complaint","los_mins","provider_to_decision_mins","event_count")
    temp <-filterVarImp(validation.set[,!names(validation.set) %in% drops],return_probs)
    temp <-data.frame(row.names(temp),temp[with(temp, order(-temp$Overall)),])                    
    print(temp)
  }
  else if (model_method == "bayes"){
      model <- naiveBayes(relationship_expr,
                        data = train_data,
                        laplace = 1)
      pred_probs <- predict(model, newdata = validation.set, type = "raw")
      return_probs <- pred_probs[,1]
      
      #Outputing Model importance
      drops<-c("dispo","complaint","los_mins","provider_to_decision_mins","event_count")
      temp <-filterVarImp(validation.set[,!names(validation.set) %in% drops],return_probs)
      temp <-data.frame(row.names(temp),temp[with(temp, order(-temp$Overall)),])                    
      print(temp)
  }else if (model_method == "logistic"){
      model <- glm(relationship_expr,
                        data = train_data, family="binomial")
      pred_probs <- predict(model, newdata = validation.set, type = "response")
      return_probs <- 1 - pred_probs
      
      #Outputing Model and importance
      print(summary(model)) 
      drops<-c("dispo","complaint","los_mins","provider_to_decision_mins","event_count")
      temp <-filterVarImp(validation.set[,!names(validation.set) %in% drops],return_probs)
      temp <-data.frame(row.names(temp),temp[with(temp, order(-temp$Overall)),])                    
      print(temp)
  }
  return(return_probs) 
}


###########################################
#### MAIN Imbalance Handling Function  ####
###########################################

library(DMwR)

myBalance <- function(train.set,imbalance_method){
  if (imbalance_method == "none"){
   train_balanced <- train.set
  }else if(imbalance_method == "upsample"){
    drops<-c("dispo")
    train_balanced <- upSample(train.set[, !(names(train.set) %in% drops)], train.set$dispo, list = FALSE, yname = "dispo")
    
  }else if(imbalance_method =="downsample"){
    drops<-c("dispo")
    train_balanced <- downSample(train.set[, !(names(train.set) %in% drops)], train.set$dispo, list = FALSE, yname = "dispo")
  }else if(imbalance_method == "mixed"){
    drops<-c("dispo")
    train.set.idx <- createDataPartition(train.set$dispo, times = 1, p=0.5, list = FALSE)
    
    train.set.1 <- train.set[train.set.idx,]
    train.set.2 <- train.set[-train.set.idx,]
    
    train.set.1 <- downSample(train.set.1[, !(names(train.set.1) %in% drops)], train.set.1$dispo, list = FALSE, yname = "dispo")
    train.set.2 <- upSample(train.set.2[, !(names(train.set.2) %in% drops)], train.set.2$dispo, list = FALSE, yname = "dispo")
    
    train_balanced <- rbind(train.set.1,train.set.2)
  }else if(imbalance_method == "smote100"){
    train_balanced <- SMOTE(relationship_expr,
                            train.set, 
                            k = 10,
                            perc.over = 100, 
                            perc.under = 600
    )
  }else if(imbalance_method == "smote200"){
    train_balanced <- SMOTE(relationship_expr,
                      train.set, 
                      k = 10,
                      perc.over = 200, 
                      perc.under = 300
    )
  }else if(imbalance_method == "smote300"){
    train_balanced <- SMOTE(relationship_expr,
                            train.set, 
                            k = 10,
                            perc.over = 300, 
                            perc.under = 200
    )
  }else if(imbalance_method == "smote500"){
    train_balanced <- SMOTE(relationship_expr,
                            k= 10,
                            train.set, 
                            perc.over = 500, 
                            perc.under = 125
    )
  }
  
  return(train_balanced) 
}

##############################################
####  Cross Validation "Wrapper" Function   ##
##############################################

# This function runs a cross validation based on provided method parameters
# for balancing data (calls myBalance function) and a model (calls myModel function)
# This is written to simplify the code for comparing multiple preprocessing methods and models 
# and to eliminate redundant code. 

myCrossValidation <- function(this.row_name,balance_method,model_method,output_table){
cv.roc <- data.frame("prob"=numeric(),"dispo"=character())

  for(i in 1:num_folds){
    validation.set.idx <- k_fold_train[[i]]
    
    train.set <- myBalance(train.save[-validation.set.idx,],balance_method) 
    validation.set <- train.save[validation.set.idx,]
    
    pred_probs <- myModel(train.set,validation.set,model_method)
    
    admits<-sum(train.set$dispo == "Admit")
    dcs<-sum(train.set$dispo == "DC")
    
    cv.roc <- rocProcessFold(pred_probs,validation.set$dispo,cv.roc)
    cv.stats[i,] <- evaluateFoldCutoff(pred_probs,validation.set$dispo,admits,dcs)
    
    print(cv.stats[i,])
  }

  rocobj<-roc(cv.roc[,"dispo"],cv.roc[,"prob"],levels=c("DC","Admit") , # creating ROC from fold data
            partial.auc = c(1,0.85), partial.auc.correct = TRUE,
            percent=FALSE
  ) 
  
  #takes existing table and adds entry
  output_table[this.row_name,] <- apply(cv.stats,2,mean) #ave stats and save 
  output_table[this.row_name,"AUC"]<-auc(rocobj) 
  output_table[this.row_name,"pAUC"]<-auc(rocobj,partial.auc = c(1,0.85),partial.auc.focus = c("specificity"),partial.auc.correct = TRUE)
  print(output_table[this.row_name,])
  
  #list of two objects, ROC and revised table 
  results<-list(roc = rocobj,table = output_table)
  
  return(results)
} 

##############################################
####   The 24 Cross Validation Instances    ##
##############################################


# The following are all the desired cross validation combinations. 
results <- myCrossValidation("Random_Forest_Original","none","forest",result.summary)
rocobj.RF_Original.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("RF_Downsampled","downsample","forest",result.summary)
rocobj.RF_Downsampled.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("RF_Upsampled","upsample","forest",result.summary)
rocobj.RF_Upsampled.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("RF_Mixed","mixed","forest",result.summary)
rocobj.RF_Mixed.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("RF_SMOTE_100","smote100","forest",result.summary)
rocobj.RF_SMOTE_100.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("RF_SMOTE_200","smote200","forest",result.summary)
rocobj.RF_SMOTE_200.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("RF_SMOTE_300","smote300","forest",result.summary)
rocobj.RF_SMOTE_300.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("RF_SMOTE_500","smote500","forest",result.summary)
rocobj.RF_SMOTE_500.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("AdaBoost_Original","none","boost",result.summary)
rocobj.AdaBoost_Original.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("AB_Downsampled","downsample","boost",result.summary)
rocobj.AB_Downsampled.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("AB_Upsampled","upsample","boost",result.summary)
rocobj.AB_Upsampled.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("AB_Mixed","mixed","boost",result.summary)
rocobj.AB_Mixed.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("AB_SMOTE_100","smote100","boost",result.summary)
rocobj.AB_SMOTE_100.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("AB_SMOTE_200","smote200","boost",result.summary)
rocobj.AB_SMOTE_200.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("AB_SMOTE_300","smote300","boost",result.summary)
rocobj.AB_SMOTE_300.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("AB_SMOTE_500","smote500","boost",result.summary)
rocobj.AB_SMOTE_500.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("Naive_Bayes_Original","none","bayes",result.summary)
rocobj.Naive_Bayes_Original.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("NB_Downsampled","downsample","bayes",result.summary)
rocobj.RF_Downsampled.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("NB_Upsampled","upsample","bayes",result.summary)
rocobj.NB_Upsampled.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("NB_Mixed","mixed","bayes",result.summary)
rocobj.NB_Mixed.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("NB_SMOTE_100","smote100","bayes",result.summary)
rocobj.NB_SMOTE_100.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("NB_SMOTE_200","smote200","bayes",result.summary)
rocobj.NB_SMOTE_200.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("NB_SMOTE_300","smote300","bayes",result.summary)
rocobj.NB_SMOTE_300.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("NB_SMOTE_500","smote500","bayes",result.summary)
rocobj.NB_SMOTE_500.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("Logistic_Original","none","logistic",result.summary)
rocobj.Logistic_Original.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("Log_Downsampled","downsample","logistic",result.summary)
rocobj.Log_Downsampled.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("Log_Upsampled","upsample","logistic",result.summary)
rocobj.Log_Upsampled.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("Log_Mixed","mixed","logistic",result.summary)
rocobj.Log_Mixed.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("Log_SMOTE_100","smote100","logistic",result.summary)
rocobj.Log_SMOTE_100.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("Log_SMOTE_200","smote200","logistic",result.summary)
rocobj.Log_SMOTE_200.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("Log_SMOTE_300","smote300","logistic",result.summary)
rocobj.Log_SMOTE_200.cv<-results$roc
result.summary<-results$table 

results <- myCrossValidation("Log_SMOTE_500","smote500","logistic",result.summary)
rocobj.Log_SMOTE_200.cv<-results$roc
result.summary<-results$table 

write.csv(result.summary, file= "validation_12_20.csv" )

#######################################
####  The Test "Wrapper" Function   ###
#######################################

# This function runs a model for the test set based on method
# for balancing data (calls myBalance function) and a model (calls myModel function)
# This is written to simplify the code for comparing multiple preprocessing methods and models 
# and to eliminate redundant code. 

myTesting <- function(this.row_name,balance_method,model_method,output_table){
   test.roc <- data.frame("prob"=numeric(),"dispo"=character())
    
    train.set <- myBalance(train.save,balance_method) 
    
    pred_probs <- myModel(train.set,test.save,model_method)
    
    admits<-sum(train.set$dispo == "Admit")
    dcs<-sum(train.set$dispo == "DC")
   
    test.roc <- data.frame("prob"=pred_probs,"dispo"=test.save$dispo)
    test.stats <- evaluateCutoffStats(pred_probs,test.save$dispo,this.row_name,admits,dcs)
  
  rocobj<-roc(test.roc[,"dispo"],test.roc[,"prob"],levels=c("DC","Admit"),
              partial.auc = c(1,0.85), partial.auc.correct = TRUE, 
              ci=TRUE, 
              percent=FALSE
  )
  #takes existing table and adds entry
  output_table[this.row_name,] <- test.stats
  output_table[this.row_name,"AUC"]<-auc(rocobj) 
  # output_table[this.row_name,"AUC_CI"]<-(mean(ci(rocobj))-min(ci(rocobj))) 
  output_table[this.row_name,"pAUC"]<-auc(rocobj,partial.auc = c(1,0.85),partial.auc.focus = c("specificity"),partial.auc.correct = TRUE)

  #list of two objects, ROC and revised table 
  results<-list(roc = rocobj,table = output_table)
  
  return(results)
} 

##############################################
####    The  24   Test Instances          ####
##############################################

# The following are all the desired testing combinations. 
results <- myTesting("Random_Forest_Original","none","forest",test.summary)
rocobj.RF_Original<-results$roc
test.summary<-results$table 

results <- myTesting("RF_Downsampled","downsample","forest",test.summary)
rocobj.RF_Downsampled<-results$roc
test.summary<-results$table 

results <- myTesting("RF_Upsampled","upsample","forest",test.summary)
rocobj.RF_Upsampled<-results$roc
test.summary<-results$table 

results <- myTesting("RF_Mixed","mixed","forest",test.summary)
rocobj.RF_Mixed<-results$roc
test.summary<-results$table 

results <- myTesting("RF_SMOTE_100","smote100","forest",test.summary)
rocobj.RF_SMOTE_100<-results$roc
test.summary<-results$table 

results <- myTesting("RF_SMOTE_200","smote200","forest",test.summary)
rocobj.RF_SMOTE_200<-results$roc
test.summary<-results$table 

results <- myTesting("RF_SMOTE_300","smote300","forest",test.summary)
rocobj.RF_SMOTE_300<-results$roc
test.summary<-results$table 

results <- myTesting("RF_SMOTE_500","smote500","forest",test.summary)
rocobj.RF_SMOTE_500<-results$roc
test.summary<-results$table 

results <- myTesting("AdaBoost_Original","none","boost",test.summary)
rocobj.AdaBoost_Original<-results$roc
test.summary<-results$table 

results <- myTesting("AB_Downsampled","downsample","boost",test.summary)
rocobj.AB_Downsampled<-results$roc
test.summary<-results$table 

results <- myTesting("AB_Upsampled","upsample","boost",test.summary)
rocobj.AB_Upsampled<-results$roc
test.summary<-results$table 

results <- myTesting("AB_Mixed","mixed","boost",test.summary)
rocobj.AB_Mixed<-results$roc
test.summary<-results$table 

results <- myTesting("AB_SMOTE_100","smote100","boost",test.summary)
rocobj.AB_SMOTE_100<-results$roc
test.summary<-results$table 

results <- myTesting("AB_SMOTE_200","smote200","boost",test.summary)
rocobj.AB_SMOTE_200<-results$roc
test.summary<-results$table 

results <- myTesting("AB_SMOTE_300","smote300","boost",test.summary)
rocobj.AB_SMOTE_300<-results$roc
test.summary<-results$table 

results <- myTesting("AB_SMOTE_500","smote500","boost",test.summary)
rocobj.AB_SMOTE_500<-results$roc
test.summary<-results$table 

results <- myTesting("Naive_Bayes_Original","none","bayes",test.summary)
rocobj.Naive_Bayes_Original<-results$roc
test.summary<-results$table 

results <- myTesting("NB_Downsampled","downsample","bayes",test.summary)
rocobj.RF_Downsampled<-results$roc
test.summary<-results$table 

results <- myTesting("NB_Upsampled","upsample","bayes",test.summary)
rocobj.NB_Upsampled<-results$roc
test.summary<-results$table 

results <- myTesting("NB_Mixed","mixed","bayes",test.summary)
rocobj.NB_Mixed<-results$roc
test.summary<-results$table 

results <- myTesting("NB_SMOTE_100","smote100","bayes",test.summary)
rocobj.NB_SMOTE_100<-results$roc
test.summary<-results$table 

results <- myTesting("NB_SMOTE_200","smote200","bayes",test.summary)
rocobj.NB_SMOTE_200<-results$roc
test.summary<-results$table 

results <- myTesting("NB_SMOTE_300","smote300","bayes",test.summary)
rocobj.NB_SMOTE_300<-results$roc
test.summary<-results$table 

results <- myTesting("NB_SMOTE_500","smote500","bayes",test.summary)
rocobj.NB_SMOTE_500<-results$roc
test.summary<-results$table 

results <- myTesting("Logistic_Original","none","logistic",test.summary)
rocobj.Logistic_Original<-results$roc
test.summary<-results$table 

results <- myTesting("Log_Downsampled","downsample","logistic",test.summary)
rocobj.Log_Downsampled<-results$roc
test.summary<-results$table 

results <- myTesting("Log_Upsampled","upsample","logistic",test.summary)
rocobj.Log_Upsampled<-results$roc
test.summary<-results$table 

results <- myTesting("Log_Mixed","mixed","logistic",test.summary)
rocobj.Log_Mixed<-results$roc
test.summary<-results$table 

results <- myTesting("Log_SMOTE_100","smote100","logistic",test.summary)
rocobj.Log_SMOTE_100<-results$roc
test.summary<-results$table 

results <- myTesting("Log_SMOTE_200","smote200","logistic",test.summary)
rocobj.Log_SMOTE_200<-results$roc
test.summary<-results$table 

results <- myTesting("Log_SMOTE_300","smote300","logistic",test.summary)
rocobj.Log_SMOTE_200<-results$roc
test.summary<-results$table 

results <- myTesting("Log_SMOTE_500","smote500","logistic",test.summary)
rocobj.Log_SMOTE_200<-results$roc
test.summary<-results$table 

write.csv(test.summary, file="test_12_20.csv")

#################################################################
####  Drawing ROC curves for display/ROC signficiance tests  ####
#################################################################

title <- ""
roc1 <- rocobj.RF_Original
roc2 <- rocobj.AdaBoost_Original
roc3 <- rocobj.Naive_Bayes_Original
roc4 <- rocobj.Logistic_Original

# roc.test(roc1,roc2)
# roc.test(roc1,roc2, reuse.auc = FALSE, partial.auc=c(1,0.85), partial.auc.focus = "sp")

plot(roc1,
     main=title,
     type="l",col="#70AD47",
     partial.auc = c(1,0.85),partial.auc.focus = c("specificity"),
     partial.auc.correct = TRUE,lwd=3,
     auc.polygon = TRUE,auc.polygon.col="#cccccc66",
     max.auc.polygon=TRUE, max.auc.polygon.col="#cccccc20",
     legacy.axes=TRUE, 
     # print.auc = TRUE, 
     xlab="False Positive Rate (%)",ylab="True Positive Rate (%)"
     ) 

plot(roc2,add = TRUE,
     type="l",col="#ED7D3180",lwd=3
) 
plot(roc3,add = TRUE,
     type="l",col="#86868690",lwd=3
) 

plot(roc4,add = TRUE,
     type="l",col="#4472C490",lwd=3
) 

legend("bottomright", legend=c("Random Forest","AdaBoost","Naive_Bayes","Logistic"), col=c("#70AD47","#ED7D31","#868686","#4472C4"), lwd=2)

