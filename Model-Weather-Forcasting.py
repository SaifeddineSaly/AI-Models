#!/usr/bin/env python
# coding: utf-8

# In[ ]:


################################# p002_data_collect
def p002_data_collect (CSV_file):
    import pandas as pd
    # i) Reading the dataset or Load CSV file: df 
    df = pd.read_csv(CSV_file)
    # ii) Print ('the size of the csv weather data frame is: ‘, df.shape)
        # Result: the size of the csv weather data frame is:  (145460, 24) =>
        # The dimension of the data frame is:  145 460 rows and 24 columns
    
    # iii) Display the first five observations in our data frame
    five_rows = df[0:5]
    # print (five rows)
    return df , five_rows
#df, five_rows  = p002_data_collect (r"C:\17_IUT_AI\Model\02_weather.csv")
df, five_rows  = p002_data_collect ("02_weather.csv")
print (df, five_rows )


# In[ ]:


################################ p003_data_preparation
def p003_data_preparation (df):
    from scipy import stats
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    
    # a. Checking Null or missing values: The count() method counts the number of not empty values for each row, 
            # or column if you specify the axis parameter as axis='columns', 
            # and returns a Series object with the result for each row (or column).
    null_val = df.count().sort_values()
    print (null_val)
    
    # b. Remove unwanted and redundant columns
      # i) Unecessary data will always increase our computations that is why it is always better
          # to remove them.
      # ii) So apart from removing the unnecessary variables, we also will remove the 
          # "location variables” and we will remove the "date variable" because both 
          # of these two variables are not needed in order to predict whether it will rain 
          # tomorrow or not.         
      # iii) # We will also remove “RISK_MM" variable because this tells us the amount 
                  # of rain that might occur the next day. 
             # This is a very informative variable and it may actually leak some information to our model. 
             # By using this variable it will be able easy to predict 'RainTom' 
             # This variable will give us too much information and that is why we are going to remove it.            
             # Because we will let the model to discover whether it rains or not based on the training process            
                 # and since this variable, leaks a lot of information so it should be dropped from the dataSet.    
    rain_drop_unwanted = df.drop(['Sunshine',  'Evaporation' , 'Cloud3pm', 'Cloud9am', 'Location', 'RISK_MM', 'Date'],axis=1)
    
    
    # c. Remove null values from the Last data Frame
    rain_drop_unwanted_and_null = rain_drop_unwanted.dropna(how='any')    
    
    # d. Remove Outliers (Tmin = 113 instead of 11.3)
       # i) Now it is the time to remove the outliers inside the dataframe.     
      # ii) The outlier is a data that is very different from the other observations.         
      # iii) Outlier’s usually occur because of miscalculations while collecting the data.    
      # iiii) (ex: T=115 instead of 11.5). These are some sort of errors in the data set.
    rain_drop_unwanted_and_null_rmvOutliers = np.abs(stats.zscore(rain_drop_unwanted_and_null._get_numeric_data())) 
    rain_drop_unwanted_and_null_rmvOutliersNull= rain_drop_unwanted_and_null [(rain_drop_unwanted_and_null_rmvOutliers < 3).all(axis=1) ]
    
    
    # e. Handling categorical variable     
       # i) Now what we will be doing is we will be assigning 0 and 1 to the place of yes and no.
       # ii) That means we are going to change the categorical variables from yes and no to 0 and 1 .
    df_lables = rain_drop_unwanted_and_null_rmvOutliersNull
    df_lables['RainToday'].replace    ({'No'  :  0 , 'Yes' : 1} , inplace=True    )
    df_lables['RainTomorrow'].replace ({'No'  :  0 , 'Yes' : 1} , inplace=True    )   
    
    # f. Handling unique keys,character values will be changed into integer values
       # If we have unique values such as any character values which are not supposed to be there, 
            # we will change them into integer values
    df_char_to_num = df_lables    
    # See unique values and change them into int using pd.getDummiies()
    categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
    for col in categorical_columns:
        # print(np.unique(df_char_to_num[col]))
        print ("")
    #Transform the categorical columns
    dfFinal = pd.get_dummies(df_lables , columns=categorical_columns)
    # print (df.iloc[4:9])
    # print (df) #[107868 rows x 62 columns]    
   
    # g. Normalize all data in the recent data-frame
       # i) Now we will be proceeding to normalizing Data
       # ii) Standardize our data by using MinMaxScaler
       # iii) Google: how to standardize data in python??: https://www.askpython.com/python/examples/standardize-data-in-python
       # iiii) This normalizing process is very important because to reduce or avoid any biases in your output 
       # v) you should normalize your input variables. 
       # vi) It was done by using the function Minm=MaxScaler provided by python in a package known as #sklearn.    
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(dfFinal)
    dfFinal_Std = pd.DataFrame(scaler.transform(dfFinal), index=dfFinal.index, columns=dfFinal.columns)
    
    return (null_val, rain_drop_unwanted_and_null_rmvOutliersNull,  df_lables, dfFinal ,dfFinal_Std)

nullValues , rain_drop_unwanted_and_null_rmvOutliersNull , df_lables, dfFinal,  df_data_prep  =  p003_data_preparation (df)
# print (df_data_prep)


# In[ ]:


################################ p004_data_explarotary
# Now we well go to the step EDA, Exploratory Data Analysis
def p004_data_explarotary (df):
    # Now what we are going to do, is get analyzed and identify the significant
     #variables that will help us to forecast the dependant variable (RainTom).
      # i) To do this, we will use the 'selctKeyBest' function getting from Sklearn library.             
      # ii) Using this function, we will select the most significant independent variables in our dataset.
      # iii) Google: how to select features using chi squared in python?
                    # OR
            # Google: how to select features  in python?    
    from sklearn.feature_selection import SelectKBest, chi2
    
    X = df.loc[: , df.columns != 'RainTomorrow']
    Y = df['RainTomorrow']
    selector = SelectKBest(chi2, k=3)
    selector.fit(X, Y)
    x_new  = selector.transform(X)
    print (X.columns[selector.get_support(indices=True)])    
    # i) Hence, we get the most significant independent variables in our dataset that influence the
    #    dependent variable 'RainTomorrow'
    # ii) Index(['Rainfall', 'Humidity3pm', 'RainToday'], dtype='object')
    # iii) We just enough to feed our models by these 3 variables instead of all variables dataset as input
    # iv) This simplifies the computation process
    # vi) Basically we will create a data frame of the significant variables overall
    df_Best_Feature = df[['Humidity3pm', 'Rainfall',  'RainToday',  'RainTomorrow']]
    
    
    # vii) What would be done later is assigning one of these significant variables as input instead of 
             #taking all three variables to predict the  'RainTomorrow' variable
    # viii) Let's use only one feature 'Humidity3pm'
    X_Best_Feature  =  df[['Humidity3pm']] 
    # ix) Obviously our outcome is  'RainTomorrow' the variable to be predicted.
    Y_Target_Fearure =  df[['RainTomorrow']]    
    return  df_Best_Feature, X_Best_Feature, Y_Target_Fearure   
    
df_Best_Feature, X_Best_Feature, Y_Target_Fearure= p004_data_explarotary(df_data_prep)


# In[ ]:


# Part 05#################################### p005_Build_Model

    # i)  Now we are processing to data modeling 
    # ii) I suppose now that we are aware of what data modeling is to solve this step.
    # iii) we will be using four classification algorithms over here in order to predict the outcome "RainTomorrow".
            #  - LogisticRegression
            #  - Random Forest 
            #  - Decision Tree Classifier
            #  - Support Vector Machine
    # iv)  Finally, we will check the best algorithm that will give us the best accuracy
    # v)   We will continue by applying the LogisticRegression algorithm

def p005_Build_Model (X_BestFeature, Y_TargetFearure):      
    # vi) Import all the necessary libraries for the 'LogisticRegression' algorithm     
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    
    # vii)  We are importing the 'time' libraries because we will calculate the accuracy
            # and the time taken by the algorithm to finish the model's execution.   
    import time    
    t0 =time.time()
    
    # Split data into 4 parts
    X_train, X_test, Y_train, Y_test = train_test_split (X_BestFeature,Y_TargetFearure, test_size=0.25)
    
    #  Create an instance of the 'LogisticRegression' algorithm
    model_LogesticRegresson = LogisticRegression(random_state=0)

    # Building the model using the training data set, that means calculated  the coefficient of the model's equation
    model_LogesticRegresson.fit(X_train, Y_train)
    
    return X_test, Y_test, model_LogesticRegresson;

    
Xtst, Ytst, model_LR = p005_Build_Model (X_Best_Feature, Y_Target_Fearure)  
    


# In[ ]:



################################ p006_Model_Evaluation
 # i) In this step we should apply the model equation on the Xtest and generate Yhat (Yhat, Ypredcit)
 # ii) Compare the Ytest real with Yhat (making the difference)
 # iii) Check the efficiency of the model and how accurately, it can predict the outcome.

def p006_Model_Evaluation (model, Xtest, Ytest):
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import time    
t0 =time.time()

Yhat = model.predict(Xtest)
accuracy = accuracy_score(Ytest, Yhat)

print("Accuracy using 'LogisticRegression'   :  "  , accuracy)
print("Time taken using 'LogisticRegression' :"   , time.time()-t0)

return (Yhat, Ytest , accuracy)


Yhat, Ytest, precision = p006_Model_Evaluation (model_LR, Xtst, Ytst)   
print(Yhat, Ytest, precision)


# In[ ]:




