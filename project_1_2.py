# -*- coding: utf-8 -*-

!pip install xgboost 
!pip install joblib
!pip install scikit-learn

# import necessary libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn. svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import joblib
from sklearn.compose import ColumnTransformer
from sklearn import metrics
import warnings

# data load
data = pd.read_csv("data_cleaned.csv")
data.head()

#backup
df = data.copy()
df.info()

# encoding churn column
df['Churn']= df['Churn'].map({'No': 0,'Yes':1})
df['Churn']

# set target and feature
X =df.drop( columns = 'Churn', axis = 1)
y = df['Churn']

# separate numeric and categorical columns 

numeric_columns = X. select_dtypes(include = [int,float]).columns.to_list()
categorical_columns = X.select_dtypes(include = object). columns.to_list()
print("numeric columns are:\n", numeric_columns)
print("categorical columns are:\n", categorical_columns)

#split dataset
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)

# data processing
preprocessor= ColumnTransformer(transformers = [
    ('num', StandardScaler(),numeric_columns),
    ('cat', OneHotEncoder(handle_unknown = 'ignore', sparse_output= False), categorical_columns)

    
])

# fit and transform training data
X_train_processed =  preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# model details

model_dict = {
    'Logistic Regression': LogisticRegression(),
    'SVC': SVC(probability = True),
    'Random Forest': RandomForestClassifier(),
    'Adaboost': AdaBoostClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder = False, eval_metric = 'logloss')

    
}
# hyperparameter details
search_space= {
    'C': [0.1,1,10],
    'kernel': ['linear', 'rbf'],
    'n_estimator': [100,200],
    'max_depth': [None,10],
    'learning_rate': [0.5,1]
}
# custom function to filter hyper parameter
def filter_hyperparameter(model, space):
    valid_keys = model.get_params().keys()
    param_grid = {k:v for k,v in space.items() if k in valid_keys}
    return param_grid

# model training with gridsearch
result =[]
for name, model in model_dict.items():
    print(f'tuning for{name}')
    
    param_grid= filter_hyperparameter(model, search_space)
    grid = GridSearchCV(estimator = model, param_grid = param_grid,cv= 5, scoring = 'accuracy',n_jobs= -1)
    grid.fit(X_train_processed,y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test_processed)

    report = metrics.classification_report(y_test, y_pred, output_dict = True)
    best_params= grid.best_params_
    print (f"best parameter is {name}: {best_params}")
    result.append({
        'model_name': name,
        'best_parameter': best_params,
        'accuracy':round( metrics.accuracy_score(y_test,y_pred), 4),
        'F1 score':round( report['weighted avg']['f1-score'],4)
    })
    
    # finding best model
model_df = pd.DataFrame(result)
model_df

#best model
model_df_sorted = model_df.sort_values(by = "accuracy", ascending = False)
best_row = model_df_sorted .iloc[0]
best_model_name = best_row['model_name']
best_parameter= best_row['best_parameter']
print(f"best model is:{best_model_name}")
print("best parameters are: \n",best_parameter)

#final model
final_model= model_dict[best_model_name].set_params(**best_parameter)

# retrain on whole data
X_full_processed= preprocessor.transform(X)
final_model.fit(X_full_processed,y)

#save pipeline 
deployment_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', final_model)
])
#save
joblib.dump(deployment_pipeline, "churn_pipeline.pkl")
print("final model is saved as 'churn_pipeline.pkl' for final deployment")