!pip install pandas numpy
import numpy as np
import pandas as pd
# data preprocessing
# data collection
data = pd.read_csv(r"C:\project\Telco_Customer_Churn.csv")
df=data.copy()
df.info()
df.duplicated().sum()
df[df.duplicated()]
print("total duplicated rows:")
df.drop(columns='customerID', axis =1, inplace = True)
df['TotalCharges']= pd.to_numeric(df['TotalCharges'], errors ='coerce')
df['TotalCharges'].dtype
df.info()
df.isnull().sum()
df.dropna(inplace = True)
df.isnull().sum()
df.describe()
# save clean data
df.to_csv("data_cleaned.csv", index=False)


