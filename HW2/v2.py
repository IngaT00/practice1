import pandas as pd
df1 = pd.read_parquet("../HW1/green_tripdata_2021-01.parquet")
df2 = pd.read_parquet("green_tripdata_2021-02.parquet")

df= pd.concat([df1, df2], ignore_index=True)
print(df.head())

df=df[["fare_amount","trip_distance","payment_type", "tip_amount"]]
df=df.dropna()
df=df[df["payment_type"].isin([1,2])]
X=df[["trip_distance","fare_amount","tip_amount"]]
y=df["payment_type"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
import joblib
model = joblib.load("model_v1.pkl")
prediction=model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,prediction)
print("Accuracy for version 2: ",accuracy)