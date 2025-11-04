## Implementation-of-K-Means-Clustering-for-Customer-Segmentation
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm:
Import the standard libraries.
Upload the dataset and check for any null values using .isnull() function.
Import LabelEncoder and encode the dataset.
Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
Predict the values of arrays.
Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
Predict the values of array.
Apply to new unknown values
## Program:
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: yashwanth asv
RegisterNumber:212224230309
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict(pd.DataFrame([[5,6]], columns=x.columns))
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=list(x.columns), filled=True)
plt.show()
```

## Output:
Data Head:

<img width="486" height="366" alt="505610421-5f0b29e0-00f8-43bb-b8bb-b0efb09bdce3" src="https://github.com/user-attachments/assets/7bd03792-77d6-4333-8cb8-ebc4e972e5f8" />

Data Info :

<img width="475" height="300" alt="505610429-9467f515-3d6c-4ff1-9117-f32db8d2a6a6" src="https://github.com/user-attachments/assets/24584a9e-b197-4dcc-8b8f-e90761d7ab61" />

Data Details :

<img width="657" height="774" alt="505610445-93ac721b-8c48-4c4f-9952-6dceb3970894" src="https://github.com/user-attachments/assets/8fee4197-c53d-48a3-be0a-3a3995cd5429" />

Data Predcition :

<img width="1034" height="391" alt="505610483-b96f5940-1f5c-44c7-9e8c-7390aeb5f416" src="https://github.com/user-attachments/assets/bcd3dff3-06b1-4ffe-b316-9066e26b301e" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
