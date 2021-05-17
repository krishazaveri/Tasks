# Tasks
#Importing the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url=r"https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
dataset=pd.read_csv(url)
x=dataset.iloc[:, :-1].values #gives until the last column
y=dataset.iloc[:, -1].values #gives the last column
print("Data Imported")
dataset.head(25)

plt.scatter(x,y)
plt.title("Visualizing the data")
plt.xlabel("Number of hours studied")
plt.ylabel("Scores")
plt.show

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #the linear regression function is assigned to the variable regressor.
regressor.fit(x_train,y_train) #the regressor.fit function is fitted with x_train and y_train on which the model will be trained

line=regressor.coef_*x + regressor.intercept_
#Plotting the test data and regression line
plt.scatter(x,y)
plt.plot(x,line)
plt.show()

print(x_test)
Y=regressor.predict(x_test)

data=pd.DataFrame({"Actual":y_test, "Predicted":Y})
print(data)

#Plotting the training set
plt.scatter(x_train,y_train, color = 'hotpink')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours") # adding the name of x-axis
plt.ylabel("Scores") # adding the name of y-axis
plt.show() 

#Plotting the test set
plt.scatter(x_test,y_test, c='hotpink')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title("Hours vs Scores(Testing Set)")
plt.xlabel("Hours") # adding the name of x-axis
plt.ylabel("Scores") # adding the name of y-axis
plt.show() 

hours=9.25
pred = regressor.predict([[hours]])
print("Predicted score when student studies 9.25 hours/day:{}".format(pred[0]))

from sklearn import metrics
print("Mean absolute error:",metrics.mean_absolute_error(y_test,Y))
