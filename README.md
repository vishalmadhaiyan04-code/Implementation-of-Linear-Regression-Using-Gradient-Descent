# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize the model parameters (slope m and intercept b) with initial values (usually 0) and choose a learning rate α.
2. Compute the predicted output using the linear equation
3. Calculate the error using the cost function (Mean Squared Error) and find the gradients of the cost with respect to m and b. 
4. Update the parameters m and b using gradient descent and repeat the steps until the error becomes minimum.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: VISHAL M
RegisterNumber: 212225240186
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data =pd.read_csv("C:/Users/acer/Downloads/50_Startups.csv")
x = data["R&D Spend"].values
y = data["Profit"].values
x_mean = np.mean(x)
x_std = np.std(x)
x = (x - x_mean)/ x_std
w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)

losses=[]
for _ in range(epochs):
    y_hat = w* x + b
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)
    
    dw = (2/n) * np.sum((y_hat-y) * x)
    db = (2/n) * np.sum(y_hat - y)
    
    w -=alpha * dw
    b -=alpha * db
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1,2,2)
plt.scatter(x,y)
x_sorted = np.argsort(x)
plt.plot(x[x_sorted],(w*x+b)[x_sorted],color='red')
plt.xlabel("R&D Spend(scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")

plt.tight_layout()
plt.show()

print("Final weight (w):",w)
print("Final bias (b):",b)
*/
```

## Output:

<img width="1062" height="492" alt="exp img4,ML" src="https://github.com/user-attachments/assets/46990996-5a88-483f-af9d-75efedea10ac" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
