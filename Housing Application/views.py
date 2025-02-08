from django.shortcuts import render

# Create your views here.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("G:/PROJECT CODES/PYTHON CODES/5 PROJECTS/5 PROJECTS\Estimating the price of houses using machine learning/PROJECT/housing/housingapp/kc_house_data.csv")
X = data.iloc[:, 3:].values
y = data.iloc[:, 2].values

#Splitting the dataset into training a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

# #Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit(X_test)

def home(request):
    return render(request,'home.html')

#Algorithm implementation
#Linear Regression
def linear(request):
    regressor_LR = LinearRegression()
    regressor_LR.fit(X_train,y_train)
    y_pred_lin = regressor_LR.predict(X_test)

    SS_Residual = sum((y_test - y_pred_lin) ** 2)
    SS_Total = sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

    plt.plot(y_test)
    plt.plot(y_pred_lin)
    plt.xlabel('count')
    plt.ylabel('TRUE AND PREDICTED VALUES ')
    plt.legend()
    plt.show()

    dic_lin = {'s':r_squared, 's1':adjusted_r_squared}
    return render(request,'linear.html',dic_lin)

def svr(request):
    regressor_svr = SVR()
    regressor_svr.fit(X_train,y_train)
    y_pred_svr = regressor_svr.predict(X_test)

    SS_Residual = sum((y_test - y_pred_svr) ** 2)
    SS_Total = sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

    plt.plot(y_test)
    plt.plot(y_pred_svr)
    plt.xlabel('count')
    plt.ylabel('TRUE AND PREDICTED VALUES ')
    plt.legend()
    plt.show()

    dic_svr = {'s':r_squared, 's1':adjusted_r_squared}
    return render(request,'svr.html',dic_svr)

def decision(request):
    regressor_DT = DecisionTreeRegressor()
    regressor_DT.fit(X_train,y_train)
    y_pred_DT = regressor_DT.predict(X_test)

    SS_Residual = sum((y_test - y_pred_DT) ** 2)
    SS_Total = sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

    plt.plot(y_test)
    plt.plot(y_pred_DT)
    plt.xlabel('count')
    plt.ylabel('TRUE AND PREDICTED VALUES ')
    plt.legend()
    plt.show()

    dic_DT = {'s':r_squared, 's1':adjusted_r_squared}
    return render(request, 'decision.html', dic_DT)

def random(request):
    regressor_RF = RandomForestRegressor(n_estimators=500)
    regressor_RF.fit(X_train,y_train)
    y_pred_RF = regressor_RF.predict(X_test)

    SS_Residual = sum((y_test - y_pred_RF) ** 2)
    SS_Total = sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

    plt.plot(y_test)
    plt.plot(y_pred_RF)
    plt.xlabel('count')
    plt.ylabel('TRUE AND PREDICTED VALUES ')
    plt.legend()
    plt.show()

    dic_DT = {'s':r_squared, 's1':adjusted_r_squared}
    return render(request,'random.html',dic_DT)



