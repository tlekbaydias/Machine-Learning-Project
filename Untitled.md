```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = pd.read_csv('Population_of_Finland.csv')
df = df.drop(['Natural increase','Intermunicipal migration','Net migration','Marriages','Divorces','Total change'], axis=1)
df = df[df['Emigration from Finland'] != '.']
print(df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 77 entries, 196 to 272
    Data columns (total 6 columns):
     #   Column                   Non-Null Count  Dtype 
    ---  ------                   --------------  ----- 
     0   Year                     77 non-null     int64 
     1   Live births              77 non-null     int64 
     2   Deaths                   77 non-null     int64 
     3   Immigration to Finland   77 non-null     object
     4   Emigration from Finland  77 non-null     object
     5   Population               77 non-null     int64 
    dtypes: int64(4), object(2)
    memory usage: 4.2+ KB
    None



```python
df.plot('Year', 'Population').plot(figsize=(10, 7))
plt.title("Figure 1", fontsize=17)
plt.ylabel('Population', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()
```


    
![png](output_1_0.png)
    



```python
df.plot('Live births', 'Population').plot(figsize=(10, 7))
plt.title("Figure 2", fontsize=17)
plt.ylabel('Population', fontsize=14)
plt.xlabel('Live births', fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()
```


    
![png](output_2_0.png)
    



```python
df.plot('Immigration to Finland','Population').plot(figsize=(10, 7))
plt.title("Figure 3", fontsize=17)
plt.ylabel('Population', fontsize=14)
plt.xlabel('Immigration to Finland', fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()
```


    
![png](output_3_0.png)
    



```python
df.plot('Immigration to Finland', 'Year').plot(figsize=(10, 7))
plt.title("Figure 3", fontsize=17)
plt.ylabel('Year', fontsize=14)
plt.xlabel('Immigration to Finland', fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()
```


    
![png](output_4_0.png)
    



```python
df.plot('Emigration from Finland', 'Year').plot(figsize=(10, 7))
plt.title("Figure 3", fontsize=17)
plt.ylabel('Year', fontsize=14)
plt.xlabel('Emigration from Finland', fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()
```


    
![png](output_5_0.png)
    



```python
df.plot('Year', 'Deaths').plot(figsize=(10, 7))
plt.title("Figure 3", fontsize=17)
plt.ylabel('Year', fontsize=14)
plt.xlabel('Immigration to Finland', fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()
```


    
![png](output_6_0.png)
    



```python
X = df[["Live births","Deaths","Immigration to Finland","Emigration from Finland"]]
y = df['Population']
#X = StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state=0)
print(len(X_train),len(X_val), len(X_test))
```

    46 16 15


Implementing Linear Regression model:


```python
regr = LinearRegression()
regr.fit(X_train, y_train) #fitting a training set
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
y_pred = regr.predict(X_train)
tr_error = mean_squared_error(y_train, y_pred)
print('The training error is: ', tr_error)
```

    Intercept: 
     5805315.104866516
    Coefficients: 
     [-18.70358876   2.40672698  18.97536721  -5.08744718]
    The training error is:  14920337950.932384



```python
fig, ax = plt.subplots()
coordinate_x=[]
for i in range(1,47):
    coordinate_x.append(i)
ax.plot(coordinate_x, y_train)
ax.plot(coordinate_x, y_pred)
plt.title('Training error = {:.5}'.format(tr_error))
plt.show()
```


    
![png](output_10_0.png)
    



```python
regr = LinearRegression()
regr.fit(X_val, y_val) #fitting a validation set
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
y_pred = regr.predict(X_val)
tr_error = mean_squared_error(y_val, y_pred)
print('The training error is: ', tr_error)
```

    Intercept: 
     2773723.2086173566
    Coefficients: 
     [ -2.72291266  47.73484278  23.44061535 -21.46940913]
    The training error is:  12756124728.661667



```python
fig, ax = plt.subplots()
coordinate_x = []
for i in range(16):
    coordinate_x.append(i)
ax.plot(coordinate_x, y_val)
ax.plot(coordinate_x, y_pred)
plt.title('Validation error = {:.5}'.format(tr_error))
plt.show()
```


    
![png](output_12_0.png)
    


Implementing Polynomial Regression model:


```python
degrees = [1,2,3,4] #Polynomial degrees
tr_errors = []
for n in range(len(degrees)):
    poly = PolynomialFeatures(degree = degrees[n])
    X_train_poly = poly.fit_transform(X_train) #fitting a training set
    regr = LinearRegression(fit_intercept = False) 
    regr.fit(X_train_poly, y_train) 
    y_pred_train = regr.predict(X_train_poly) 
    tr_error = mean_squared_error(y_train,y_pred_train)
    tr_errors.append(tr_error)
    X_fit = np.linspace(0, 46, 46) 
    plt.figure(figsize=(8, 5))
    plt.plot(X_fit, y_pred_train)
    plt.scatter(X_fit, y_train, color="b", s=10)
    plt.title('Polynomial degree = {}\nTraining error = {:.5}'.format(degrees[n], tr_error))
plt.show()
print(tr_errors)
```


    
![png](output_14_0.png)
    



    
![png](output_14_1.png)
    



    
![png](output_14_2.png)
    



    
![png](output_14_3.png)
    


    [14920337950.932379, 3777242758.5993376, 593354659.0069556, 3.1390133313913757e-08]



```python
degrees = [1,2,3,4]
val_errors = []
for n in range(len(degrees)):
    poly = PolynomialFeatures(degree = degrees[n])
    X_val_poly = poly.fit_transform(X_val) #fitting a validation set
    regr = LinearRegression(fit_intercept = False) 
    regr.fit(X_val_poly, y_val) 
    y_pred_val = regr.predict(X_val_poly) 
    val_error = mean_squared_error(y_val, y_pred_val)
    val_errors.append(val_error)
    X_fit = np.linspace(0, 16, 16) 
    plt.figure(figsize=(8, 5))
    plt.plot(X_fit, y_pred_val)
    plt.scatter(X_fit, y_val, color="b", s=10)
    plt.title('Polynomial degree = {}\nValidation error = {:.5}'.format(degrees[n], val_error))
plt.show()
print(val_errors)
```


    
![png](output_15_0.png)
    



    
![png](output_15_1.png)
    



    
![png](output_15_2.png)
    



    
![png](output_15_3.png)
    


    [12756124728.661676, 87341325.65646063, 3.952393967665557e-14, 7.827072323607354e-15]



```python
degrees = [4]
test_errors = []
for n in range(len(degrees)):
    polyx = PolynomialFeatures(degree = degrees[n])
    X_test_poly = poly.fit_transform(X_test) #fitting a test set
    regr = LinearRegression(fit_intercept = False) 
    regr.fit(X_test_poly, y_test) 
    y_pred_test = regr.predict(X_test_poly) 
    test_error = mean_squared_error(y_test, y_pred_test)
    test_errors.append(test_error)
    X_fit = np.linspace(0, 15, 15) 
    plt.figure(figsize=(10, 5))
    plt.plot(X_fit, y_pred_test)
    plt.scatter(X_fit, y_test, color="b", s=10)
    plt.title('Polynomial degree = {}\nTesting error = {:.5}'.format(degrees[n], test_error))
plt.show()
print(test_errors)
```


    
![png](output_16_0.png)
    


    [1.101711314769697e-14]



```python
#poly_pred = regr2.predict(polyx.fit_transform([[2025]]))
#print(poly_pred)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred_test)
```




    1.0




```python

```
