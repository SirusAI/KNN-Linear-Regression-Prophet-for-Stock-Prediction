# KNN-Linear-Regression-Prophet-for-Stock-Price-Prediction
KNN + Linear Regression + Prophet Prediction for Stocks, and Prophet base seasonal trend analysis. 
I developed this tool mainly to gain more experience in time series analysis and object-oriented programming. I will occasionally update this project by adding more fuctions. Please do not use it for making any investment or trading decision. 
# Requirements
* [Pandas](https://pandas.pydata.org)
* [Pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest)
* [Datetime](https://docs.python.org/3/library/datetime.html)
* [Prophet](https://facebook.github.io/prophet/docs/installation.html#installation-in-python)
* [Matplotlib](https://matplotlib.org/stable/users/index.html)
* [Sklearn](https://scikit-learn.org/stable/user_guide.html)
* [Numpy](https://numpy.org/doc/stable/)
# Methods
The tool (Prodictors) includes 4 main methods for predicting analyzing stock prices. 
# Examples
In this case, I will use IWM Which is the Russell 2000 ETF and see the price prediction by applying linear-regression / KNN / Prophet. 
```
from prodictors import Prodictors #import Prodictors
ticker = 'IWM'
test_ticker = Prodictors(ticker)
```

## Linear Regresion Stock Price Prediction
```
test_ticker.linear_prediction() 
```
![](images/Linear Regression Prediction.png)

## KNN Stock Price Prediction
```
test_ticker.knn_prediction()
```
![](images/KNN Prediction.png)

## Prophet Stock Price Prediction
```
test_ticker.prophet_predict(days = 365)
```
![](images/Prophet Prediction.png)

## Prophet Seasonal trend
```
test_ticker.trend_analizer(days = 365)
```
![](Trend Analizer.png)

