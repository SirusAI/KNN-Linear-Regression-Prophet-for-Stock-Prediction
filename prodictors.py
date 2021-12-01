import pandas as pd
import datetime
import pandas_datareader.data as web
from prophet import Prophet
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.today().strftime('%Y-%m-%d')

class Prodictors:
    def __init__ (self, ticker):

        ticker = ticker.upper()
        
        self.symbol = ticker #for future fuction
        
        try:
            stock = web.DataReader(ticker, 'yahoo', start, end)
        except Exception as e:
            print('Error Retrieving Data.')
            print (e)
            return
        df_a = stock.reset_index(col_level=0)
        df_a.rename(columns={'Date':'ds',"Adj Close":'y'}, inplace=True)
        df_a = df_a[['ds','y']]
        self.stock = df_a.copy()
        self.dfg = stock.loc[:,['Adj Close','Volume']]
        self.dfg['HL_PCT'] = (stock['High'] - stock['Low']) / stock['Close'] * 100.0
        self.dfg['PCT_change'] = (stock['Close'] - stock['Open']) / stock['Open'] * 100.0

        self.dfg.fillna(value=-99999, inplace=True)
        # separate 1 percent of the data to forecast
        forecast_data = int(math.ceil(0.01 * len(self.dfg)))
        # Separating the label, predict 'Adj Close'
        forecast_obj = 'Adj Close'
        self.dfg['label'] = self.dfg[forecast_obj].shift(-forecast_data)
        X = np.array(self.dfg.drop(['label'], 1))
        X = preprocessing.scale(X)
        X_lately = X[-forecast_data:]
        X = X[:-forecast_data]
        y = np.array(self.dfg['label'])
        y = y[:-forecast_data]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
       
        # Linear regression
        clfg = LinearRegression(n_jobs=-1)
        clfg.fit(X_train, y_train)

        # KNN Regression
        clfknn = KNeighborsRegressor(n_neighbors=2)
        clfknn.fit(X_train, y_train)
        
        #Quadratic Regression PolynomiaFeatures = 2
        clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
        clfpoly2.fit(X_train, y_train)
        
        #Quadratic Regression PolynomiaFeatures = 3
        clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
        clfpoly3.fit(X_train, y_train)
        
        #Linear confidence
        self.confidencereg = clfg.score(X_test, y_test)
        #KNN confidence
        self.confidenceknn = clfknn.score(X_test, y_test)
        #Quadratic Regression confidence PolynomiaFeatures = 2
        confidencepoly2 = clfpoly2.score(X_test,y_test)
        #Quadratic Regression confidence PolynomiaFeatures = 3
        confidencepoly3 = clfpoly3.score(X_test,y_test)

        self.forecast_set_reg = clfg.predict(X_lately)
        self.forecast_set_knn = clfknn.predict(X_lately)
        self.forecast_set_poly2 = clfpoly2.predict(X_lately)
        self.forecast_set_poly3 = clfpoly3.predict(X_lately)
        self.dfg['Forecast'] = np.nan
        last_date = self.dfg.iloc[-1].name
        last_unix = last_date
        self.next_unix = last_unix + datetime.timedelta(days=1)
    
    #Price prediction for the next year by using Prophet
    def prophet_predict(self,days):
        
        p = Prophet(daily_seasonality=False)
        
        p.fit(self.stock)
        
        future = p.make_future_dataframe(periods = days)
        
        forecast = p.predict(future)
        
        fig_a = p.plot(forecast,xlabel = self.symbol,ylabel = 'Price')       
    
    #Analyze seasonal trend
    def trend_analizer(self,days):
        
        p = Prophet(daily_seasonality=False)
        
        p.fit(self.stock)
        
        future = p.make_future_dataframe(periods = days)
        
        forecast = p.predict(future)

        fig_b = p.plot_components(forecast)
        
    #customize price prodiction for upcoming n days by using Prophet method
    def prophet_prediction(self,days): 
        
        p = Prophet(daily_seasonality=False)
        
        p.fit(self.stock)
        
        future = p.make_future_dataframe(periods = days)
        
        forecast = p.predict(future)
        
        prediction = forecast.iloc[-days:,0:4]
        
        prediction.rename(columns={'ds':'Date',"trend":'Prediction',\
                                   'yhat_lower':'Low','yhat_upper':'High'},\
                          inplace=True)
        
        table = prediction
        
#         plt.plot(list(range(days+1)),forecast.iloc[-days-1:,1])       
#         plt.title ("Price Prediction for the next {} days".format(days),\
#                    fontsize = 18)    
#         plt.xlabel ("Date",\
#                     fontsize = 14)
#         plt.ylabel ("Price",\
#                     fontsize = 14)              
#         plt.show()
        
        return display(table)
    
    #Predicting price for the next year by applying Linear regression method
    def linear_prediction(self): 
        plt.figure(figsize=(20,10),dpi = 75)
        for i in self.forecast_set_reg:
            next_date = self.next_unix
            self.next_unix += datetime.timedelta(days=1)
            self.dfg.loc[next_date] = [np.nan for _ in range(len(self.dfg.columns)-1)]+[i]
        
        self.dfg['Adj Close'].tail(500).plot()
        self.dfg['Forecast'].tail(500).plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        print("Linear Regression Prediction | ","Ticker:",self.symbol.upper(),'|',\
              'Linear Regression Confidence: ',self.confidencereg)
        plt.figure(figsize=(20,10))
        return plt.show()
    
    #Predicting price for the next year by appling KNN method.
    def knn_prediction(self):
        plt.figure(figsize=(20,10),dpi = 75)
        for i in self.forecast_set_knn:
            next_date = self.next_unix
            self.next_unix += datetime.timedelta(days=1)
            self.dfg.loc[next_date] = [np.nan for _ in range(len(self.dfg.columns)-1)]+[i]
            
        self.dfg['Adj Close'].tail(500).plot()
        self.dfg['Forecast'].tail(500).plot()
        
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        print("KNN Prediction | ","Ticker:",self.symbol.upper(),'| ',\
              'KNN Confidence:',self.confidenceknn)
        return plt.show()
    
    #Predicting price for the next year by applying Quadratic regression PolynomiaFeatures = 2
    def quad_regression2(self):
        plt.figure(figsize=(20,10),dpi = 75)
        for i in self.forecast_set_poly2:
            next_date = self.next_unix
            self.next_unix += datetime.timedelta(days=1)
            self.dfg.loc[next_date] = [np.nan for _ in range(len(self.dfg.columns)-1)]+[i]
        self.dfg['Adj Close'].tail(500).plot()
        self.dfg['Forecast'].tail(500).plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        print("Quadratic Regression 2 Prediction | ","Ticker",self.symbol.upper(),'| ',\
              'Quad_R2 Confidence:',self.confidenceknn)
        plt.show()
        
    #Predicting price for the next year by applying Quadratic regression PolynomiaFeatures = 3
    def quad_regression3(self):
        plt.figure(figsize=(20,10),dpi = 75)
        for i in self.forecast_set_poly3:
            next_date = self.next_unix
            self.next_unix += datetime.timedelta(days=1)
            self.dfg.loc[next_date] = [np.nan for _ in range(len(self.dfg.columns)-1)]+[i]
        self.dfg['Adj Close'].tail(500).plot()
        self.dfg['Forecast'].tail(500).plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        print("Quadratic Regression 3 Prediction | ","Ticker",self.symbol.upper(),'| ',\
              'Quad_R3 Confidence:',self.confidenceknn)
        plt.show()    