import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
import pickle
import statsmodels.api as sm

def standard_deviation(data):
    return np.std(data)

def variance(data):
    return np.var(data)

def mean(data):
    return np.mean(data)

def median(data):
    return np.median(data)

def floor(data):
    return list(np.floor(data))

def ceil(data):
    return list(np.ceil(data))

def fix(data):
    return list(np.fix(data))

def nanprod(data):
    return np.nanprod(data)

def nansum(data):
    return np.nansum(data)

def nancumsum(data):
    return list(np.nancumsum(data))

def exp(data):
    return list(np.exp(data))

def expm1(data):#Calculate exp(x) - 1 for all elements in the array.
    return list(np.expm1(data))

def exp2(data):
    return list(np.exp2(data))

def log(data):# base e.
    return list(np.log(data))

def log10(data):
    return list(np.log10(data))

def log2(data):
    return list(np.log2(data))

def interquartile_range(data):
    Q1 = np.percentile(data, 25, interpolation='midpoint')
    Q3 = np.percentile(data, 75, interpolation='midpoint')
    diff=Q3-Q1
    return diff


def Q1(data):
    Q1 = np.percentile(data, 25, interpolation='midpoint')
    return Q1

def Q3(data):
    Q3 = np.percentile(data, 75, interpolation='midpoint')
    return Q3

def OutlierLimits(colValues, pMul=3):
    """
    returns:
        upper boud & lower bound for array values or df[col]
    usage:
        OutlierLimits(df[col]):
    """
    if (pMul != 3 and pMul != 2.5 and pMul != 2 and pMul != 1.5):
        pMul = 3
    pMul = float(pMul)
    quartile_1, quartile_3 = np.percentile(colValues, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * pMul)
    upper_bound = quartile_3 + (iqr * pMul)
    return lower_bound, upper_bound

# get multi colinearity columns

def find_outliers_IQR(l1):
    l1=pd.DataFrame({'col':l1})
    Q1 = np.percentile(l1, 25, interpolation='midpoint')
    Q3 = np.percentile(l1, 75, interpolation='midpoint')

    IQR=Q3-Q1

    outliers = l1[((l1<(Q1-1.5*IQR)) | (l1>(Q3+1.5*IQR)))]
    outliers.dropna(inplace=True)
    outliers=outliers.values

    return list(outliers.flatten())



def Y_intercept(x,y):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
        lm = LinearRegression()
        lm.fit(x_train, y_train)

        return lm.intercept_

def Slope(x, y):

    lm = LinearRegression()
    lm.fit(x, y)

    return lm.coef_


def box(l1):
    l1=pd.DataFrame({'col':l1})
    Q1 = np.percentile(l1, 25, interpolation='midpoint')
    Q3 = np.percentile(l1, 75, interpolation='midpoint')

    IQR=Q3-Q1

    outliers = l1[((l1<(Q1-1.5*IQR)) | (l1>(Q3+1.5*IQR)))]
    outliers.dropna(inplace=True)
    outliers=outliers.values


    lower_bound = Q1 - (IQR * 1.5)
    upper_bound = Q3 + (IQR * 1.5)
    return [Q1,Q3,list(outliers.flatten()),IQR,lower_bound,upper_bound]


def LabelEncoding(data,cols):
    df=pd.DataFrame(data[cols])
    df=df.fillna('')
    le=LabelEncoder()
    for i in cols:
     df[i]=le.fit_transform(df[i])
    return df


def Normalize(data):
    df=pd.DataFrame(data)
    model=MinMaxScaler()
    output1=list(model.fit_transform(df))
    return output1


def Normalize1(data,cols):

    cols_norm=[]
    for i in cols:
        cols_norm.append('Norm_' + i)
    cols_norm
    data=pd.DataFrame(data)
    df=pd.DataFrame(data[list(cols)])
    model=MinMaxScaler()
    output1=model.fit_transform(df)

    a=output1
    b=pd.DataFrame(a,columns=cols_norm)
    addons=list(b.columns)
    j=pd.concat([data,b],axis=1)
    j=j.fillna('')
    return j.to_dict('list')





def Replace_Outliers(l1):
    l1 = pd.DataFrame({'col': l1})
    Q1 = np.percentile(l1, 25, method='midpoint')
    Q3 = np.percentile(l1, 75, method='midpoint')

    IQR = Q3 - Q1
    df2 = pd.DataFrame(l1)
    outliers = l1[((l1 < (Q1 - 1.5 * IQR)) | (l1 > (Q3 + 1.5 * IQR)))]
    outliers.dropna(inplace=True, axis=0)
    # df=outliers

    outliers = list(outliers.values)
    rep = np.median(l1)
    df2[df2.columns[0]] = df2[df2.columns[0]].replace(outliers, rep)

    #   return list(outliers.flatten())
    return list(df2.values)

print(Replace_Outliers([27.1, 27.2, 26.9, 30, 22, 28, 25, 23, 27, 25, 101, 0]))
def Kmeans_cluster(X,Y):

  # Loop through a range of possible k values
  silhouette=[]
  X1=X
  for i in range(1,11):
      k=KMeans(n_clusters=i)
      k.fit_predict(X)
      silhouette.append(k.inertia_)


  # Find the index of the highest silhouette score
  best_k = silhouette.index(max(silhouette)) + 2


  kmeans = KMeans(n_clusters=best_k)
  kmeans.fit(X)
  class_col = list(kmeans.labels_)
  df = pd.DataFrame()
  df['Class'] = class_col
  df['X1'] = X1
  df['Y']=Y
  return df


def create_dataset(data, look_back=1):
    data_X, data_y = [], []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back), :]
        data_X.append(a)
        data_y.append(data[i + look_back, :])
    return np.array(data_X), np.array(data_y)


def forecasting(data,col,datecol):
  data=pd.DataFrame(data)
  data[datecol] = pd.to_datetime(data[datecol])
#  data[datecol]=data[datecol].astype(str)
  data=data.groupby([datecol]).sum([col]).reset_index()
  tempdate=list(data[datecol].values)
  datadup=data
  data.drop(columns=datecol,inplace=True)
  # Scale the data using MinMaxScaler
  scaler = MinMaxScaler(feature_range=(0, 1))
  data_scaled = scaler.fit_transform(data)
  train_size = int(len(data_scaled) * 0.8)
  test_size = len(data_scaled) - train_size
  train, test = data_scaled[0:train_size, :], data_scaled[train_size:len(data_scaled), :]
  look_back = 1
  train_X, train_y = create_dataset(train, look_back)
  test_X, test_y = create_dataset(test, look_back)
  train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
  test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
  model = Sequential()
  model.add(LSTM(50, input_shape=(1, look_back)))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(train_X, train_y, epochs=100, batch_size=1, verbose=0)
  # Make predictions on the test data
  predictions = model.predict(test_X)
  predictions = scaler.inverse_transform(predictions)
  test_y = scaler.inverse_transform(test_y)
  tomorrow_sales = model.predict(np.array([test_X[-1]]))
  tomorrow_sales = scaler.inverse_transform(tomorrow_sales)
  nextdate=[]
  datadup[datecol]=tempdate
  for i in range(0,len(tomorrow_sales)):
    nextdate.append(max(datadup[datecol]) +pd.DateOffset(i+1))
  tempdf=pd.DataFrame()
  tempdf[datecol]=nextdate
  tempdf[col]=tomorrow_sales
  tempdf=pd.concat([datadup[[datecol,col]],tempdf],ignore_index=True)
  tempdf[datecol]=tempdf[datecol].astype(str)

  return tempdf



def model_forecasting(data,col,datecol):
  # Loading the dataset

  dataset=pd.DataFrame(data)
  dataset=dataset.groupby([datecol]).sum([col]).reset_index()
  # Preprocessing the Data
  dataset = dataset.dropna()

  dataset = dataset[[datecol, col]]
  dataset[datecol] = pd.to_datetime(dataset[datecol])
  df=dataset.set_index(datecol)

  train = df.iloc[:round(len(df)*0.8)]
  test = df.iloc[round(len(df)*0.8):]
  scaler = MinMaxScaler()
  scaler.fit(train)
  scaled_train = scaler.transform(train)
  scaled_test = scaler.transform(test)
  # define generator
  n_input = 3
  n_features = 1
  generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
  X,y = generator[0]
  # We do the same thing, but now instead for 12 months
  n_input = 12
  generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

  # define model
  model = Sequential()
  model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')

  # fit model
  model.fit(generator,epochs=50)
  pickle.dump(model, open('model.pkl', 'wb'))
  return 'Model Trained...!!!'


#def forecasting1(data, col, datecol,days):
def forecasting1(data, col, datecol):
    # Loading the dataset

    dataset = pd.DataFrame(data)
    dataset = dataset.groupby([datecol]).sum([col]).reset_index()
    # Preprocessing the Data
    dataset = dataset.dropna()

    dataset = dataset[[datecol, col]]
    dataset[datecol] = pd.to_datetime(dataset[datecol])
    df = dataset.set_index(datecol)

    train = df.iloc[:round(len(df) * 0.8)]
    test = df.iloc[round(len(df) * 0.8):]
    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)
    # define generator
    n_input = 3
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
    X, y = generator[0]
    # We do the same thing, but now instead for 12 months
    n_input = 12
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

    '''# define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # fit model
    model.fit(generator,epochs=50)'''

    model = pickle.load(open('model.pkl', 'rb'))
    # predictions=pickled_model.predict(test_X)

    # loss_per_epoch = model.history.history['loss']
    last_train_batch = scaled_train[-12:]
    last_train_batch = last_train_batch.reshape((1, n_input, n_features))
    test_predictions = []

    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))

    for i in range(len(test) + 30):
        # get the prediction value for the first batch
        current_pred = model.predict(current_batch)[0]

        # append the prediction into the array
        test_predictions.append(current_pred)

        # use the prediction to update the batch and remove the first value
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    true_predictions = scaler.inverse_transform(test_predictions)
    pred = scaler.inverse_transform(test_predictions[-30:])
    df1 = df.reset_index()
    nextdate = []
    for i in range(0, len(pred)):
        nextdate.append(max(df1[datecol]) + pd.DateOffset(i + 1))
    tempdf = pd.DataFrame()
    tempdf[datecol] = nextdate
    tempdf[col] = pred
    tempdf = pd.concat([df1[[datecol, col]], tempdf], ignore_index=True)
    tempdf[datecol] = tempdf[datecol]
    return tempdf


def seasonality(data,datecol,col):
  data=pd.DataFrame(data)
  data[datecol] = pd.to_datetime(data[datecol])
  data=data.groupby([datecol]).sum([col])
 # data[datecol] = pd.to_datetime(data[datecol])
 # data.set_index(datecol, inplace=True)
  data[col]=Replace_Outliers(list(data[col]))
  sales_data_monthly = data.resample('q').sum()
  decomposition = sm.tsa.seasonal_decompose(sales_data_monthly, model='additive')
  obs=pd.DataFrame(decomposition.observed).reset_index()
  season=pd.DataFrame(decomposition.seasonal).reset_index()
  res=pd.DataFrame(decomposition.resid).reset_index()
  output=pd.DataFrame()
  output[datecol]=list(res[datecol].values)
  output['obs']=list(obs[0].values)
  output['seasonality']=list(season['seasonal'].values)
  output['residual']=list(res['resid'].values)
  output.fillna('',inplace=True)


  return output



#22100482

