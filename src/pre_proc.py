import numpy as np
import pandas as pd
import glob

from sklearn.preprocessing import MinMaxScaler

def to_single_csv(path="/dltraining/datasets/Stocks/", 
                  n_days=1000,
                  f_type='.us.txt'):
    
    #only the last n lines are the most recent prices
    #we want aapl dates since that is what we are trying to predict
    df = pd.read_csv(path+"aapl"+f_type, 
                     usecols=[0])[-1:-n_days:-1]
    df = df[-1::-1] #date_range starts with older dates

    #create column with datetime object
    df['datetime'] = pd.to_datetime(df.stack(),
                            format='%Y-%m-%d').unstack()

    df = df.set_index('datetime')

    filenames = glob.glob(path +"*"+ f_type)
    count = 0
    #add files to dataframe
    for file in filenames:
        stock = pd.read_csv(file, usecols=[0,4])[-1:-n_days:-1]
        stock = stock[-1::-1]

        count = count + 1
        print(count," / 4195", end="\r")
        #stock ticker name fom file path
        ticker = file.split('/')[4].split('.')[0]
        
        #only keep the closing price column and Date range
        stock = stock.loc[:,['Date','Close']]

        #convert index to date range
        stock['Date'] = pd.to_datetime(stock['Date'], format='%Y-%m-%d')
        stock = stock.set_index('Date')
        
        #rename column to stock name
        stock.rename(columns={'Close':ticker}, inplace=True)

        #add to single dataframe
        df = df.join(stock)

    df = df.drop(['Date'], axis='columns')
    df = drop_too_many_nan(df)
    df = nan_fill(df)

    df.to_csv('/dltraining/datasets/single_file.csv')
    
def nan_fill(df):
    #first fill missing values with next available day value
    #next previous available day value any remaining
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    return df

def drop_too_many_nan(df):
    #if the number of missing values exceeds a certain number
    #drop the column
    drop_col = []
    n = len(df)/20

    for name, items in df.iteritems():
        if(items.isna().sum() > n):
            drop_col.append(name)

    return df.drop( drop_col, axis='columns')

def corr_with_aapl(df):
    #pearsons correlation between different companies
    #only interested in the highest correlated companies wioth apple

    print("correlation between columns")
    co = df.corr().loc['aapl'].sort_values(ascending=False)

    #keep 10% most correlated stocks
    n = int(len(df.columns)*9/10)
    drop_cols = co.tail(n).index.values

    return df.drop(drop_cols, axis='columns')

def preproc_main(df):
    #drop uncorrelated
    df = corr_with_aapl(df) 
    col_names = df.columns.values
    date_index = df.index.values
    
    n_features = len(col_names)
    n_seq = len(date_index)

    df = nan_fill(df)

    #scale 
    col_names = df.columns.values
    date_index = df.index.values
    scaler = MinMaxScaler(feature_range=[-1,1])
    scaled_data = scaler.fit_transform(df)
    
    #split data
    df_scaled = pd.DataFrame(scaled_data, columns=col_names, index=date_index)
    train, test = test_train_split(df_scaled)
    
    #t+1 for test_y and train_y
    train_x, train_y = x_y_split(train)
    test_x, test_y = x_y_split(test)
    
    #to np array
    train_x = train_x.values
    train_y = train_y.values
    
    test_x = test_x.values
    test_y = test_y.values
    
    #export to csv
    np.savetxt('/dltraining/datasets/train_x.csv', X=train_x, delimiter=',')
    np.savetxt('/dltraining/datasets/train_y.csv', X=train_y, delimiter=',')
    np.savetxt('/dltraining/datasets/test_x.csv', X=test_x, delimiter=',')
    np.savetxt('/dltraining/datasets/test_y.csv', X=test_y, delimiter=',')

def x_y_split(df):
    #split for neural network testing

    print("x-y split")
    df_y = df['aapl'].shift(1).dropna()
    df_x = df.drop(df.tail(1).index.values)
    return df_x, df_y

def test_train_split(df):
    #testing will be over 30 trading days

    print("test train split")
    test = df[-1:-32:-1]
    test = test.iloc[-1::-1]
    train = df.drop(df.tail(31).index, axis=0)

    return train, test

print("starting to single file")
#to_single_csv()

print("pre processing main")
n_days = 1000

data = pd.read_csv("/dltraining/datasets/single_file.csv", infer_datetime_format=True, index_col=[0])[-1:-n_days:-1]
data = data[-1::-1]

data = nan_fill(data)
data = data[-1::-1]

preproc_main(data)

print("done, please run Enc_Dec_LSTM.py")
