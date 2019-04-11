import numpy as np
import pandas as pd
import glob

from sklearn.preprocessing import MinMaxScaler

def to_single_csv(path="/dltraining/datasets/Stocks/", 
                  n_days=1000,
                  f_type='.us.txt'):

    #only the last n lines are the most recent prices
    df = pd.read_csv(path+"aapl"+f_type, 
                     usecols=[0])[-1:-n_days:-1]
    df = df[-1::-1] #date_range starts with older dates

    #create column with datetime object
    df['datetime'] = pd.to_datetime(df.stack(),
                            format='%Y-%m-%d').unstack()

    df = df.set_index('datetime')

    filenames = glob.glob(path +"*"+ f_type)
    #add files to dataframe
    for file in filenames:
        stock = pd.read_csv(path+file+f_type )[-1:-n_days:-1]
        stock = stock[-1::-1]

        #stock ticker name fom file path
        ticker = file.split('.')[0]
        
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

    df.to_csv("single_file.csv")
    
def nan_fill(df):
    #first fill missing values with next available day value
    #next previous available day value any remaining
    df.fillna(method='bfill')
    df.fillna(method='ffill')

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

print("Done")

def corr_with_aapl(df):
    #pearsons correlation between different companies
    #only interested in the highest correlated companies wioth apple
    co = df.corr().sort_values(by=['aapl'], ascending=False)
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
    #scale 
    scaler = MinMaxScaler(feature_range=[-1,1])
    scaled_data = scaler.fit_transform(df)
    
    #split data
    df_scaled = pd.DataFrame(scaled_data, columns=col_names, index=date_index)
    train, test = test_train_split(df_scaled)
    
    #t+1 test_y and train_y
    train_x, train_y = x_y_split(train)
    test_x, test_y = x_y_split(test)
    
    #to np array
    train_x = train_x.values
    train_y = train_y.values
    
    test_x = test_x.values
    test_y = test_y.values
    
    #export to csv
    np.savetxt('train_x.csv', X=train_x, delimiter=',')
    np.savetxt('train_y.csv', X=train_y, delimiter=',')
    np.savetxt('test_x.csv', X=test_x, delimiter=',')
    np.savetxt('test_y.csv', X=test_y, delimiter=',')

def x_y_split(df):
    #split for neural network testing
    df_y = df['aapl'].shift(1).dropna()
    df_x = df.drop(df.tail(1).index.values)
    return df_x, df_y

def test_train_split(df):
    #testing will be over 30 trading days
    test = df[-1:-32:-1]
    test = test.iloc[-1::-1]
    train = df.drop(df.tail(31).index, axis=0)

    return train, test

to_single_csv()

n_days = 1000
data = pd.read_csv("single_file.csv", index_col=0)[-1:-n_days:-1]
data = data[-1::-1]

preproc_main(data)
