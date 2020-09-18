import os

import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    return x, y


def load_mnist_test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_test
    y = y_test
    x = np.divide(x, 255.)
    x = x.reshape((x.shape[0], -1))
    return x, y


def load_fashion():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    y_names = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
               5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
    return x, y, y_names


def load_har():
    x_train = pd.read_csv(
        'data/har/train/X_train.txt',
        sep=r'\s+',
        header=None)
    y_train = pd.read_csv('data/har/train/y_train.txt', header=None)
    x_test = pd.read_csv('data/har/test/X_test.txt', sep=r'\s+', header=None)
    y_test = pd.read_csv('data/har/test/y_test.txt', header=None)
    # print(x_train.head())
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    # labels start at 1 so..
    y = y - 1
    y = y.reshape((y.size,))
    # print(x.shape)
    # print(y.shape)
    # print(y)
    y_names = {0: 'Walking', 1: 'Upstairs', 2: 'Downstairs', 3: 'Sitting', 4: 'Standing', 5: 'Laying', }
    return x, y, y_names


def load_usps(data_path='data/usps'):
    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64')
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y


def load_pendigits(data_path='data/pendigits'):
    if not os.path.exists(data_path + '/pendigits.tra'):
        os.makedirs(data_path,  exist_ok=True)
        
        os.system(
            'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra -P %s' %
            data_path)
        os.system(
            'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes -P %s' %
            data_path)
        os.system(
                'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.names -P %s' %
            data_path)

    # load training data
    with open(data_path + '/pendigits.tra') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_train, labels_train = data[:, :-1], data[:, -1]

    # load testing data
    with open(data_path + '/pendigits.tes') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_test, labels_test = data[:, :-1], data[:, -1]

    x = np.concatenate((data_train, data_test)).astype('float32')
    y = np.concatenate((labels_train, labels_test))
    x /= 100.
    y = y.astype('int')
    return x, y

def load_tesla(data_path='data/TSLA.csv', window_size=2, period=None):
    # Read Data "TSLA.csv" and set "Date" as INDEX for the dataset
    data = pd.read_csv(data_path, index_col ="Date", parse_dates = True)
    if period == 0:
        data = data[(data.index>'2010-06-29') & (data.index<'2013-01-29')]
    if period == 1:
        data = data[(data.index>'2014-01-01') & (data.index<'2016-12-29')]
    if period == 2:
        data = data[(data.index>'2017-01-01') & (data.index<'2019-12-29')]
    if period == 3:
        data = data[(data.index>'2013-04-01') & (data.index<'2013-8-29')]
    if period == 4:
        data = data[(data.index>'2018-12-20') & (data.index<'2019-04-29')]
    if period == 5:
        data = data[(data.index>'2017-01-01') & (data.index<'2017-04-30')]

    x= np.array((data))

    scaler = MinMaxScaler()
    x_tran = scaler.fit_transform(x)
    
    ret_x =None
    labels = []
    trade_res = []
    for i in range(window_size,len(x)-window_size):
        trade_res.append(x[i][3]-x[i][0])
        if trade_res[i-window_size] > 0 :
            labels.append(1)
            # print(x_tran[i+window_size][0] , x_tran[i+window_size][3])
            # print("1", trade_res[i-window_size] )
        else:
            labels.append(0)
            # print("0", trade_res[i-window_size] )

        tmp = np.array([])
        for j in range(window_size):
            tmp = np.concatenate((tmp, [np.log(x[i+j][3] / x[i+j-window_size][0])]))

        # print(tmp, window_size)

        if ret_x is None:
            ret_x =np.array([tmp])  
        else: 
            ret_x = np.concatenate((ret_x, [tmp]))

    y = np.array((pd.DataFrame(labels)))

    y = y.reshape((y.size,))

    return ret_x, y, trade_res

def load_tesla_debug(data_path='data/TSLA.csv', window_size=2):
    # Read Data "TSLA.csv" and set "Date" as INDEX for the dataset
    ret_x_0, y_0, trade_res_0 = load_tesla(data_path='data/TSLA.csv', window_size=window_size, period=3)
    ret_x_1, y_1, trade_res_1 = load_tesla(data_path='data/TSLA.csv', window_size=window_size, period=4)
    ret_x_2, y_2, trade_res_2= load_tesla(data_path='data/TSLA.csv', window_size=window_size, period=5)
    print(ret_x_0.shape)
    print(np.concatenate((ret_x_0,ret_x_1,ret_x_2)).shape)
    return np.concatenate((ret_x_0,ret_x_1,ret_x_2)), np.concatenate((y_0,y_1,y_2)), np.concatenate((trade_res_0,trade_res_1,trade_res_2))


if __name__ == "__main__":
    # load_har()
    load_tesla()

