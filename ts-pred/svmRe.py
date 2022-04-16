import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from sklearn.metrics import mean_squared_error

batch_size = 64
time_steps = 20
input_dim = 1 #feature
output_window = 1
epochs = 100
dropout_rate=0.2

def create_inout_sequences(input_data, tw):
    seq, label = [], []
    L = len(input_data)
    for i in range(L-tw):
        seq.append(input_data[i:i+tw])
        label.append(input_data[i+tw])
    return seq, label

def get_data(end=3500, path='BTC_1h.csv'):
    time        = np.arange(0, end, 0.1)
    amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))
    
    from sklearn.preprocessing import MinMaxScaler
    
    #loading weather data from a file
    from pandas import read_csv
    series = read_csv(path, header=0, index_col=0, parse_dates=True, squeeze=True)
    # looks like normalizing input values curtial for the model
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    #amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    sampels = int(len(series)*0.8)
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]

    train_x, train_y = create_inout_sequences(train_data,time_steps)
    train_x, train_y = train_x[:-output_window], train_y[:-output_window]

    #test_data = torch.FloatTensor(test_data).view(-1) 
    test_x, test_y = create_inout_sequences(test_data,time_steps)
    test_x, test_y = test_x[:-output_window], test_y[:-output_window]

    return (train_x, train_y), (test_x, test_y)


def plot_loss(test_result, truth):
    # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']
    plt.rc('font',family='Times New Roman')
    plt.title('SVM Prediction Result', fontdict={'family' : 'Times New Roman', 'size': 16})
    plt.ylabel('Normalized Price', fontdict={'family' : 'Times New Roman', 'size': 16})
    plt.xlabel('Time(hour)', fontdict={'family' : 'Times New Roman', 'size': 16})

    plt.plot(test_result,color="tomato", label='Prediction')
    plt.plot(truth,color="deepskyblue", label='True Data') 

    #show legend
    plt.legend()
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=16)
    plt.tick_params(axis='both', labelsize=16)

    #plt.grid(True, which='both')
    #plt.axhline(y=0, color='k')
    plt.savefig('graph/SVM.png', dpi=750, bbox_inches='tight')
    plt.close()

def call_back():
    from keras.callbacks import ReduceLROnPlateau
    Reduce=ReduceLROnPlateau(monitor='val_loss',
                         factor=0.2,
                         patience=5,
                         verbose=1,
                         mode='auto',
                         epsilon=0.0001,
                         cooldown=0,
                         min_lr=0)

    from keras.callbacks import EarlyStopping
    EarlyStop=EarlyStopping(monitor='val_loss',
                        patience=10,verbose=1, mode='auto')
    return [EarlyStop, Reduce]


(train_x, train_y), (test_x, test_y) = get_data(3500)
train_x, train_y, test_x, test_y = np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
'''
train_x = train_x.reshape(-1, time_steps, 1)
test_x = test_x.reshape(-1, time_steps, 1)
train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)
'''

from sklearn import svm
import time
t0 = time.time()
regre=svm.SVR(kernel='linear',verbose=True)
regre.fit(train_x,train_y)
print(time.time()-t0)
exit()

pred = regre.predict(test_x)
print('MSE:', mean_squared_error(test_y, pred))
plot_loss(pred, test_y)






