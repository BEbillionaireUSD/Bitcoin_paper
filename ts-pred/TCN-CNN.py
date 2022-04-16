from tcn import TCN, tcn_full_summary
from keras.layers.convolutional import Convolution1D #这个是1纬卷积
from keras.layers import Dense, Dropout, Flatten #全联接层
from keras.models import Sequential
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab

batch_size = 64 
time_steps = 20 #这个表示用前多少天的数据预测第t天的，比如用t-20, t-19....t-1预测t，则这个值就是20
input_dim = 1
output_window = 1 #输出纬度，在这里也是1
epochs = 2
dropout_rate=0.2

def create_inout_sequences(input_data, tw): #处理输入输出
    seq, label = [], []
    L = len(input_data)
    for i in range(L-tw):
        seq.append(input_data[i:i+tw])
        label.append(input_data[i+tw])
    return seq, label

def get_data(end=3500, path='BTC_1h.csv'): #把输入输出数据分割成训练集和测试集
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

def plot_loss(test_result, truth, path='graph/CNN.png'): #这个是绘制图像用的
    # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']
    plt.rc('font',family='Times New Roman')
    plt.title('CNN Prediction Result', fontdict={'family' : 'Times New Roman', 'size': 16})
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
    plt.savefig(path, dpi=750, bbox_inches='tight')
    plt.close()

def call_back(): #这里我设置了一种机制防止过拟合，不必在意
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


(train_x, train_y), (test_x, test_y) = get_data(3500)#需要修改这里的值（根据你自己数据的长度），以及数据文件的路径
train_x, train_y, test_x, test_y = np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
train_x = train_x.reshape(-1, time_steps, 1)
test_x = test_x.reshape(-1, time_steps, 1)
train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)


#tcn_layer = TCN(input_shape=(time_steps, input_dim), dropout_rate=dropout_rate) 
#m = Sequential([tcn_layer, Dense(1)])
# tcn -> cnn

print(train_x.shape, train_y.shape)
filter_size = 1 # apply a convolution 1d of length 1
filter_num = 512 # with 512 output filters
cnn_layer = Convolution1D(filter_num, filter_size, input_shape=(time_steps, input_dim)) 
m = Sequential([cnn_layer,  Dropout(dropout_rate), Flatten(), Dense(1)])
m.compile(optimizer='adam', loss='mse')
#tcn_full_summary(m, expand_residual_blocks=False)

m.fit(train_x, train_y, epochs=epochs, validation_data = (test_x, test_y), batch_size=batch_size, callbacks=call_back())

pred = m.predict(test_x)
plot_loss(pred, test_y)






