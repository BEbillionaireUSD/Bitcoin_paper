from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from keras.optimizers import Adam

batch_size = 64
hidden_size = 50
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

    origin_test_y = scaler.inverse_transform(np.array(test_y).reshape(-1, 1))

    return (train_x, train_y), (test_x, test_y), (scaler, origin_test_y)

def plot_loss(test_result, truth):
    # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']
    plt.rc('font',family='Times New Roman')
    plt.title('LSTM Prediction Result', fontdict={'family' : 'Times New Roman', 'size': 16})
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
    plt.savefig('graph/LSTM.png', dpi=750, bbox_inches='tight')
    plt.close()

def call_back(filepath='lstm_model.hdf5'):
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
    
    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,mode='min')
    return [EarlyStop, Reduce, checkpoint]


(train_x, train_y), (test_x, test_y), (scaler, real) = get_data(3500)
train_x, train_y, test_x, test_y = np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
train_x = train_x.reshape(-1, time_steps, 1)
test_x = test_x.reshape(-1, time_steps, 1)
train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)

adam = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#model.add(LSTM(hidden_size, activation='relu',input_shape=(train_x.shape[1], train_x.shape[2]), dropout=dropout_rate))
#model.add(Dense(1, activation='linear'))

model = Sequential()
model.add(LSTM(units=hidden_size, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dropout(dropout_rate))
model.add(LSTM(units=hidden_size, return_sequences=True))
model.add(Dropout(dropout_rate))
model.add(LSTM(units=hidden_size, return_sequences=True))
model.add(Dropout(dropout_rate))
model.add(LSTM(units=hidden_size))
model.add(Dropout(dropout_rate))
model.add(Dense(units=1))
model.compile(optimizer=adam, loss='mse')
model.load_weights("lstm_model.hdf5")

#model.fit(train_x, train_y, validation_data = (test_x, test_y), epochs=epochs, batch_size=batch_size, callbacks=call_back())

pred = model.predict(test_x)
pred = scaler.inverse_transform(pred.reshape(-1, 1))
real = real.squeeze(1)
pred = pred.squeeze(1)
print(real)
print(pred)
cash = 10000
u = 0.25
coin = 0

for t in range(len(pred)-1):
    if pred[t+1] > real[t]: #buy
        count = u
        if cash >= count * real[t]:
            cash -= count * real[t]
            coin += count
        else:
            coin += cash / real[t]
            cash = 0
    else: #sell
        count = u
        if coin >= count:
            coin -= count
            cash += count * real[t]
        else:
            cash += coin * real[t]
            coin = 0
if coin > 0:
    cash += real[-1]*coin
print(coin)
print(cash)

cash = 10000
u = 0.25
coin = 0
# buy
for t in range(len(pred)-1):
    if pred[t+1] > real[t]: #buy
        if cash >= u * real[t]:
            cash -= u*real[t]
            coin += u
        else:
            coin += cash / real[t]
            cash = 0
            break
print(coin)
cash += real[-1]*coin 
print(cash)

cash = 0
u = 0.25
coin = 10000/real[0]

#sell
for t in range(1, len(pred)-1):
    if pred[t+1] < real[t]: #sell
        if coin >= u:
            coin -= u
            cash += u*real[t]
        else:
            cash += coin*real[t]
            coin = 0
            break
print(coin)
cash += real[-1]*coin 
print(cash)

#plot_loss(pred, test_y)







