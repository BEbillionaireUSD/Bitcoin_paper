import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)


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

    # convert our train data into a pytorch train tensor
    #train_tensor = torch.FloatTensor(train_data).view(-1)
    train_sequence = create_inout_sequences(train_data,input_window)
    train_sequence = train_sequence[:-output_window] #todo: fix hack? -> din't think this through, looks like the last n sequences are to short, so I just remove them. Hackety Hack.. 

    #test_data = torch.FloatTensor(test_data).view(-1) 
    test_data = create_inout_sequences(test_data,input_window)
    test_data = test_data[:-output_window] #todo: fix hack?

    return train_sequence.to(device),test_data.to(device)

def plot_loss(test_result, truth):
    # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']
    plt.rc('font',family='Times New Roman')
    plt.title('Transformer Prediction Result', fontdict={'family' : 'Times New Roman', 'size': 16})
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
    plt.savefig('graph/transformer-epoch%d.png'%epoch, dpi=750, bbox_inches='tight')
    plt.close()
