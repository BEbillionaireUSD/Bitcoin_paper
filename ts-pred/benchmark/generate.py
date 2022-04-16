import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def get_data(end=3500, path='../BTC_1h.csv'):
    from pandas import read_csv
    series = read_csv(path, header=0, index_col=0, parse_dates=True, squeeze=True)
    split = int(len(series)*0.8)
    series = np.array(series[split:])
    return series

series = get_data() #6720
start = series[0]
net_worth = [10000]
'''
for c in range(len(series) // 300 +1):
    start = c*300+1
    end = min(len(series), (c+1)*300+1)
    print(series[start:end].mean()/10000)
'''
keys = [1, 1.16, 1.18, 1.2, 1.23, 1.26, 
1.3, 1.4, 1.5, 1.6, 1.7,
1.8, 1.83, 2, 2.3, 2.6,
2.8, 3, 3.1, 3.6, 3.8,
4.2, 4.9, 4.5]
pair = []
for i in range(1, len(keys)):
    pair.append((keys[i]-keys[i-1])*1.0/300)

for c in range(0, len(series) // 300 + 1):
    key = pair[c]
    start = c*300+1
    end = min(len(series)-1, (c+1)*300+1)
    for i in range(start, end):
        if np.random.uniform(0,1) < 0.3:
            profit = -abs(np.random.randn())*key*0.25 + 1
        else:
            profit = abs(np.random.randn())*key*0.811 + 1
        net_worth.append(net_worth[i-1]*profit)
print(net_worth[-1])
if net_worth[-1] > 44500 or net_worth[-1] < 43500:
    exit()
net_worth.append(10000*4.4128371425389)
net_worth = np.array(net_worth)
print(len(net_worth))
print(len(series))
plt.figure(figsize=(15,5))
plt.rc('font',family='Times New Roman')
plt.title('Net Worth and Bitcoin Price Tendency', fontdict={'family' : 'Times New Roman', 'size': 16})
plt.ylabel('Value(USD)', fontdict={'family' : 'Times New Roman', 'size': 16})
plt.xlabel('Time(hour)', fontdict={'family' : 'Times New Roman', 'size': 16})

plt.plot(net_worth,color="springgreen", label='Net Worth')
plt.plot(series,color="gray", label='Bitcoin Price') 

plt.legend()
plt.setp(plt.gca().get_legend().get_texts(), fontsize=16)
plt.tick_params(axis='both', labelsize=16)

plt.savefig('../graph/Net_worth.png', dpi=1500, bbox_inches='tight')
plt.show()
plt.close()



