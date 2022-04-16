import numpy as np
def get_data(end=3500, path='../BTC_1h.csv'):
    from pandas import read_csv
    series = read_csv(path, header=0, index_col=0, parse_dates=True, squeeze=True)
    split = int(len(series)*0.8)
    series = np.array(series[split:])
    return series

def calculate(series):
    cash=10000
    u = 0.0125*4
    coin = 0
    r0 = 5
    for t in range(20, len(series)-1):
        avg_5 = series[t-5:t].mean()
        avg_20 = series[t-20:t].mean()
        price = series[t]
        if avg_5 > avg_20 and (avg_5-avg_20)/avg_20*100 > r0: #buy
            count = u*((avg_5-avg_20)/avg_20*100)
            if cash >= count * price:
                cash -= count * price
                coin += count
            else:
                coin += cash / price
                cash = 0
        elif avg_5 < avg_20 and (avg_20-avg_5)/avg_5*100 > r0: #sell
            count = u*((avg_20-avg_5)/avg_5*100)
            if coin >= count:
                coin -= count
                cash += count * price
            else:
                cash += coin * price
                coin = 0
    if coin > 0:
        cash += coin * series[-1]
    return cash
import math

def vma(series):
    u = 0.25
    n = 50
    cash = 10000
    coin = 0
    for t in range(n, len(series)):
        long = math.log(sum(series[t-50:t]))/n
        short = math.log(series[t])
        if short > long:
            print('Yes')
            if cash > series[t] * u :
                cash -= series[t] * u
                coin += u
            else:
                coin += cash / series[t]
                cash = 0
    if coin > 0:
        cash += coin * series[-1]
    return cash

if __name__ == '__main__':
    series = get_data()
    cash = vma(series)
    print(cash)
    



