import numpy as np
import pandas as pd
import sklearn as sk
import datetime
import json, random
import yfinance as yf
yf.pdr_override()

histories = {}

def store_stock_history(stock_name, data):
    histories[stock_name] = data

def generate_random_stocks(n):
    with open('constituents_json.json', 'r') as f:
        sp500 = json.loads(f.read());

    random_stocks = [random.choice(sp500)['Symbol'] for x in range(n)]
    #append spy as a "random" stock for future use
    random_stocks.append('SPY')
    return random_stocks

def get_date_data_available_from(random_stocks):
    oldest_date = datetime.date.min
    for stock in random_stocks:
        tcker = yf.Ticker(stock)
        history = tcker.history('max')
        store_stock_history(stock, history)
        beginning = history.index[0]
        if beginning > oldest_date:
            oldest_date = beginning
    print('Stocks all exist from: ' + str(oldest_date))
    return oldest_date

def get_price_data_from_date(oldest_date, histories):
    data = {}
    for stock in histories:
        history = histories[stock]
        subset = np.array(history[history.index > oldest_date])
        data[stock] = subset
    return data

def beta(index, stock):
    index_closing_pct_chg = index['Close'].pct_change()
    stock_closing_pct_chg = stock['Close'].pct_change()
    df = pd.DataFrame(stock_closing_pct_chg,index_closing_pct_chg)
    return df.cov()


histories = {}
random_stocks = generate_random_stocks(2)
date = get_date_data_available_from(random_stocks)
data = get_price_data_from_date(date, histories)
for stock in data:
    print(beta(data['SPY'], data[stock]))

#print(history[history.columns.tolist()[1]].tolist())
#data = yf.download('uber', '', '2019-01-01')


stocks = generate_random_stocks(20)
oldest_date = get_date_data_available_from(stocks)
prices = get_price_data_from_date(stocks, oldest_date, histories)