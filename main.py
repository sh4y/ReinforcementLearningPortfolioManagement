import numpy as np
import pandas as pd
import sklearn as sk
import datetime, dateutil.relativedelta, time
import json, random, operator
import yfinance as yf
yf.pdr_override()

with open('constituents_json.json', 'r') as f:
    sp500 = json.loads(f.read());
histories = {}

def store_spy_data():
    spy = yf.Ticker('SPY')
    index_data = spy.history('max')
    histories['SPY'] = index_data

def store_stock_history(stock_name, data):
    if len(data) != 0:
        print('stored data for ' + stock_name)
        histories[stock_name] = data
    else:
        print('Data is empty for stock ' + stock_name)

def generate_random_stocks(n):
    random_stocks = [random.choice(sp500)['Symbol'] for x in range(n)]
    #append spy as a "random" stock for future use
    return random_stocks

def get_date_data_available_from(stocks):
    oldest_date = datetime.date.min
    print(stocks)
    for stock in stocks:
        try:
            print('Getting data for stock: ' + stock)
            tcker = yf.Ticker(stock)
            history = tcker.history('max')
            beginning = history.index[0]
            difference_in_years = dateutil.relativedelta.relativedelta(datetime.date.today(), beginning).years
            if difference_in_years > 5:
                print(stock + ' exists. Getting data.')
            else:
                print(stock + ' does not exist before 5 years from now. Removing.')
                stocks.remove(stock)
                continue
            if beginning > oldest_date:
                oldest_date = beginning
            store_stock_history(stock, history)
        except ValueError as error:
            print('Value Error found for stock ', stock)
            print('Removing stock ' + str(stock) + ' from list, adding a new one.')
            stocks.append(random.choice(sp500)['Symbol'])
            if 'delisted' in str(error):
                continue
        except Exception as error:
            raise error
    print('Stocks all exist from: ' + str(oldest_date))
    return oldest_date


def get_price_data_from_date(oldest_date, histories):
    data = {}
    for stock in histories:
        history = histories[stock]
        subset = history[history.index > oldest_date]
        data[stock] = subset
    print(data.keys())
    return data

def pct_change(a):
    a = np.array(a)
    return np.diff(a) / a[:, 1:]

def beta(index, stock):
    index_closing_pct_chg = index['Close'].pct_change()
    stock_closing_pct_chg = stock['Close'].pct_change()
    d = {'Index Changes': index_closing_pct_chg, 'Stock Change': stock_closing_pct_chg}
    df = pd.DataFrame(d)
    return df.cov()

def calculate_covariances(prices):
    covars = {}
    for stock in prices:
        try:
            # calculate covar between stock and index
            if stock != 'SPY':
                cov = beta(prices['SPY'], prices[stock])
                covars[stock] = cov['Index Changes'][1]
        except KeyError as error:
            print('Got key error for: ' + str(stock))
            raise error
    sorted_x = sorted(covars.items(), key=operator.itemgetter(1))
    return sorted_x

stocks = generate_random_stocks(20)
store_spy_data()
oldest_date = get_date_data_available_from(stocks)
prices = get_price_data_from_date(oldest_date, histories)
betas = calculate_covariances(prices)
print(betas)