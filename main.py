import numpy as np
import pandas as pd
import datetime, dateutil.relativedelta, time
import json, random, operator, itertools
import sklearn.model_selection as skm
import yfinance as yf

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
yf.pdr_override()

class State:
    def __init__(self, name, shares_owned, returns):
        self.name = name
        #only 7 day returns for now
        self.shares = shares_owned
        self.returns = returns


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
    store_spy_data()
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
    covariance = df.cov()
    variance = np.var(stock_closing_pct_chg)
    return covariance / variance

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

portfolio_value = 1000000
stocks = generate_random_stocks(3)  # 2 + spy
oldest_date = get_date_data_available_from(stocks)
prices = get_price_data_from_date(oldest_date, histories)
covs = calculate_covariances(prices)
chosen_stocks = [covs[x][0] for x in range(-2, 2)]
two_stock_combinations = list(itertools.combinations(chosen_stocks, 2))
train_comb,test_comb = skm.train_test_split(two_stock_combinations)

actions = np.array([0.25, 0.10, 0.05, 0.1, -0.25, -0.10, -0.05, -0.1])
state_list = {}
weights = {}

# Build states
for stock_combination in train_comb:
    for stock in stock_combination:
        states = []
        returns = pct_change(prices[stock])
        for x in range(len(returns)-7):
            sevendayreturn = returns[x:x+7]
            current_state = State(stock, 0, sevendayreturn)
            states.append(current_state)
        if stock not in state_list:
            state_list[stock] = states

agent = Sequential()
agent.add(Flatten(input_shape=(1,) + actions.shape))
print(agent)