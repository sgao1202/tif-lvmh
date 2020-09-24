import functools
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

FILE = 'tif.csv'

def sum(a, b):
    return a + b;

def main():
    # Create a Data Frame from reading in FILE
    df = pd.read_csv(FILE, parse_dates=True, index_col=0)

    # Create the Daily Return column from raw closing prices
    df['Daily Return'] = df['Close'].pct_change()

    # List of tuples that represent date intervals 
    dates = []
    dates.append(tuple((dt.date(2019, 11, 26), dt.date(2020, 3, 6))))
    dates.append(tuple((dt.date(2020, 3, 10), dt.date(2020, 9, 8))))
    dates.append(tuple((dt.date(2020, 9, 10), dt.date(2020, 9, 18))))

    # Calculate weighted daily average return
    intervalReturns = []
    weights = [0, 0.4, 0.6]
    for i in range(3):
        currentDf = df.loc[dates[i][0]: dates[i][1]]
        currentReturns = currentDf['Daily Return']
        intervalReturns.append(functools.reduce(sum, currentReturns) / len(currentReturns) * weights[i])

    # Find the average daily return with the weights factored in
    mu = functools.reduce(sum, intervalReturns)
    def compute(row):
        return (row['Daily Return'] - mu) ** 2

    # Create column in data frame for (Daily Return - Average Daily Return)^2
    df['Variance Num'] = df.apply(compute, axis=1)
    n = len(df.index)
    var = df.sum(axis=0)['Variance Num'] / (n - 1)
    sigma = math.sqrt(var)

    # Metrics from our data set
    print('Mu: {}'.format(mu))
    print('Var: {}'.format(var))
    print('Sigma: {}'.format(sigma))

    # df['Time Delta'] = list(map(lambda d: (targetDate - d).days, df.index.date))
    # print(df['Time Delta'])
    def computePrediction(currentPrice):
        return currentPrice * math.exp((mu - (var / 2) * 1 + sigma * np.random.normal(0,1)))
    
    def simulate(count):
        targetPrices = []
        # November 1st, 2020 is our prediction date but increment by one to fix the loop
        targetDate = dt.date(2020, 11, 1)
        # Run through 1000 samples of stock prices
        for i in range(count):
            currentDate = dt.date(2020, 9, 18)
            # Initial price is the last day in our data set
            currentPrice = df['Close'][-1]
            while currentDate != targetDate:
                day = currentDate.weekday()
                # Do not trade on weekends
                if day != 5 and day != 6:
                    currentPrice = computePrediction(currentPrice)
                # Increment day by 1
                currentDate += dt.timedelta(days=1)
            # When startDate == targetDate, the targetDate prediction will not be made so therefore we should include it here
            currentPrice = computePrediction(currentPrice)
            targetPrices.append(currentPrice)
        # Return the average target price from our predictions
        return functools.reduce(sum, targetPrices) / len(targetPrices)

    print(simulate(1000))
    print(simulate(10000))
    print(simulate(100000))

if __name__ == "__main__":
    main()