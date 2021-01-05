#%%
import pandas as pd
import pandas_datareader.data as web
import backtrader as bt
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from concurrent import futures
import sys

#%%
TICKER_FILE = 'spy/tickers.csv' 
SANDP_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

#%%

## Grab the list of symbols from Wikipedia
dfs = pd.read_html(SANDP_URL)
df = dfs[0]
tickers = df['Symbol']

# Save it to file
try:
    pd.Series(tickers).to_csv(TICKER_FILE)
except Exception as e:
    sys.exit(f'Unable to create file for tickers in spy/ after creating list from Wikipedia\n{e}')

#%%

end = datetime.now()
start = datetime(end.year - 5, end.month, end.day)

# Open the ticker list
try:
    df = pd.read_csv(TICKER_FILE)
except Exception as e:
    sys.exit('Unable to create file for tickers in spy/\n{e}')

# Change the ticker names of BRK.B to BRK-B and BF.B to BF-B.
try:
    df['Symbol'] = df['Symbol'].replace('BRK.B', 'BRK-B')
    df['Symbol'] = df['Symbol'].replace('BF.B', 'BF-B')
except Exception as e:
    sys.exit(f'Unable to replace BRK.B or BF.B\n{e}')


tickers = df['Symbol']

# Download all of the ticker list files from Yahoo
def download(ticker):
    try:
        dfn = web.DataReader(ticker, 'yahoo', start, end)
        dfn.to_csv(f'spy/{ticker}.csv')
    except Exception as e:
        print(f'Ticker download error:  Unable to retrieve {ticker}.Removing from ticker list.\n{e}')
        tickers.drop(df[df['Symbol'] == ticker].index)

# with futures.ThreadPoolExecutor(50) as executor:
#     res = executor.map(download, tickers)

try:
    tickers.to_csv(TICKER_FILE)
except Exception as e:
    sys.exit(f'Unable to create file for tickers in spy/ after downloading all files.\n{e}')


#%%
class CrossSectionalMR(bt.Strategy):
    def prenext(self):
        self.next()

    def next(self):
        available = list(filter(lambda d: len(d), self.datas))

        rets = np.zeros((len(available)))
        for i, d in enumerate(available):
            # Calculate inidivdual daily returns
            rets[i] = (d.close[0]-d.close[-1] / d.close[-1])

        # Calculate weights using formula
        market_ret = np.mean(rets)
        weights = -(rets - market_ret)
        weights = weights / np.sum(np.abs(weights))

        for i, d in enumerate(available):
            self.order_target_percent(d, target=weights[i])

#%%
cerebro = bt.Cerebro(stdstats=False)
cerebro.broker.set_coc(True)
df = pd.read_csv(TICKER_FILE)
tickers = df['Symbol']

#%%
print(f'Gradding data from files...')
for _, ticker in tickers.iteritems():
    try:
        data = bt.feeds.GenericCSVData(
            fromdate=start,
            todate=end,
            dataname=f'spy/{ticker}.csv',
            dtformat=( '%Y-%m-%d'),
            openinterest=-1,
            nullvalue=0.0,
            plot=False
        )
        cerebro.adddata(data)
    except Exception as e:
        print(f'Unable to load data from spy/{ticker}.csv, so ignorning\n{e}')

print('Data added.  Running Cerebro to get your results...')
cerebro.broker.setcash(1_000_000)
cerebro.addobserver(bt.observers.Value)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
cerebro.addanalyzer(bt.analyzers.Returns)
cerebro.addanalyzer(bt.analyzers.DrawDown)
cerebro.addstrategy(CrossSectionalMR)
results = cerebro.run()

print(f"Sharpe{results[0].analyzers.sharperatio.get_analysis()}")
print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
cerebro.plot()[0][0]



