import pandas as pd 
import sqlite3

"""
From "Advances in Financial Machine Learning" by Marcos Lopez de Prado -- Volume Bars

When dealing with financial data it is customary to use daily prices and convert then 
to daily returns, which produces a stationary Gaussian distribution (albeit with fat 
tails, skewness). The distribution is highly non-normal and per the research cited here: 

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2034858

Using a sampling by volume method restores some normality. Sample the closing price
only after a certain volume (# of shares) are traded INSTEAD of based on unit time e.g
1 day. 
"""
QUERY = lambda volume_per_bar: f""" 
                    with cte_modulus as (select 
                          "Close", 
                          "Volume",
                          "Timestamp",
                          sum(cast(Volume as float)) over (order by "Timestamp") / {volume_per_bar} as volume_window
                        from 
                          minutes
                        )
                        select distinct 
                          last_value("Close") over (
                            partition by floor(volume_window)) as "close",
                          last_value("Timestamp") over (
                            partition by floor(volume_window)) as "close_time",
                          count(1) over (
                            partition by floor(volume_window)) as "time_per_bar"           
                        from 
                          cte_modulus
                    """


class VolumeClock: 
  
  def __init__(self):
    ''' Initialize the model by defining a SQLite database
    '''
    self.conn = sqlite3.connect('regimelab_VolumeClock.db')

  def load_table(self, name, table):
    ''' Initialize a single database table from the CSV data
    '''
    if 'Unnamed: 0' in table.columns: 
      table.drop(columns=['Unnamed: 0'], inplace=True)
    table.to_sql(name, self.conn, if_exists='replace', index=False)  

  def load_data(self, path): 
    ''' Load the CSVs into SQLite by passing in the command line paths to each respective 
        CSV and validating the data
    '''
    self.csv = pd.read_csv(path)
    self.load_table('minutes', self.csv)
    return self

  def load_volume_bars(self, volume_per_bar=1.):
    ''' Gets volume bars using the SQL query 
    '''
    cur = self.conn.cursor()
    cur.execute(QUERY(volume_per_bar))
    cur_data = cur.fetchall()
    cur.close()

    return pd.DataFrame(data=cur_data, columns=['close','close_time','time_per_bar'])
