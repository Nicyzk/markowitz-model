# create bar chart from csv file
import pandas as pd

stats_df = pd.read_csv('./stats.csv')

max_sr_total_log_means = stats_df.loc[(stats_df['stock'] == 'Max_SR')]['means'].sum()
gspc_total_log_means = stats_df.loc[(stats_df['stock'] == '^GSPC')]['means'].sum()

max_sr_mean_risk = stats_df.loc[(stats_df['stock'] == 'Max_SR')]['risks'].mean()
gspc_mean_risk = stats_df.loc[(stats_df['stock'] == '^GSPC')]['risks'].mean()

max_sr_mean_sharpe = stats_df.loc[(stats_df['stock'] == 'Max_SR')]['sharpe'].mean()
gspc_mean_sharpe = stats_df.loc[(stats_df['stock'] == '^GSPC')]['sharpe'].mean()

total_log_returns_df = pd.DataFrame({'stock': ['Max_SR', '^GSPC'], 
'total log returns': [max_sr_total_log_means, gspc_total_log_means],
'mean risks': [max_sr_mean_risk, gspc_mean_risk],
'mean sharpe': [max_sr_mean_sharpe, gspc_mean_sharpe]
})
print(total_log_returns_df)