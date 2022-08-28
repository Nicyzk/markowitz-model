# create bar chart from csv file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stats_df = pd.read_csv('./stats.csv')
years = stats_df.loc[stats_df['stock'] == 'Max_SR']['year'].to_numpy()
max_sr_means = stats_df.loc[stats_df['stock'] == 'Max_SR']['means'].to_numpy()
gspc_means = stats_df.loc[stats_df['stock'] == '^GSPC']['means'].to_numpy()

n = len(years)
X_axis = np.arange(n)
width = 1/n
  
plt.bar(X_axis - 0.2, max_sr_means, 0.4, label = 'max_sr')
plt.bar(X_axis + 0.2, gspc_means, 0.4, label = '^gspc')
  
plt.xticks(X_axis, years)
plt.xlabel("Year")
plt.ylabel("Log Mean Annual Returns")
plt.title("Comparing log annual mean returns of Max_SR portfolio vs S&P500")
plt.legend()
plt.show()
  