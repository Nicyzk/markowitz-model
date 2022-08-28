import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
   
# x = pd.DataFrame({'x': [1, 2, 3, 1], 'y': [1, 1, 5, 1]})
# print(x.loc[(x['x'] == 1) & (x['y'] == 1)].iloc[0]['x'])
# print(x.loc[(x['x'] == 1) & (x['y'] == 1)]['x'].sum())


chart_data = pd.DataFrame({'year': [2000, 2001, 2002, 2003], 'stock': ['max_sr', 'gspc', 'max_sr', 'gspc'], 'means': [1, 1, 5, 1], 'risks': [1, 2, 3, 4]})
max_sr_df = chart_data.loc[chart_data['stock'] == 'max_sr']
gspc_df = chart_data.loc[chart_data['stock'] == 'gspc']
print(max_sr_df)

years = chart_data['year'].to_numpy()
max_sr_means = max_sr_df['means'].to_numpy()
gspc_means = gspc_df['means'].to_numpy()


print(max_sr_means)
print(gspc_means)

# Women = [115, 215, 250, 200]
# Men = [114, 230, 510, 370]
  
n = len(years)
r = np.arange(n)
width = 1/n
  
  
# plt.bar(r, Women, color = 'b',
#         width = width, edgecolor = 'black',
#         label='Women')
# plt.bar(r + width, Men, color = 'g',
#         width = width, edgecolor = 'black',
#         label='Men')
  
# plt.xlabel("Year")
# plt.ylabel("Annual log mean returns")
# plt.title("Annual log mean returns (Max_SR vs S&P500)")
  
# # plt.grid(linestyle='--')
# plt.xticks(r + width/2,['2018','2019','2020','2021'])
# plt.legend()
  
# plt.show()