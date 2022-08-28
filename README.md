# Comparison of Modern Portfolio Theory against S&P500

## Overview
This project implements the Markowitz Model and seeks to compare its performance against S&P500 over different time periods. For a selected year e.g. 2020, we select the largest 20 stocks by market cap in the S&P500 at the end of the **previous year** i.e., 2019 (to prevent lookahead bias). Using the Markowitze model and historical price data from the previous 5 years (2015-2019), we construct 2 portfolios - 1. maximum sharpe ratio and 2. minimum volatility. The stock weightings from the 2 portfolios are used for a buy-and-hold strategy over the year 2020 and performance is compared against the S&P500. 

The strategy comparison is conducted over the time period 2001-2020, and the performance outcomes (sharpe ratio and returns) are shown below. 

Top 10 stocks per year is manually compiled from https://www.youtube.com/watch?v=kfMFDcuDKYA

</br>

## Implementation Details

### Formula for portfolio variance
Let $\sum$ be the covariance matrix of the log daily returns of all the stocks.  
Let W be the vector of portfolio weights.  
Portfolio variance = W<sup>T</sup> $\sum$ W

</br>

### The Objective Function
Constraints: 
- sum of weights = 1 
- weight of each stock is in [0,1]. 

</br>

## Performance Outcomes

### 1.Comparison of annual returns of Max_SR portfolio vs S&P500
![Comparison of annual returns of Max_SR portfolio vs S&P500](./max_sr%20vs%20S%26P500%20means.png)

</br>

### 2. Overall stats of Max_SR portfolio vs S&P500 over time period (the detailed stats are found in stats.csv)
| stock  | total log returns | mean risks | mean sharpe |
|--------|-------------------|------------|-------------|
| Max_SR | 1.675638          | 0.195354   | 0.607432    |
| S&P500 | 1.267836          | 0.175987   | 0.655678    |

When translated to % returns, we have 5.3422 (Max_SR) vs 3.5531 (S&P500).  
</br>

## Conclusion
Constructing a portfolio that maximizes sharpe ratio using the method above, outperforms the S&P500 by a significant margin whilst having a lower mean sharpe ratio. 