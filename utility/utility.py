
import numpy as  np
import datetime as dt
import streamlit as st
import requests
import pandas as pd
import datetime as dt
from tiingo import TiingoClient
import pandas_datareader as pdr
import numpy as np
import plotly.express as px

headers = {
    'Content-Type': 'application/json',
    'Authorization' : 'Token b21b49eae4c5423a0a68887ebf581eaa9a09ea51'
}

requestResponse = requests.get("https://api.tiingo.com/api/test/",
                                    headers=headers)

def get_data(ETFs, startDate):
    api_key = 'b21b49eae4c5423a0a68887ebf581eaa9a09ea51'
    config={}
    config['session'] = True
    config['api_key'] = api_key
    client = TiingoClient(config)
    startDate = startDate.strftime('%Y-%m-%d')
    endDate = dt.datetime.now().strftime('%Y-%m-%d')
    ETFs = [ticker.strip().upper() for ticker in ETFs.split(',')]
    historical = client.get_dataframe(ETFs, frequency='daily', startDate=startDate, endDate=endDate, metric_name='adjClose')
    historical = pd.DataFrame(historical)
    historical.reset_index(inplace=True)
    first_non_nan_indices = {column: historical[column].first_valid_index() + 1 for column in historical.columns if historical[column].first_valid_index() is not None}
    historical.set_index('index',inplace=True)
    hist = historical.iloc[max(first_non_nan_indices.values()):,:]

    return hist

def get_historical_mean_var(historical):
    returns = historical.pct_change()
    returns = returns.iloc[1:]
    meanReturns = returns.mean()
    covMatrix = returns.cov()

    return returns, meanReturns, covMatrix

def plot_price_return(historical,returns):
    fig1 = px.line(historical)
    fig1.update_layout(
        xaxis_title='Date',
        yaxis_title='Price'
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(returns)
    fig2.update_layout(
        xaxis_title='Date',
        yaxis_title='Return'
    )
    st.plotly_chart(fig2, use_container_width=True)

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights)*252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))*np.sqrt(252)
    return std, returns

def random_portfolios(num_sims, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((num_sims,3+len(mean_returns)))
    weights_record = []
    for i in range(num_sims):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[i,0] = portfolio_std_dev
        results[i,1] = portfolio_return
        results[i,2] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        col_assets = []
        for j in range(len(weights)):
            results[i,3+j] = weights[j]
            col_assets.append(mean_returns.index[j])
        cols = ['std','return','sharpe'] + col_assets
        flattened_cols = [item for sublist in cols for item in (sublist if isinstance(sublist, list) else [sublist])]
        results_df = pd.DataFrame(
            data=results,
            columns=flattened_cols
        )
    return results_df, results, weights_record

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results_df, results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    max_sharpe_idx = np.argmax(results[:,2])
    sdp, rp = results[max_sharpe_idx, 0], results[max_sharpe_idx, 1]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,6)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation.index = mean_returns.index
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[:,0])
    sdp_min, rp_min = results[min_vol_idx,0], results[min_vol_idx,1]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,6)for i in min_vol_allocation.allocation]
    min_vol_allocation.index = mean_returns.index
    min_vol_allocation = min_vol_allocation.T

    st.write("Maximum Sharpe Ratio Portfolio Allocation")
    st.dataframe(max_sharpe_allocation)
    st.write("Annualised Return:", round(rp,3))
    st.write("Annualised Volatility:", round(sdp,3))
    st.write("-"*80)
    st.write("Minimum Volatility Portfolio Allocation")
    st.dataframe(min_vol_allocation)
    st.write("Annualised Return:", round(rp_min,3))
    st.write("Annualised Volatility:", round(sdp_min,3))
    cols = ['Annualised Volatility','Annualised Return','Sharpe Ratio'] + list(mean_returns.index)
    #results = results[:,0:3]
    results_df = pd.DataFrame(
        data=results,
        columns=cols
    )
    print(results_df.iloc[:,3:].columns)
    fig = px.scatter(
        results_df,
        x='Annualised Volatility', y='Annualised Return', 
        color='Sharpe Ratio', 
        color_continuous_scale='Blues',
        labels={'x': 'Annualised Volatility', 'y': 'Annualised Return'},
        title='Simulated Portfolio Optimization based on Efficient Frontier',
        hover_data={**{'Sharpe Ratio': ':.2f'},**{x:':.2f' for x in results_df.iloc[:,3:].columns}}
    )
    fig.add_scatter(x=[sdp], y=[rp], marker_symbol='star', marker_size=35, marker_color='red', name='Maximum Sharpe Ratio', hoverinfo='skip')
    fig.add_scatter(x=[sdp_min], y=[rp_min], marker_symbol='star', marker_size=35, marker_color='green', name='Minimum Volatility', hoverinfo='skip')
    fig.update_layout(
        width=2000, height=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig)

