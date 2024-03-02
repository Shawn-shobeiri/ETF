import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from tiingo import TiingoClient
import datetime
from utility import utility as helper

st.set_page_config(layout="wide")

st.title('ETF Portfolio Optimizer')
st.sidebar.header('User Input Features')
input_tickers = st.sidebar.text_input('Enter ETF tickers separated by commas', 'RSP,VTI,SCHD')
start_date = st.sidebar.date_input('Start Date for Historical Prices')
num_sims = st.sidebar.number_input('Number of simulations', min_value=1000, max_value=250000, step=1000, value=50000)
risk_free_rate = st.sidebar.number_input('Risk-Free Rate', min_value=0.0, max_value=5.0, step=0.01, value=0.01)

if st.sidebar.button("Run Optimizer"):
    data = helper.get_data(input_tickers,start_date)
    rets, meanRets, covMat = helper.get_historical_mean_var(data)
    helper.plot_price_return(data,rets)
    helper.display_simulated_ef_with_random(mean_returns=meanRets, cov_matrix=covMat,num_portfolios=num_sims,risk_free_rate=risk_free_rate)