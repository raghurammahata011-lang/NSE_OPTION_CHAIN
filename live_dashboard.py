# [file name]: live_dashboard.py
# [file content begin]
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import time

def create_live_dashboard_tab():
    """Create a live market dashboard with real-time data."""
    st.markdown("### 📈 Live Market Dashboard")
    
    # Fetch real-time market data
    with st.spinner("Fetching live market data..."):
        market_data = fetch_live_market_data()
    
    # Display market overview
    st.markdown("##### 🎯 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        nifty_change = market_data.get('NIFTY', {}).get('change', 0)
        nifty_change_pct = market_data.get('NIFTY', {}).get('change_pct', 0)
        st.metric("NIFTY 50", f"{market_data.get('NIFTY', {}).get('price', 0):.2f}", 
                 delta=f"{nifty_change:.2f} ({nifty_change_pct:.2f}%)")
    
    with col2:
        banknifty_change = market_data.get('BANKNIFTY', {}).get('change', 0)
        banknifty_change_pct = market_data.get('BANKNIFTY', {}).get('change_pct', 0)
        st.metric("BANKNIFTY", f"{market_data.get('BANKNIFTY', {}).get('price', 0):.2f}", 
                 delta=f"{banknifty_change:.2f} ({banknifty_change_pct:.2f}%)")
    
    with col3:
        sensex_change = market_data.get('SENSEX', {}).get('change', 0)
        sensex_change_pct = market_data.get('SENSEX', {}).get('change_pct', 0)
        st.metric("SENSEX", f"{market_data.get('SENSEX', {}).get('price', 0):.2f}", 
                 delta=f"{sensex_change:.2f} ({sensex_change_pct:.2f}%)")
    
    with col4:
        vix_change = market_data.get('INDIAVIX', {}).get('change', 0)
        vix_change_pct = market_data.get('INDIAVIX', {}).get('change_pct', 0)
        st.metric("India VIX", f"{market_data.get('INDIAVIX', {}).get('price', 0):.2f}", 
                 delta=f"{vix_change:.2f} ({vix_change_pct:.2f}%)")
    
    # Top gainers and losers
    st.markdown("##### 📊 Top Gainers & Losers")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏆 Top Gainers**")
        gainers = market_data.get('top_gainers', [])
        if gainers:
            for i, stock in enumerate(gainers[:5]):
                st.write(f"{i+1}. {stock.get('symbol', '')}: ₹{stock.get('price', 0):.2f} (+{stock.get('change_pct', 0):.2f}%)")
        else:
            st.info("No gainers data available")
    
    with col2:
        st.markdown("**📉 Top Losers**")
        losers = market_data.get('top_losers', [])
        if losers:
            for i, stock in enumerate(losers[:5]):
                st.write(f"{i+1}. {stock.get('symbol', '')}: ₹{stock.get('price', 0):.2f} ({stock.get('change_pct', 0):.2f}%)")
        else:
            st.info("No losers data available")
    
    # Sector performance
    st.markdown("##### 🏢 Sector Performance")
    
    sectors = market_data.get('sector_performance', {})
    if sectors:
        sector_data = []
        for sector, performance in sectors.items():
            sector_data.append({
                'Sector': sector,
                'Performance': performance
            })
        
        sector_df = pd.DataFrame(sector_data)
        sector_df = sector_df.sort_values('Performance', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=sector_df['Performance'],
            y=sector_df['Sector'],
            orientation='h',
            marker_color=np.where(sector_df['Performance'] > 0, 'green', 'red')
        ))
        
        fig.update_layout(
            title="Sector Performance",
            xaxis_title="Percentage Change",
            yaxis_title="Sector",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sector performance data not available")
    
    # Market heatmap
    st.markdown("##### 🔥 Market Heatmap")
    
    # This would typically use a real API for heatmap data
    # For demonstration, we'll create mock data
    heatmap_data = {
        'NIFTY': np.random.uniform(-2, 3),
        'BANKNIFTY': np.random.uniform(-2, 3),
        'FINNIFTY': np.random.uniform(-2, 3),
        'RELIANCE': np.random.uniform(-2, 3),
        'TCS': np.random.uniform(-2, 3),
        'HDFCBANK': np.random.uniform(-2, 3),
        'INFY': np.random.uniform(-2, 3),
        'ICICIBANK': np.random.uniform(-2, 3),
        'SBIN': np.random.uniform(-2, 3),
        'HINDUNILVR': np.random.uniform(-2, 3),
        'ITC': np.random.uniform(-2, 3),
        'BAJFINANCE': np.random.uniform(-2, 3),
        'BHARTIARTL': np.random.uniform(-2, 3),
        'KOTAKBANK': np.random.uniform(-2, 3),
        'AXISBANK': np.random.uniform(-2, 3)
    }
    
    heatmap_df = pd.DataFrame.from_dict(heatmap_data, orient='index', columns=['Change'])
    heatmap_df = heatmap_df.sort_values('Change', ascending=False)
    
    fig = go.Figure(data=go.Heatmap(
        z=[heatmap_df['Change'].values],
        x=heatmap_df.index,
        y=[''],
        colorscale='RdYlGn',
        hoverongaps=False,
        text=[heatmap_df['Change'].values],
        texttemplate="%{text}%",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Live Market Heatmap",
        xaxis_title="Stocks/Indices",
        height=200
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Real-time charts
    st.markdown("##### 📈 Real-time Charts")
    
    # Select a symbol for real-time chart
    selected_symbol = st.selectbox("Select Symbol for Live Chart", 
                                 options=["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "HDFCBANK"])
    
    # Fetch intraday data
    intraday_data = fetch_intraday_data(selected_symbol)
    
    if not intraday_data.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=intraday_data.index,
            open=intraday_data['Open'],
            high=intraday_data['High'],
            low=intraday_data['Low'],
            close=intraday_data['Close'],
            name='Price'
        ))
        
        fig.update_layout(
            title=f"Intraday Chart - {selected_symbol}",
            xaxis_title="Time",
            yaxis_title="Price (₹)",
            height=400,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Intraday data not available")
    
    # Market news
    st.markdown("##### 📰 Latest Market News")
    
    news = fetch_market_news()
    if news:
        for i, item in enumerate(news[:5]):
            with st.expander(f"{item.get('title', '')} - {item.get('time', '')}"):
                st.write(item.get('summary', ''))
                if item.get('url'):
                    st.markdown(f"[Read more]({item.get('url')})")
    else:
        st.info("Market news not available at the moment")

def fetch_live_market_data():
    """Fetch live market data from various sources."""
    # This is a mock implementation - in a real app, you would use APIs
    # like NSE, Yahoo Finance, or other financial data providers
    
    market_data = {
        'NIFTY': {
            'price': np.random.uniform(19500, 19800),
            'change': np.random.uniform(-100, 100),
            'change_pct': np.random.uniform(-2, 2)
        },
        'BANKNIFTY': {
            'price': np.random.uniform(43500, 44500),
            'change': np.random.uniform(-200, 200),
            'change_pct': np.random.uniform(-2, 2)
        },
        'SENSEX': {
            'price': np.random.uniform(65000, 66500),
            'change': np.random.uniform(-300, 300),
            'change_pct': np.random.uniform(-2, 2)
        },
        'INDIAVIX': {
            'price': np.random.uniform(15, 25),
            'change': np.random.uniform(-2, 2),
            'change_pct': np.random.uniform(-10, 10)
        },
        'top_gainers': [
            {'symbol': 'RELIANCE', 'price': np.random.uniform(2500, 2800), 'change_pct': np.random.uniform(1, 5)},
            {'symbol': 'TCS', 'price': np.random.uniform(3500, 3800), 'change_pct': np.random.uniform(1, 4)},
            {'symbol': 'HDFCBANK', 'price': np.random.uniform(1600, 1650), 'change_pct': np.random.uniform(1, 3)},
            {'symbol': 'INFY', 'price': np.random.uniform(1800, 1850), 'change_pct': np.random.uniform(1, 3)},
            {'symbol': 'ICICIBANK', 'price': np.random.uniform(950, 1000), 'change_pct': np.random.uniform(1, 3)}
        ],
        'top_losers': [
            {'symbol': 'ITC', 'price': np.random.uniform(400, 420), 'change_pct': np.random.uniform(-3, -1)},
            {'symbol': 'SBIN', 'price': np.random.uniform(550, 600), 'change_pct': np.random.uniform(-3, -1)},
            {'symbol': 'HINDUNILVR', 'price': np.random.uniform(2400, 2500), 'change_pct': np.random.uniform(-2, -1)},
            {'symbol': 'BAJFINANCE', 'price': np.random.uniform(6500, 7000), 'change_pct': np.random.uniform(-2, -1)},
            {'symbol': 'BHARTIARTL', 'price': np.random.uniform(900, 950), 'change_pct': np.random.uniform(-2, -1)}
        ],
        'sector_performance': {
            'IT': np.random.uniform(-2, 3),
            'BANKING': np.random.uniform(-2, 3),
            'AUTO': np.random.uniform(-2, 3),
            'PHARMA': np.random.uniform(-2, 3),
            'FMCG': np.random.uniform(-2, 3),
            'METAL': np.random.uniform(-2, 3),
            'REALTY': np.random.uniform(-2, 3),
            'ENERGY': np.random.uniform(-2, 3)
        }
    }
    
    return market_data

def fetch_intraday_data(symbol, period="1d", interval="5m"):
    """Fetch intraday data for a symbol."""
    try:
        # Map symbol to yfinance format
        symbol_map = {
            "NIFTY": "^NSEI",
            "BANKNIFTY": "^NSEBANK", 
            "SENSEX": "^BSESN",
            "RELIANCE": "RELIANCE.NS",
            "TCS": "TCS.NS",
            "HDFCBANK": "HDFCBANK.NS"
        }
        
        yf_symbol = symbol_map.get(symbol, f"{symbol}.NS")
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period=period, interval=interval)
        return hist
    except:
        return pd.DataFrame()

def fetch_market_news():
    """Fetch market news from various sources."""
    # This is a mock implementation - in a real app, you would use news APIs
    # or scrape financial news websites
    
    news = [
        {
            'title': 'RBI Keeps Repo Rate Unchanged at 6.5%',
            'summary': 'The Reserve Bank of India has decided to maintain the repo rate at 6.5% in its latest monetary policy meeting, citing inflationary concerns.',
            'time': '2 hours ago',
            'url': '#'
        },
        {
            'title': 'Government Announces Infrastructure Boost Package',
            'summary': 'The government has unveiled a new infrastructure development package worth ₹1.5 lakh crore, focusing on roads, railways, and renewable energy projects.',
            'time': '4 hours ago',
            'url': '#'
        },
        {
            'title': 'IT Sector Q2 Results Show Mixed Performance',
            'summary': 'Major IT companies have reported mixed results for Q2, with some showing strong growth while others face margin pressures due to global economic uncertainties.',
            'time': '6 hours ago',
            'url': '#'
        },
        {
            'title': 'Monsoon Progress Boosts Agricultural Stocks',
            'summary': 'Better-than-expected monsoon progress has led to a rally in agricultural and fertilizer stocks, with analysts predicting strong rural demand.',
            'time': 'Yesterday',
            'url': '#'
        },
        {
            'title': 'Global Markets Show Volatility Amid Economic Data',
            'summary': 'International markets have shown increased volatility as investors react to mixed economic data from major economies and central bank policy signals.',
            'time': 'Yesterday',
            'url': '#'
        }
    ]
    
    return news
# [file content end]