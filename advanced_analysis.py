# advanced_analysis.py - Additional analysis modules for NSE Option Chain Analytics Pro

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import talib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_sector_analysis_tab(hist_data, symbol):
    """Create a sector analysis tab with peer comparison."""
    st.markdown(f"### 📊 Sector Analysis for {symbol}")
    
    # This would typically fetch from a database or API
    # For demonstration, we'll create mock sector data
    sector_peers = {
        "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
        "BANKING": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
        "AUTO": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO"],
        "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "BIOCON"],
        "FMCG": ["HINDUNILVR", "ITC", "NESTLE", "BRITANNIA", "DABUR"]
    }
    
    # Determine which sector the symbol belongs to
    symbol_sector = None
    for sector, peers in sector_peers.items():
        if symbol in peers:
            symbol_sector = sector
            break
    
    if symbol_sector is None:
        st.info("Sector information not available for this symbol.")
        return
    
    st.markdown(f"##### 🏢 Sector: {symbol_sector}")
    
    # Mock sector performance data
    sector_performance = {
        "IT": {"1D": 0.8, "1W": 2.1, "1M": 5.3, "YTD": 15.7},
        "BANKING": {"1D": -0.3, "1W": 1.2, "1M": 3.8, "YTD": 12.4},
        "AUTO": {"1D": 0.5, "1W": 1.8, "1M": 4.2, "YTD": 9.6},
        "PHARMA": {"1D": 1.2, "1W": 2.5, "1M": 6.1, "YTD": 18.3},
        "FMCG": {"1D": 0.3, "1W": 1.1, "1M": 3.2, "YTD": 10.8}
    }
    
    # Display sector performance
    if symbol_sector in sector_performance:
        perf = sector_performance[symbol_sector]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("1D Return", f"{perf['1D']}%", delta=f"{perf['1D']}%")
        with col2:
            st.metric("1W Return", f"{perf['1W']}%", delta=f"{perf['1W']}%")
        with col3:
            st.metric("1M Return", f"{perf['1M']}%", delta=f"{perf['1M']}%")
        with col4:
            st.metric("YTD Return", f"{perf['YTD']}%", delta=f"{perf['YTD']}%")
    
    # Peer comparison (mock data)
    st.markdown("##### 📈 Peer Comparison")
    
    # Create mock peer data
    peer_data = []
    for peer in sector_peers[symbol_sector]:
        # Generate random performance data for demonstration
        peer_data.append({
            "Symbol": peer,
            "Price": np.random.uniform(100, 5000),
            "1D %": np.random.uniform(-2, 3),
            "1W %": np.random.uniform(-5, 7),
            "1M %": np.random.uniform(-10, 15),
            "YTD %": np.random.uniform(-15, 25)
        })
    
    peer_df = pd.DataFrame(peer_data)
    peer_df = peer_df.sort_values("YTD %", ascending=False)
    
    # Highlight the current symbol
    def highlight_current_symbol(row):
        if row['Symbol'] == symbol:
            return ['background-color: lightblue'] * len(row)
        return [''] * len(row)
    
    st.dataframe(peer_df.style.apply(highlight_current_symbol, axis=1))
    
    # Sector heatmap
    st.markdown("##### 🔥 Sector Performance Heatmap")
    
    # Create heatmap data
    sectors = list(sector_performance.keys())
    timeframes = ["1D", "1W", "1M", "YTD"]
    
    heatmap_data = []
    for sector in sectors:
        row = [sector]
        for tf in timeframes:
            row.append(sector_performance[sector][tf])
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, columns=["Sector"] + timeframes)
    heatmap_df = heatmap_df.set_index("Sector")
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=timeframes,
        y=sectors,
        colorscale='RdYlGn',
        hoverongaps=False,
        text=heatmap_df.values,
        texttemplate="%{text}%",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Sector Performance Heatmap",
        xaxis_title="Timeframe",
        yaxis_title="Sector",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_market_sentiment_tab():
    """Create a market-wide sentiment analysis tab."""
    st.markdown("### 📊 Market-Wide Sentiment Analysis")
    
    # Mock market sentiment data
    market_data = {
        "NIFTY": {"PCR": 1.2, "IV": 18.5, "Trend": "Bullish", "Support": 19500, "Resistance": 19800},
        "BANKNIFTY": {"PCR": 0.9, "IV": 22.3, "Trend": "Neutral", "Support": 43500, "Resistance": 44500},
        "FINNIFTY": {"PCR": 1.1, "IV": 20.1, "Trend": "Bullish", "Support": 19200, "Resistance": 19600},
        "MIDCAP": {"PCR": 1.3, "IV": 24.7, "Trend": "Bullish", "Support": 32500, "Resistance": 33500},
        "SENSEX": {"PCR": 1.0, "IV": 17.8, "Trend": "Neutral", "Support": 65000, "Resistance": 66500}
    }
    
    # Overall market sentiment
    bullish_count = sum(1 for data in market_data.values() if data["Trend"] == "Bullish")
    total_count = len(market_data)
    market_sentiment = "Bullish" if bullish_count / total_count > 0.6 else "Bearish" if bullish_count / total_count < 0.4 else "Neutral"
    
    st.markdown(f"##### 🎯 Overall Market Sentiment: {market_sentiment}")
    
    # Market metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_pcr = np.mean([data["PCR"] for data in market_data.values()])
        st.metric("Average PCR", f"{avg_pcr:.2f}")
    with col2:
        avg_iv = np.mean([data["IV"] for data in market_data.values()])
        st.metric("Average IV", f"{avg_iv:.1f}%")
    with col3:
        st.metric("Bullish Indices", f"{bullish_count}/{total_count}")
    with col4:
        vix = 18.2  # Mock VIX value
        st.metric("India VIX", f"{vix:.1f}")
    
    # Market overview table
    st.markdown("##### 📋 Index Overview")
    market_df = pd.DataFrame.from_dict(market_data, orient='index')
    market_df.reset_index(inplace=True)
    market_df.rename(columns={'index': 'Index'}, inplace=True)
    
    st.dataframe(market_df, width='stretch')
    
    # Market breadth (mock data)
    st.markdown("##### 📊 Market Breadth")
    
    breadth_data = {
        "Time": ["9:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"],
        "Advances": [450, 620, 780, 850, 920, 880, 950],
        "Declines": [1050, 880, 720, 650, 580, 620, 550],
        "Unchanged": [100, 100, 100, 100, 100, 100, 100]
    }
    
    breadth_df = pd.DataFrame(breadth_data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=breadth_df['Time'], y=breadth_df['Advances'], 
                            name='Advances', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=breadth_df['Time'], y=breadth_df['Declines'], 
                            name='Declines', line=dict(color='red')))
    
    fig.update_layout(
        title="Market Breadth Throughout the Day",
        xaxis_title="Time",
        yaxis_title="Number of Stocks",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # FII/DII data (mock)
    st.markdown("##### 💰 Institutional Activity")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("FII Net Investment", "₹1,250 Cr", delta="₹1,250 Cr")
        st.metric("FII Derivative Position", "Long", delta="Bullish")
    with col2:
        st.metric("DII Net Investment", "₹980 Cr", delta="₹980 Cr")
        st.metric("DII Derivative Position", "Neutral", delta="Neutral")

def create_decision_dashboard(symbol, hist_data, analytics, predictions):
    """Create a comprehensive decision-making dashboard."""
    st.markdown(f"### 🎯 Decision Dashboard for {symbol}")
    
    if hist_data.empty:
        st.warning("Insufficient data for decision dashboard")
        return
    
    # Get current price and recent performance
    current_price = hist_data['Close'].iloc[-1]
    prev_close = hist_data['Close'].iloc[-2] if len(hist_data) > 1 else current_price
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close * 100) if prev_close != 0 else 0
    
    # Overall recommendation score
    recommendation_score = 0
    factors = []
    
    # Factor 1: Price trend
    if len(hist_data) > 20:
        sma_20 = hist_data['Close'].rolling(20).mean().iloc[-1]
        if current_price > sma_20:
            recommendation_score += 1
            factors.append(("Price above 20 SMA", "🟢"))
        else:
            recommendation_score -= 1
            factors.append(("Price below 20 SMA", "🔴"))
    
    # Factor 2: PCR
    pcr = analytics.get('pcr', 1)
    if pcr > 1.2:
        recommendation_score += 1
        factors.append((f"PCR Bullish ({pcr:.2f})", "🟢"))
    elif pcr < 0.8:
        recommendation_score -= 1
        factors.append((f"PCR Bearish ({pcr:.2f})", "🔴"))
    else:
        factors.append((f"PCR Neutral ({pcr:.2f})", "🟡"))
    
    # Factor 3: IV
    iv_atm = analytics.get('iv_atm', 20)
    if iv_atm > 25:
        recommendation_score += 0.5  # High IV can be good for option sellers
        factors.append((f"High IV ({iv_atm:.1f}%)", "🟢"))
    elif iv_atm < 15:
        recommendation_score -= 0.5  # Low IV can be good for option buyers
        factors.append((f"Low IV ({iv_atm:.1f}%)", "🔴"))
    else:
        factors.append((f"Moderate IV ({iv_atm:.1f}%)", "🟡"))
    
    # Factor 4: ML predictions
    if predictions:
        bull_count = sum(1 for pred in predictions.values() if pred.get('prediction', 0) == 1)
        bear_count = sum(1 for pred in predictions.values() if pred.get('prediction', 0) == 0)
        
        if bull_count > bear_count:
            recommendation_score += 1
            factors.append((f"ML Bullish ({bull_count}/{len(predictions)})", "🟢"))
        elif bear_count > bull_count:
            recommendation_score -= 1
            factors.append((f"ML Bearish ({bear_count}/{len(predictions)})", "🔴"))
        else:
            factors.append(("ML Neutral", "🟡"))
    
    # Determine overall recommendation
    if recommendation_score >= 2:
        overall_recommendation = "STRONG BUY"
        recommendation_color = "green"
    elif recommendation_score >= 1:
        overall_recommendation = "BUY"
        recommendation_color = "lightgreen"
    elif recommendation_score >= 0:
        overall_recommendation = "HOLD"
        recommendation_color = "orange"
    elif recommendation_score >= -1:
        overall_recommendation = "SELL"
        recommendation_color = "lightcoral"
    else:
        overall_recommendation = "STRONG SELL"
        recommendation_color = "red"
    
    # Display recommendation
    st.markdown(f"""
    <div style="background-color: {recommendation_color}; padding: 20px; border-radius: 10px; text-align: center;">
        <h2 style="margin: 0;">{overall_recommendation}</h2>
        <p style="margin: 0;">Recommendation Score: {recommendation_score:.1f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display factors
    st.markdown("##### 📋 Decision Factors")
    
    cols = st.columns(3)
    for i, (factor, emoji) in enumerate(factors):
        with cols[i % 3]:
            st.markdown(f"{emoji} {factor}")
    
    # Risk assessment
    st.markdown("##### ⚠️ Risk Assessment")
    
    # Calculate volatility
    volatility = hist_data['Close'].pct_change().std() * np.sqrt(252) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if volatility > 30:
            st.metric("Volatility", f"{volatility:.1f}%", delta="High", delta_color="inverse")
        elif volatility > 20:
            st.metric("Volatility", f"{volatility:.1f}%", delta="Medium", delta_color="off")
        else:
            st.metric("Volatility", f"{volatility:.1f}%", delta="Low", delta_color="normal")
    
    with col2:
        # Liquidity assessment (mock)
        liquidity = "High" if current_price > 1000 else "Medium" if current_price > 500 else "Low"
        st.metric("Liquidity", liquidity)
    
    with col3:
        # Market cap assessment (mock)
        market_cap = "Large" if symbol in ["RELIANCE", "TCS", "HDFCBANK", "INFY"] else "Mid" if symbol in ["WIPRO", "TECHM", "HCLTECH"] else "Small"
        st.metric("Market Cap", market_cap)
    
    # Suggested strategies
    st.markdown("##### 🎯 Suggested Strategies")
    
    if overall_recommendation in ["STRONG BUY", "BUY"]:
        strategies = [
            "Long Call Options",
            "Bull Call Spread",
            "Cash Secured Put Selling",
            "Stock Accumulation"
        ]
    elif overall_recommendation in ["STRONG SELL", "SELL"]:
        strategies = [
            "Long Put Options",
            "Bear Put Spread",
            "Covered Call Writing",
            "Stock Reduction"
        ]
    else:  # HOLD
        strategies = [
            "Iron Condor",
            "Calendar Spread",
            "Straddle/Strangle",
            "Theta Decay Strategies"
        ]
    
    for strategy in strategies:
        st.markdown(f"- {strategy}")
    
    # Position sizing guidance
    st.markdown("##### 📊 Position Sizing Guidance")
    
    # Risk-based position sizing
    account_size = st.number_input("Enter your account size (₹):", min_value=10000, value=100000, step=10000)
    risk_per_trade = st.slider("Risk per trade (%):", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    
    risk_amount = account_size * (risk_per_trade / 100)
    stop_loss_pct = st.slider("Stop Loss (%):", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
    
    position_size = risk_amount / (stop_loss_pct / 100 * current_price)
    
    st.metric("Recommended Position Size", f"{position_size:.0f} shares")
    st.metric("Risk Amount", f"₹{risk_amount:,.0f}")
    
    # Trade plan
    st.markdown("##### 📝 Trade Plan Template")
    
    with st.expander("Create Your Trade Plan"):
        st.text_input("Entry Price:", value=f"{current_price:.2f}")
        st.text_input("Stop Loss:", value=f"{current_price * (1 - stop_loss_pct/100):.2f}")
        st.text_input("Target 1:", value=f"{current_price * (1 + stop_loss_pct/100):.2f}")
        st.text_input("Target 2:", value=f"{current_price * (1 + stop_loss_pct/100 * 2):.2f}")
        st.text_area("Trade Rationale:", placeholder="Why are you taking this trade?")
        st.text_area("Risk Factors:", placeholder="What could go wrong?")

def create_advanced_analytics_tab(hist_data, symbol):
    """Create an advanced analytics tab with statistical analysis."""
    st.markdown(f"### 📊 Advanced Analytics for {symbol}")
    
    if hist_data.empty or len(hist_data) < 20:
        st.warning("Insufficient data for advanced analytics")
        return
    
    # Statistical analysis
    st.markdown("##### 📈 Statistical Analysis")
    
    returns = hist_data['Close'].pct_change().dropna()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Return", f"{returns.mean() * 100:.4f}%")
    with col2:
        st.metric("Return Std Dev", f"{returns.std() * 100:.4f}%")
    with col3:
        st.metric("Skewness", f"{returns.skew():.4f}")
    with col4:
        st.metric("Kurtosis", f"{returns.kurtosis():.4f}")
    
    # Distribution analysis
    st.markdown("##### 📊 Return Distribution")
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Return Distribution", "QQ Plot"))
    
    # Histogram
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name="Returns"), row=1, col=1)
    
    # QQ plot
    qq_data = stats.probplot(returns, dist="norm")
    theoretical_quantiles = qq_data[0][0]
    sample_quantiles = qq_data[0][1]
    
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sample_quantiles, 
                            mode='markers', name="QQ Plot"), row=1, col=2)
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles, 
                            mode='lines', name="Normal"), row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatility analysis
    st.markdown("##### 📉 Volatility Analysis")
    
    # Calculate rolling volatility
    rolling_volatility = returns.rolling(20).std() * np.sqrt(252) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_volatility.index, y=rolling_volatility, 
                            name="20-Day Volatility"))
    fig.add_hline(y=rolling_volatility.mean(), line_dash="dash", 
                 annotation_text=f"Mean: {rolling_volatility.mean():.1f}%")
    
    fig.update_layout(
        title="Rolling 20-Day Volatility",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis (mock)
    st.markdown("##### 🔗 Correlation Analysis")
    
    # Mock correlation data
    correlation_data = {
        "NIFTY": np.random.uniform(0.7, 0.9),
        "BANKNIFTY": np.random.uniform(0.6, 0.8),
        "USD/INR": np.random.uniform(-0.3, 0.3),
        "Gold": np.random.uniform(-0.2, 0.2),
        "Crude Oil": np.random.uniform(-0.1, 0.4)
    }
    
    corr_df = pd.DataFrame.from_dict(correlation_data, orient='index', columns=['Correlation'])
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    
    st.dataframe(corr_df.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1))
    
    # Monte Carlo simulation (simplified)
    st.markdown("##### 🎲 Monte Carlo Simulation")
    
    n_simulations = st.slider("Number of Simulations:", 100, 10000, 1000)
    n_days = st.slider("Forecast Period (Days):", 5, 365, 30)
    
    if st.button("Run Simulation"):
        with st.spinner("Running Monte Carlo simulation..."):
            # Simple Monte Carlo simulation
            last_price = hist_data['Close'].iloc[-1]
            mu = returns.mean()
            sigma = returns.std()
            
            # Generate random walks
            simulations = np.zeros((n_days, n_simulations))
            simulations[0] = last_price
            
            for day in range(1, n_days):
                shock = np.random.normal(mu, sigma, n_simulations)
                simulations[day] = simulations[day-1] * (1 + shock)
            
            # Plot results
            fig = go.Figure()
            for i in range(min(100, n_simulations)):  # Plot first 100 simulations
                fig.add_trace(go.Scatter(y=simulations[:, i], mode='lines', 
                                        line=dict(width=1), showlegend=False))
            
            # Add percentiles
            p5 = np.percentile(simulations, 5, axis=1)
            p95 = np.percentile(simulations, 95, axis=1)
            
            fig.add_trace(go.Scatter(y=p5, mode='lines', 
                                    line=dict(width=3, color='red'), name='5th Percentile'))
            fig.add_trace(go.Scatter(y=p95, mode='lines', 
                                    line=dict(width=3, color='green'), name='95th Percentile'))
            
            fig.update_layout(
                title=f"Monte Carlo Simulation ({n_simulations} runs)",
                xaxis_title="Days",
                yaxis_title="Price",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics
            final_prices = simulations[-1]
            expected_price = np.mean(final_prices)
            confidence_interval = np.percentile(final_prices, [5, 95])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Price", f"₹{expected_price:.2f}")
            with col2:
                st.metric("5% Confidence", f"₹{confidence_interval[0]:.2f}")
            with col3:
                st.metric("95% Confidence", f"₹{confidence_interval[1]:.2f}")