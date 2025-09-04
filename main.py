# ---------------- MAIN APP ----------------
def main():
    """Enhanced main Streamlit application."""
    st.set_page_config(
        page_title="NSE Option Chain Analytics Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom CSS
    st.markdown(PRO_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 NSE Option Chain Analytics Pro</h1>
        <p>Advanced Market Analysis with Live Dashboard & Automated Reporting</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Configuration")

    # Symbol type selection with enhanced options
    symbol_type = st.sidebar.radio(
        "Select Data Type:",
        ["Indices", "Popular Stocks", "IT Sector", "Banking", "Custom"]
    )

    # Enhanced symbol selection based on type
    if symbol_type == "Indices":
        available_symbols = list(st.session_state.indices) + ["NIFTYIT", "NIFTYPHARMA", "NIFTYAUTO", "NIFTYMETAL"]
    elif symbol_type == "Popular Stocks":
        available_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "ITC", "LT", "BHARTIARTL", "ASIANPAINT", "MARUTI"]
    elif symbol_type == "IT Sector":
        available_symbols = ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MINDTREE", "COFORGE"]
    elif symbol_type == "Banking":
        available_symbols = ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "BANDHANBNK"]
    else:  # Custom
        available_symbols = st.session_state.watchlist if st.session_state.watchlist else ["TCS"]

    symbol = st.sidebar.selectbox(
        "Select Symbol:",
        options=available_symbols,
        index=0 if available_symbols else None
    )

    # Manual symbol input for custom selection
    if symbol_type == "Custom":
        manual_symbol = st.sidebar.text_input("Or enter symbol manually:", placeholder="e.g., BHARTIARTL, NESTLEIND")
        if manual_symbol:
            symbol = manual_symbol.upper()

    # Enhanced data source options
    st.sidebar.markdown("#### 📡 Data Source Options")
    use_alternative_source = st.sidebar.checkbox("Use Alternative NSE API", value=False)
    historical_period = st.sidebar.selectbox("Historical Data Period", 
                                           ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"], 
                                           index=4)  # Default to 6mo

    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 30)

    if auto_refresh != st.session_state.auto_refresh:
        st.session_state.auto_refresh = auto_refresh

    # Manage watchlist
    manage_watchlist()

    # Risk management settings
    st.sidebar.markdown("#### 🎯 Risk Parameters")
    risk_free_rate = st.sidebar.slider("Risk Free Rate (%)", 0.0, 10.0, 5.0) / 100
    days_to_expiry = st.sidebar.number_input("Days to Expiry", 1, 365, 30)

    # Fetch data with enhanced error handling
    with st.spinner(f"Fetching {symbol} data from NSE & Yahoo Finance..."):
        # Get historical data first to validate symbol
        hist_data = fetch_historical_data(symbol, period=historical_period)

        if hist_data.empty:
            st.error(f"❌ Unable to fetch historical data for {symbol}.")
            st.info("💡 **Troubleshooting Tips:**")
            st.info("1. Check if the symbol is correct (e.g., TCS, RELIANCE)")
            st.info("2. Try a different time period")
            st.info("3. Use the 'Popular Stocks' or 'IT Sector' categories")
            return
        else:
            # Calculate technical indicators
            hist_data = calculate_technical_indicators(hist_data)

            # Show basic info about the stock
            current_price = hist_data['Close'].iloc[-1]
            price_change = hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2] if len(hist_data) > 1 else 0
            price_change_pct = (price_change / hist_data['Close'].iloc[-2] * 100) if len(hist_data) > 1 and hist_data['Close'].iloc[-2] != 0 else 0

            st.info(f"📈 **{symbol}** - Current Price: ₹{current_price:.2f} ({price_change:+.2f}, {price_change_pct:+.2f}%)")

        # Try to fetch option chain data
        if use_alternative_source:
            df_alt = fetch_nse_options_alternative(symbol)
            if not df_alt.empty:
                df = df_alt
                st.success("✅ Using alternative NSE data source")
            else:
                session = get_nse_session()
                raw_data = fetch_option_chain(symbol, session)
                df = parse_data(symbol, raw_data)
        else:
            session = get_nse_session()
            raw_data = fetch_option_chain(symbol, session)
            df = parse_data(symbol, raw_data)

    # Calculate analytics
    analytics = calculate_analytics(df, current_price if not hist_data.empty else None)
    spot_price = analytics.get('spot_price', current_price if not hist_data.empty else 0)

    # Main dashboard metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}" if not hist_data.empty else "N/A",
                 delta=f"{price_change:+.2f}" if not hist_data.empty else None)
    with col2:
        pcr = analytics.get('pcr', 0)
        pcr_delta = "📈" if pcr > 1.2 else "📉" if pcr < 0.8 else "➡️"
        st.metric("PCR", f"{pcr:.2f} {pcr_delta}" if pcr > 0 else "N/A")
    with col3:
        st.metric("Max Pain", f"₹{analytics.get('max_pain', 0):,.0f}" if analytics.get('max_pain', 0) > 0 else "N/A")
    with col4:
        st.metric("IV ATM", f"{analytics.get('iv_atm', 0):.2f}%" if analytics.get('iv_atm', 0) > 0 else "N/A")
    with col5:
        bias = analytics.get('directional_bias', 'Neutral')
        bias_emoji = "🟢" if bias == "Bullish" else "🔴" if bias == "Bearish" else "🟡"
        st.metric("Bias", f"{bias} {bias_emoji}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14 = st.tabs([
        "📊 Option Chain", "📈 Charts", "🧠 ML Predictions", 
        "📋 Technical Analysis", "🎯 Strategy Builder", "💾 Export", "🔍 More Analysis", 
        "📱 Intraday Analysis", "🏢 Sector Analysis", "📊 Market Sentiment", "🎯 Decision Dashboard",
        "📈 Live Dashboard", "📰 News & Social", "⏰ Excel Automation"
    ])

        with tab1:
        st.markdown("### 📊 Option Chain Data")
        
        if df.empty:
            st.warning("⚠️ Option chain data not available for this symbol")
            st.info("This may be because:")
            st.info("• Options are not traded for this symbol")
            st.info("• NSE servers are temporarily unavailable")
            st.info("• Symbol might be delisted or suspended")
        else:
            # Enhanced analytics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🔍 Key Metrics")
                metrics_data = {
                    'Metric': ['PCR', 'ATM PCR', 'IV Skew', 'Expected Move'],
                    'Value': [
                        f"{analytics.get('pcr', 0):.2f}",
                        f"{analytics.get('atm_pcr', 0):.2f}",
                        f"{analytics.get('iv_skew', 0):.2f}%",
                        f"₹{analytics.get('expected_move_30d', 0):.2f}"
                    ]
                }
                metrics_df = create_safe_dataframe(metrics_data)
                st.dataframe(metrics_df, width='stretch')
            
            with col2:
                st.markdown("#### 🎯 Support/Resistance")
                sr_data = {
                    'Level': ['Strong Support', 'Max Pain', 'Strong Resistance'],
                    'Strike': [
                        f"₹{analytics.get('support', 0):,.0f}",
                        f"₹{analytics.get('max_pain', 0):,.0f}",
                        f"₹{analytics.get('resistance', 0):,.0f}"
                    ]
                }
                sr_df = create_safe_dataframe(sr_data)
                st.dataframe(sr_df, width='stretch')
            
            with col3:
                st.markdown("#### 💰 Option Flow")
                flow_data = {
                    'Type': ['Call Flow', 'Put Flow', 'Net Flow'],
                    'Value': [
                        f"₹{analytics.get('call_flow', 0):,.2f}",
                        f"₹{analytics.get('put_flow', 0):,.2f}",
                        f"₹{analytics.get('net_flow', 0):,.2f}"
                    ]
                }
                flow_df = create_safe_dataframe(flow_data)
                st.dataframe(flow_df, width='stretch')
            
            # Option chain table
            st.markdown("#### 📋 Complete Option Chain")
            display_df = df[['STRIKE', 'CALL_OI', 'CALL_CHNG_IN_OI', 'CALL_IV', 'CALL_LTP',
                            'PUT_LTP', 'PUT_IV', 'PUT_CHNG_IN_OI', 'PUT_OI']].copy()
            
            display_df = ensure_dataframe_types(display_df)
            st.dataframe(display_df, width='stretch', height=400)
    
    with tab2:
        st.markdown("### 📈 Visual Analytics")
        
        if not df.empty:
            # OI Distribution
            st.markdown("##### 📊 Open Interest Distribution")
            oi_fig = plot_oi_distribution(df, spot_price)
            st.plotly_chart(oi_fig, width='stretch')
            
            # OI Change Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📈 OI Change Analysis")
                oi_change_fig = plot_oi_change(df, spot_price)
                st.plotly_chart(oi_change_fig, width='stretch')
            
            with col2:
                st.markdown("##### 🌊 Implied Volatility Surface")
                iv_fig = plot_iv_surface(df, spot_price)
                st.plotly_chart(iv_fig, width='stretch')
        else:
            st.info("📊 Option chain charts will be displayed when option data is available")
        
        # Always show technical analysis chart if historical data is available
        if not hist_data.empty:
            st.markdown("##### 📈 Enhanced Technical Analysis")
            tech_fig = plot_technical_indicators(hist_data)
            st.plotly_chart(tech_fig, width='stretch')
    
    with tab3:
        st.markdown("### 🧠 Machine Learning Predictions")
        
        # Enhanced ML predictions with fallback
        predictions = predict_price_movement_enhanced(hist_data, analytics, df)
        
        if predictions and len(predictions) > 0:
            st.markdown("##### 🎯 AI-Powered Price Movement Predictions")
            
            # Create dynamic columns based on number of predictions
            num_models = len(predictions)
            cols = st.columns(min(num_models, 4))  # Max 4 columns
            
            for i, (model_name, result) in enumerate(predictions.items()):
                col_idx = i % 4  # Wrap to new row after 4 columns
                
                with cols[col_idx]:
                    accuracy = result['accuracy']
                    prediction = "Bullish 📈" if result['prediction'] == 1 else "Bearish 📉"
                    confidence = result.get('confidence', 0.5)
                    color = "green" if result['prediction'] == 1 else "red"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{model_name}</h4>
                        <p style="color: {color}; font-size: 18px; font-weight: bold;">{prediction}</p>
                        <p>Accuracy: {accuracy:.2%}</p>
                        <p>Confidence: {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Model comparison table
            st.markdown("##### 📊 Model Performance Comparison")
            model_data = {
                'Model': list(predictions.keys()),
                'Accuracy': [f"{pred['accuracy']:.2%}" for pred in predictions.values()],
                'Prediction': ["Bullish 📈" if pred['prediction'] == 1 else "Bearish 📉" for pred in predictions.values()],
                'Confidence': [f"{pred.get('confidence', 0.5):.2%}" for pred in predictions.values()]
            }
            model_df = create_safe_dataframe(model_data)
            st.dataframe(model_df, width='stretch')
            
            # Enhanced explanation
            st.markdown("##### 🔍 Enhanced Prediction Methodology")
            st.info("""
            **Our enhanced ML models now use:**
            - **Comprehensive Historical Data**: Price patterns, volume analysis, technical indicators
            - **Advanced Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, ATR, Williams %R
            - **Option Chain Analytics**: PCR, IV skew, max pain, option flow (when available)
            - **Market Microstructure**: Gamma exposure, support/resistance levels
            - **Multiple Time Frames**: Short-term and long-term trend analysis
            
            **Model Types:**
            - **Random Forest**: Ensemble learning with decision trees
            - **Logistic Regression**: Statistical probability model with regularization
            - **Gradient Boosting**: Sequential learning algorithm (for larger datasets)
            - **Ensemble**: Weighted combination of all models for better accuracy
            """)
            
        else:
            st.warning("⚠️ Unable to generate ML predictions.")
            st.info("""
            **This may be due to:**
            - Insufficient historical data points
            - Data quality issues
            - Symbol recently listed
            
            **The system will still provide option-based analysis when available.**
            """)
    
    with tab4:
        st.markdown("### 📋 Enhanced Technical Analysis")
        
        if not hist_data.empty and len(hist_data) > 5:
            # Latest values with enhanced metrics
            st.markdown("##### 📊 Current Technical Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                latest_close = hist_data['Close'].iloc[-1]
                prev_close = hist_data['Close'].iloc[-2] if len(hist_data) > 1 else latest_close
                change = latest_close - prev_close
                st.metric("Current Price", f"₹{latest_close:.2f}", delta=f"{change:.2f}")
            
            with col2:
                if 'RSI' in hist_data.columns:
                    latest_rsi = hist_data['RSI'].iloc[-1]
                    rsi_signal = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
                    st.metric("RSI", f"{latest_rsi:.1f}", delta=rsi_signal)
            
            with col3:
                if 'SMA_20' in hist_data.columns:
                    sma20 = hist_data['SMA_20'].iloc[-1]
                    sma_trend = "Above" if latest_close > sma20 else "Below"
                    distance = abs(latest_close - sma20)
                    st.metric("vs SMA20", sma_trend, delta=f"₹{distance:.2f}")
            
            with col4:
                volatility = hist_data['Close'].pct_change().std() * np.sqrt(252) * 100
                st.metric("Historical Vol", f"{volatility:.1f}%")
            
            with col5:
                if 'ATR' in hist_data.columns:
                    atr = hist_data['ATR'].iloc[-1]
                    st.metric("ATR", f"₹{atr:.2f}")
            
            # Support and resistance levels
            st.markdown("##### 🎯 Technical & Option-Based Levels")
            high_20 = hist_data['High'].rolling(20).max().iloc[-1]
            low_20 = hist_data['Low'].rolling(20).min().iloc[-1]
            
            levels_data = {
                'Level Type': ['20-Day High', '20-Day Low', 'OI Support', 'OI Resistance', 'Max Pain'],
                'Price': [
                    f"₹{high_20:.2f}",
                    f"₹{low_20:.2f}",
                    f"₹{analytics.get('support', 0):.0f}" if analytics.get('support', 0) > 0 else "N/A",
                    f"₹{analytics.get('resistance', 0):.0f}" if analytics.get('resistance', 0) > 0 else "N/A",
                    f"₹{analytics.get('max_pain', 0):.0f}" if analytics.get('max_pain', 0) > 0 else "N/A"
                ],
                'Distance from Current': [
                    f"{((high_20 - latest_close) / latest_close * 100):+.2f}%",
                    f"{((low_20 - latest_close) / latest_close * 100):+.2f}%",
                    f"{((analytics.get('support', latest_close) - latest_close) / latest_close * 100):+.2f}%" if analytics.get('support', 0) > 0 else "N/A",
                    f"{((analytics.get('resistance', latest_close) - latest_close) / latest_close * 100):+.2f}%" if analytics.get('resistance', 0) > 0 else "N/A",
                    f"{((analytics.get('max_pain', latest_close) - latest_close) / latest_close * 100):+.2f}%" if analytics.get('max_pain', 0) > 0 else "N/A"
                ]
            }
            levels_df = create_safe_dataframe(levels_data)
            st.dataframe(levels_df, width='stretch')
            
            # Technical signals summary
            st.markdown("##### 🎯 Technical Signals Summary")
            signals = []
            
            if 'RSI' in hist_data.columns:
                rsi = hist_data['RSI'].iloc[-1]
                if rsi > 70:
                    signals.append("🔴 RSI Overbought - Consider selling")
                elif rsi < 30:
                    signals.append("🟢 RSI Oversold - Consider buying")
                else:
                    signals.append("🟡 RSI Neutral")
            
            if 'MACD' in hist_data.columns and 'MACD_Signal' in hist_data.columns:
                macd = hist_data['MACD'].iloc[-1]
                macd_signal = hist_data['MACD_Signal'].iloc[-1]
                if macd > macd_signal:
                    signals.append("🟢 MACD Bullish crossover")
                else:
                    signals.append("🔴 MACD Bearish crossover")
            
            if analytics.get('pcr', 0) > 0:
                pcr = analytics.get('pcr')
                if pcr > 1.3:
                    signals.append("🟢 High PCR - Bullish sentiment")
                elif pcr < 0.7:
                    signals.append("🔴 Low PCR - Bearish sentiment")
                else:
                    signals.append("🟡 PCR Neutral")
            
            for signal in signals:
                st.write(signal)
                
        else:
            st.warning("⚠️ Insufficient historical data for detailed technical analysis")
            st.info("Try selecting a longer time period or a different symbol")
    
    with tab5:
        st.markdown("### 🎯 Strategy Builder")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            strategy = st.selectbox("Strategy Type", [
                "Long Call", "Long Put", "Short Call", "Short Put",
                "Bull Call Spread", "Bear Put Spread", "Straddle", "Strangle"
            ])
            
            strike_price = st.number_input("Strike Price", value=int(spot_price), step=50)
            premium = st.number_input("Premium", value=100.0, step=10.0)
            quantity = st.number_input("Quantity", value=1, step=1)
        
        with col2:
            st.markdown("##### 📊 Strategy Metrics")
            
            # Calculate Greeks
            try:
                greeks = option_greeks(
                    strike=strike_price,
                    premium=premium,
                    spot_price=spot_price,
                    iv=analytics.get('iv_atm', 20),
                    risk_free_rate=risk_free_rate,
                    days_to_expiry=days_to_expiry
                )
                
                # Display strategy summary
                summary = strategy_summary(strategy, strike_price, premium, quantity, greeks)
                
                # Format summary for display
                summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
                summary_df = ensure_dataframe_types(summary_df)
                st.table(summary_df)
                
            except Exception as e:
                st.error(f"Error calculating strategy metrics: {e}")
                st.info("Please check your inputs and try again.")
        
        # Plot strategy payoff
        try:
            st.markdown("##### 📈 Strategy Payoff Diagram")
            fig = plot_strategy(strategy, strike_price, premium, quantity, spot_price)
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.error(f"Error generating payoff diagram: {e}")
    
    with tab6:
        st.markdown("### 💾 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📋 Export Options")
            
            export_format = st.radio("Export Format:", ["Excel", "CSV"])
            include_charts = st.checkbox("Include Charts (Excel only)", value=True)
            include_historical = st.checkbox("Include Historical Data", value=True)
            
            if st.button("💾 Export Data"):
                with st.spinner("Preparing export file..."):
                    try:
                        if export_format == "Excel":
                            # Prepare charts for export
                            charts = {}
                            if include_charts and not df.empty:
                                charts = {
                                    "OI Distribution": plot_oi_distribution(df, spot_price),
                                    "OI Change": plot_oi_change(df, spot_price),
                                    "IV Surface": plot_iv_surface(df, spot_price)
                                }
                                
                                if not hist_data.empty:
                                    charts["Technical Indicators"] = plot_technical_indicators(hist_data)
                            
                            filename = export_to_excel(df, analytics, symbol, charts)
                            st.success(f"✅ Data exported successfully!")
                            
                            # Download link
                            with open(filename, "rb") as f:
                                bytes_data = f.read()
                            st.download_button(
                                label="📥 Download Excel File",
                                data=bytes_data,
                                file_name=os.path.basename(filename),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:  # CSV
                            if not df.empty:
                                csv_data = df.to_csv(index=False)
                            else:
                                csv_data = hist_data.to_csv(index=True) if not hist_data.empty else "No data available"
                                
                            st.download_button(
                                label="📥 Download CSV File",
                                data=csv_data,
                                file_name=f"{symbol}_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            st.success("✅ CSV file ready for download!")
                    except Exception as e:
                        st.error(f"❌ Export failed: {e}")
        
        with col2:
            st.markdown("##### 📋 Export Contents")
            st.info("""
            **Excel Export Includes:**
            - Option chain data (if available)
            - Historical price data
            - Analytics summary with key metrics
            - Interactive charts as images (if selected)
            - Technical indicators
            - Professional formatting
            
            **CSV Export Includes:**
            - Raw option chain data OR historical data
            - Compatible with Excel and other analysis tools
            
            **Enhanced Features:**
            - TCS and all major stocks supported
            - Comprehensive technical analysis
            - Multiple timeframe data
            """)
    
    # NEW: MORE ANALYSIS TAB
    with tab7:
        st.markdown("### 🔍 Advanced Analytics")
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### 📈 Volatility Analysis")
            
            if not df.empty:
                # Calculate volatility metrics
                volatility_metrics = calculate_volatility_metrics(df, hist_data)
                
                if volatility_metrics:
                    # IV Percentiles
                    if 'call_iv_25pct' in volatility_metrics:
                        iv_data = {
                            'Metric': ['25th Percentile', 'Median', '75th Percentile'],
                            'Call IV': [
                                f"{volatility_metrics.get('call_iv_25pct', 0):.2f}%", 
                                f"{volatility_metrics.get('call_iv_median', 0):.2f}%", 
                                f"{volatility_metrics.get('call_iv_75pct', 0):.2f}%"
                            ],
                            'Put IV': [
                                f"{volatility_metrics.get('put_iv_25pct', 0):.2f}%", 
                                f"{volatility_metrics.get('put_iv_median', 0):.2f}%", 
                                f"{volatility_metrics.get('put_iv_75pct', 0):.2f}%"
                            ]
                        }
                        iv_df = create_safe_dataframe(iv_data)
                        st.dataframe(iv_df, width='stretch')
                    
                    # IV Rank
                    if 'iv_rank' in volatility_metrics:
                        iv_rank = volatility_metrics['iv_rank']
                        st.metric("IV Rank", f"{iv_rank:.1f}%", 
                                 delta="High" if iv_rank > 70 else "Low" if iv_rank < 30 else "Medium")
            
            st.markdown("##### 🎯 Gamma Exposure Analysis")
            
            if not df.empty:
                gamma_metrics = calculate_gamma_exposure(df, spot_price)
                
                if gamma_metrics:
                    if 'peak_gamma_strike' in gamma_metrics:
                        st.metric("Peak Gamma Strike", f"₹{gamma_metrics.get('peak_gamma_strike', 0):,.0f}")
                    
                    if 'atm_gamma_concentration' in gamma_metrics:
                        atm_gamma_pct = gamma_metrics.get('atm_gamma_concentration', 0)
                        st.metric("ATM Gamma Concentration", f"{atm_gamma_pct:.1f}%")
        
        with col2:
            st.markdown("##### 📊 Option Flow Analysis")
            
            if not df.empty:
                flow_metrics = calculate_option_flow(df)
                
                if flow_metrics:
                    # Largest OI changes
                    if 'largest_call_oi_change' in flow_metrics:
                        flow_data = {
                            'Flow Type': ['Largest Call OI Increase', 'Largest Put OI Increase'],
                            'Strike': [
                                f"₹{flow_metrics.get('largest_call_oi_strike', 0):,.0f}", 
                                f"₹{flow_metrics.get('largest_put_oi_strike', 0):,.0f}"
                            ],
                            'Change': [
                                f"{flow_metrics.get('largest_call_oi_change', 0):,.0f}", 
                                f"{flow_metrics.get('largest_put_oi_change', 0):,.0f}"
                            ]
                        }
                        flow_df = create_safe_dataframe(flow_data)
                        st.dataframe(flow_df, width='stretch')
                    
                    # Volume PCR
                    if 'volume_pcr' in flow_metrics:
                        volume_pcr = flow_metrics.get('volume_pcr', 0)
                        st.metric("Volume PCR", f"{volume_pcr:.2f}")
        
        # Advanced technical analysis
        st.markdown("##### 🔍 Advanced Technical Indicators")
        
        if not hist_data.empty and len(hist_data) > 20:
            # Calculate ADX
            hist_data_adx = calculate_adx(hist_data)
            
            if not hist_data_adx.empty and 'ADX' in hist_data_adx.columns:
                # Display current values
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_adx = hist_data_adx['ADX'].iloc[-1]
                    adx_strength = "Strong Trend" if current_adx > 25 else "Weak Trend"
                    st.metric("ADX", f"{current_adx:.1f}", delta=adx_strength)
                
                with col2:
                    if 'Plus_DI' in hist_data_adx.columns and 'Minus_DI' in hist_data_adx.columns:
                        current_plus_di = hist_data_adx['Plus_DI'].iloc[-1]
                        current_minus_di = hist_data_adx['Minus_DI'].iloc[-1]
                        di_signal = "Bullish" if current_plus_di > current_minus_di else "Bearish"
                        st.metric("DI Signal", di_signal)
                
                with col3:
                    if 'RSI' in hist_data_adx.columns:
                        current_rsi = hist_data_adx['RSI'].iloc[-1]
                        rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                        st.metric("RSI", f"{current_rsi:.1f}", delta=rsi_signal)
                
                # Plot ADX
                fig_adx = go.Figure()
                
                fig_adx.add_trace(go.Scatter(
                    x=hist_data_adx.index, 
                    y=hist_data_adx['ADX'],
                    name='ADX', 
                    line=dict(color='rgb(52, 152, 219)', width=2)
                ))
                
                if 'Plus_DI' in hist_data_adx.columns:
                    fig_adx.add_trace(go.Scatter(
                        x=hist_data_adx.index, 
                        y=hist_data_adx['Plus_DI'],
                        name='+DI', 
                        line=dict(color='rgb(46, 204, 113)', width=2)
                    ))
                
                if 'Minus_DI' in hist_data_adx.columns:
                    fig_adx.add_trace(go.Scatter(
                        x=hist_data_adx.index, 
                        y=hist_data_adx['Minus_DI'],
                        name='-DI', 
                        line=dict(color='rgb(231, 76, 60)', width=2)
                    ))
                
                fig_adx.update_layout(
                    title="Average Directional Index (ADX)",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    height=400,
                    showlegend=True,
                    plot_bgcolor='rgba(248, 249, 250, 0.8)',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig_adx, width='stretch')
        
        # Market profile and value area analysis
        st.markdown("##### 📊 Market Profile Analysis")
        
        if not hist_data.empty and len(hist_data) > 5:
            market_profile = calculate_market_profile(hist_data)
            
            if market_profile:
                profile_data = {
                    'Profile Element': ['Point of Control (POC)', 'Value Area High', 'Value Area Low', 'Value Area Width'],
                    'Value': [
                        f"₹{market_profile.get('poc', 0):.2f}",
                        f"₹{market_profile.get('value_area_high', 0):.2f}",
                        f"₹{market_profile.get('value_area_low', 0):.2f}",
                        f"₹{market_profile.get('value_area_width', 0):.2f}"
                    ]
                }
                profile_df = create_safe_dataframe(profile_data)
                st.dataframe(profile_df, width='stretch')
                
                # Check if current price is in value area
                current_price = hist_data['Close'].iloc[-1]
                in_value_area = market_profile.get('price_in_value_area', False)
                st.metric("Price in Value Area", "Yes" if in_value_area else "No")
        
        # Seasonality analysis (if we have enough historical data)
        if not hist_data.empty and len(hist_data) > 252:  # At least 1 year of data
            st.markdown("##### 📅 Seasonal Patterns")
            
            seasonality = calculate_seasonality(hist_data)
            
            if seasonality:
                # Analyze performance by day of week
                daily_data = {
                    'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                    'Avg Return': [
                        f"{seasonality.get('monday_avg_return', 0):.4f}%" if 'monday_avg_return' in seasonality else "N/A",
                        f"{seasonality.get('tuesday_avg_return', 0):.4f}%" if 'tuesday_avg_return' in seasonality else "N/A",
                        f"{seasonality.get('wednesday_avg_return', 0):.4f}%" if 'wednesday_avg_return' in seasonality else "N/A",
                        f"{seasonality.get('thursday_avg_return', 0):.4f}%" if 'thursday_avg_return' in seasonality else "N/A",
                        f"{seasonality.get('friday_avg_return', 0):.4f}%" if 'friday_avg_return' in seasonality else "N/A"
                    ]
                }
                daily_df = create_safe_dataframe(daily_data)
                st.dataframe(daily_df, width='stretch')
                
                # Best and worst performing days
                if 'best_day' in seasonality and 'worst_day' in seasonality:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Best Day", seasonality.get('best_day', 'N/A'))
                    with col2:
                        st.metric("Worst Day", seasonality.get('worst_day', 'N/A'))
    
    with tab8:
        create_intraday_analysis_tab(hist_data, symbol)
    
    with tab9:
        create_sector_analysis_tab(hist_data, symbol)

    with tab10:
        create_market_sentiment_tab()

    with tab11:
        create_decision_dashboard(symbol, hist_data, analytics, predictions)
        
        # Add advanced analytics as an expander in the decision dashboard
        with st.expander("Advanced Statistical Analysis"):
            create_advanced_analytics_tab(hist_data, symbol)

    # Footer with enhanced info
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        st.markdown(f"**Data Sources:** NSE India, Yahoo Finance")
    
    with col3:
        if not hist_data.empty:
            data_points = len(hist_data)
            st.markdown(f"**Data Points:** {data_points}")
    
    with col4:
        if st.button("🔄 Refresh Data"):
            st.rerun()

    with tab12:
        # NEW: Live Dashboard Tab
        create_live_dashboard_tab()

    with tab13:
        # NEW: News & Social Tab
        create_news_social_tab()

    with tab14:
        # NEW: Excel Automation Tab
        create_excel_automation_tab()

    # Footer
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        st.markdown(f"**Data Sources:** NSE India, Yahoo Finance")
    with col3:
        if not hist_data.empty:
            data_points = len(hist_data)
            st.markdown(f"**Data Points:** {data_points}")
    with col4:
        if st.button("🔄 Refresh Data"):
            st.rerun()

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
