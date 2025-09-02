# strategies.py - Complete Fixed Version
# Professional options trading strategies with Greeks calculation and visualization
# for NSE Option Chain Analytics Pro

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from datetime import datetime, timedelta
import math

# ---------------- OPTION GREEKS CALCULATION ----------------

def option_greeks(strike, premium, spot_price, iv, risk_free_rate=0.05, days_to_expiry=30):
    """
    Calculate option Greeks using Black-Scholes model with comprehensive error handling.
    
    Parameters:
    - strike: Strike price of the option
    - premium: Premium paid/received for the option
    - spot_price: Current underlying price
    - iv: Implied volatility (as decimal, e.g., 0.20 for 20%)
    - risk_free_rate: Risk-free interest rate (as decimal)
    - days_to_expiry: Days until expiration
    
    Returns:
    - Dictionary with all Greeks for both calls and puts
    """
    # Convert inputs to appropriate types and handle edge cases
    try:
        strike = float(strike)
        premium = float(premium)
        spot_price = float(spot_price)
        iv = float(iv) / 100 if iv > 1 else float(iv)  # Handle percentage input
        risk_free_rate = float(risk_free_rate)
        days_to_expiry = float(days_to_expiry)
    except (ValueError, TypeError):
        return {
            'call_delta': 0.0, 'put_delta': 0.0, 'gamma': 0.0,
            'call_theta': 0.0, 'put_theta': 0.0, 'vega': 0.0,
            'call_rho': 0.0, 'put_rho': 0.0, 'delta': 0.0,
            'gamma': 0.0, 'theta': 0.0, 'vega': 0.0
        }
    
    # Time to expiration in years
    T = days_to_expiry / 365.0
    
    # Avoid division by zero and invalid inputs
    if T <= 0 or iv <= 0 or strike <= 0 or spot_price <= 0:
        return {
            'call_delta': 0.0, 'put_delta': 0.0, 'gamma': 0.0,
            'call_theta': 0.0, 'put_theta': 0.0, 'vega': 0.0,
            'call_rho': 0.0, 'put_rho': 0.0, 'delta': 0.0,
            'gamma': 0.0, 'theta': 0.0, 'vega': 0.0
        }
    
    try:
        # Calculate d1 and d2
        d1 = (np.log(spot_price / strike) + (risk_free_rate + (iv ** 2) / 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        
        # Calculate Greeks for calls
        call_delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (spot_price * iv * np.sqrt(T))
        call_theta = ((-spot_price * norm.pdf(d1) * iv) / (2 * np.sqrt(T)) - 
                     risk_free_rate * strike * np.exp(-risk_free_rate * T) * norm.cdf(d2))
        vega = spot_price * norm.pdf(d1) * np.sqrt(T)
        call_rho = strike * T * np.exp(-risk_free_rate * T) * norm.cdf(d2)
        
        # Calculate Greeks for puts
        put_delta = call_delta - 1
        put_theta = ((-spot_price * norm.pdf(d1) * iv) / (2 * np.sqrt(T)) + 
                    risk_free_rate * strike * np.exp(-risk_free_rate * T) * norm.cdf(-d2))
        put_rho = -strike * T * np.exp(-risk_free_rate * T) * norm.cdf(-d2)
        
        return {
            'call_delta': round(call_delta, 4),
            'put_delta': round(put_delta, 4),
            'gamma': round(gamma, 4),
            'call_theta': round(call_theta / 365.0, 4),  # Convert to per day
            'put_theta': round(put_theta / 365.0, 4),    # Convert to per day
            'vega': round(vega / 100.0, 4),              # Per 1% change in IV
            'call_rho': round(call_rho / 100.0, 4),      # Per 1% change in rate
            'put_rho': round(put_rho / 100.0, 4),
            # Legacy compatibility
            'delta': round(call_delta, 4),
            'theta': round(call_theta / 365.0, 4),
        }
        
    except Exception as e:
        print(f"Error calculating Greeks: {e}")
        return {
            'call_delta': 0.0, 'put_delta': 0.0, 'gamma': 0.0,
            'call_theta': 0.0, 'put_theta': 0.0, 'vega': 0.0,
            'call_rho': 0.0, 'put_rho': 0.0, 'delta': 0.0,
            'gamma': 0.0, 'theta': 0.0, 'vega': 0.0
        }

# ----------------- PAYOFF CALCULATIONS -----------------

def payoff_option(S, strike, premium, option_type, position):
    """
    Calculates the payoff for a single option contract.
    
    Parameters:
    - S: Array of underlying prices at expiration
    - strike: Strike price
    - premium: Premium paid/received
    - option_type: 'call' or 'put'
    - position: Number of contracts (positive for long, negative for short)
    """
    payoff = np.zeros_like(S)
    
    if option_type.lower() == 'call':
        intrinsic_value = np.maximum(S - strike, 0)
        payoff = (intrinsic_value - premium) * position
    elif option_type.lower() == 'put':
        intrinsic_value = np.maximum(strike - S, 0)
        payoff = (intrinsic_value - premium) * position
    
    return payoff

def calculate_strategy_payoff(strategy, strike, premium, quantity, prices):
    """
    Calculate payoff for different strategy types with enhanced strategies.
    """
    if strategy == "Long Call":
        return payoff_option(prices, strike, premium, 'call', quantity)
    
    elif strategy == "Long Put":
        return payoff_option(prices, strike, premium, 'put', quantity)
    
    elif strategy == "Short Call":
        return payoff_option(prices, strike, premium, 'call', -quantity)
    
    elif strategy == "Short Put":
        return payoff_option(prices, strike, premium, 'put', -quantity)
    
    elif strategy == "Bull Call Spread":
        # Long lower strike call, short higher strike call
        lower_strike = strike
        higher_strike = strike + 50  # Assuming 50 point spread
        long_call = payoff_option(prices, lower_strike, premium, 'call', quantity)
        short_call = payoff_option(prices, higher_strike, premium/2, 'call', -quantity)
        return long_call + short_call
    
    elif strategy == "Bear Put Spread":
        # Long higher strike put, short lower strike put
        higher_strike = strike
        lower_strike = strike - 50  # Assuming 50 point spread
        long_put = payoff_option(prices, higher_strike, premium, 'put', quantity)
        short_put = payoff_option(prices, lower_strike, premium/2, 'put', -quantity)
        return long_put + short_put
    
    elif strategy == "Straddle":
        # Long call and long put at same strike
        call_payoff = payoff_option(prices, strike, premium/2, 'call', quantity)
        put_payoff = payoff_option(prices, strike, premium/2, 'put', quantity)
        return call_payoff + put_payoff
    
    elif strategy == "Strangle":
        # Long call at higher strike, long put at lower strike
        call_strike = strike + 50
        put_strike = strike - 50
        call_payoff = payoff_option(prices, call_strike, premium/2, 'call', quantity)
        put_payoff = payoff_option(prices, put_strike, premium/2, 'put', quantity)
        return call_payoff + put_payoff
    
    elif strategy == "Iron Condor":
        # Short straddle + long strangle
        # Short call and put at middle strikes
        short_call = payoff_option(prices, strike + 25, premium/4, 'call', -quantity)
        short_put = payoff_option(prices, strike - 25, premium/4, 'put', -quantity)
        # Long call and put at outer strikes  
        long_call = payoff_option(prices, strike + 75, premium/8, 'call', quantity)
        long_put = payoff_option(prices, strike - 75, premium/8, 'put', quantity)
        return short_call + short_put + long_call + long_put
    
    elif strategy == "Butterfly":
        # Long 2 ATM, short 1 ITM and 1 OTM
        long_atm = payoff_option(prices, strike, premium/2, 'call', 2*quantity)
        short_itm = payoff_option(prices, strike - 50, premium/4, 'call', -quantity)
        short_otm = payoff_option(prices, strike + 50, premium/4, 'call', -quantity)
        return long_atm + short_itm + short_otm
    
    else:
        # Default to long call for unsupported strategies
        return payoff_option(prices, strike, premium, 'call', quantity)

# ----------------- PLOT PAYOFF DIAGRAM -----------------

def plot_strategy(strategy_name, strike_price, premium, quantity, current_spot, min_price=None, max_price=None):
    """
    Plots the payoff diagram for the selected strategy with enhanced visualization.
    """
    # Set price range
    if min_price is None:
        min_price = current_spot * 0.8
    if max_price is None:
        max_price = current_spot * 1.2
    
    prices = np.linspace(min_price, max_price, 200)
    payoffs = calculate_strategy_payoff(strategy_name, strike_price, premium, quantity, prices)
    
    # Create the plot
    fig = go.Figure()
    
    # Add payoff line
    fig.add_trace(go.Scatter(
        x=prices, 
        y=payoffs, 
        mode='lines', 
        name='Payoff',
        line=dict(color='#0b69ff', width=3),
        hovertemplate='<b>Price: ₹%{x:.2f}</b><br>P&L: ₹%{y:.2f}<extra></extra>'
    ))
    
    # Add breakeven lines
    breakeven_points = []
    for i in range(len(payoffs)-1):
        if (payoffs[i] <= 0 and payoffs[i+1] >= 0) or (payoffs[i] >= 0 and payoffs[i+1] <= 0):
            breakeven_price = prices[i] + (prices[i+1] - prices[i]) * (-payoffs[i]) / (payoffs[i+1] - payoffs[i])
            breakeven_points.append(breakeven_price)
    
    # Add breakeven vertical lines
    for be_price in breakeven_points:
        fig.add_vline(
            x=be_price, 
            line_width=2, 
            line_dash="dot", 
            line_color="orange",
            annotation_text=f"BE: ₹{be_price:.0f}",
            annotation_position="top"
        )
    
    # Add current spot price line
    fig.add_vline(
        x=current_spot, 
        line_width=2, 
        line_dash="dash", 
        line_color="#ef4444",
        annotation_text="Current Spot", 
        annotation_position="bottom right"
    )
    
    # Add zero line
    fig.add_hline(
        y=0, 
        line_width=2, 
        line_dash="solid", 
        line_color="#888"
    )
    
    # Fill profit/loss areas
    profit_mask = payoffs >= 0
    loss_mask = payoffs < 0
    
    if np.any(profit_mask):
        fig.add_trace(go.Scatter(
            x=prices[profit_mask], 
            y=payoffs[profit_mask],
            fill='tozeroy',
            mode='none',
            name='Profit Zone',
            fillcolor='rgba(40, 167, 69, 0.3)',
            showlegend=False
        ))
    
    if np.any(loss_mask):
        fig.add_trace(go.Scatter(
            x=prices[loss_mask], 
            y=payoffs[loss_mask],
            fill='tozeroy',
            mode='none',
            name='Loss Zone',
            fillcolor='rgba(220, 53, 69, 0.3)',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{strategy_name} Strategy Payoff Diagram',
        xaxis_title='Underlying Price at Expiry (₹)',
        yaxis_title='Profit / Loss (₹)',
        legend=dict(x=0.01, y=0.99),
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    return fig

# ----------------- STRATEGY SUMMARY -----------------

def strategy_summary(strategy_name, strike_price, premium, quantity, greeks):
    """
    Provides a comprehensive summary table for a given strategy with risk metrics.
    """
    summary = {}
    
    # Greeks summary with proper formatting
    if "Call" in strategy_name or strategy_name in ["Long Call", "Short Call", "Bull Call Spread"]:
        summary['Delta'] = f"{greeks.get('call_delta', 0):.4f}"
        summary['Theta (Daily)'] = f"{greeks.get('call_theta', 0):.4f}"
        summary['Rho'] = f"{greeks.get('call_rho', 0):.4f}"
    else:
        summary['Delta'] = f"{greeks.get('put_delta', 0):.4f}"
        summary['Theta (Daily)'] = f"{greeks.get('put_theta', 0):.4f}"
        summary['Rho'] = f"{greeks.get('put_rho', 0):.4f}"
    
    summary['Gamma'] = f"{greeks.get('gamma', 0):.4f}"
    summary['Vega'] = f"{greeks.get('vega', 0):.4f}"
    
    # Calculate comprehensive PnL metrics for each strategy
    total_premium = premium * quantity
    
    if strategy_name == "Long Call":
        max_profit = "Unlimited"
        max_loss = f"₹{total_premium:,.2f}"
        breakeven = f"₹{strike_price + premium:,.2f}"
        
    elif strategy_name == "Long Put":
        max_profit = f"₹{(strike_price - premium) * quantity:,.2f}"
        max_loss = f"₹{total_premium:,.2f}"
        breakeven = f"₹{strike_price - premium:,.2f}"
        
    elif strategy_name == "Short Call":
        max_profit = f"₹{total_premium:,.2f}"
        max_loss = "Unlimited"
        breakeven = f"₹{strike_price + premium:,.2f}"
        
    elif strategy_name == "Short Put":
        max_profit = f"₹{total_premium:,.2f}"
        max_loss = f"₹{(strike_price - premium) * quantity:,.2f}"
        breakeven = f"₹{strike_price - premium:,.2f}"
        
    elif strategy_name == "Bull Call Spread":
        net_premium = premium - (premium / 2)  # Assuming we pay net premium
        spread_width = 50  # Assuming 50 point spread
        max_profit = f"₹{(spread_width - net_premium) * quantity:,.2f}"
        max_loss = f"₹{net_premium * quantity:,.2f}"
        breakeven = f"₹{strike_price + net_premium:,.2f}"
        
    elif strategy_name == "Bear Put Spread":
        net_premium = premium - (premium / 2)  # Assuming we pay net premium
        spread_width = 50  # Assuming 50 point spread
        max_profit = f"₹{(spread_width - net_premium) * quantity:,.2f}"
        max_loss = f"₹{net_premium * quantity:,.2f}"
        breakeven = f"₹{strike_price - net_premium:,.2f}"
        
    elif strategy_name == "Straddle":
        total_premium_straddle = premium * quantity  # Total premium for both legs
        max_profit = "Unlimited"
        max_loss = f"₹{total_premium_straddle:,.2f}"
        breakeven = f"₹{strike_price - premium:,.2f} & ₹{strike_price + premium:,.2f}"
        
    elif strategy_name == "Strangle":
        call_strike = strike_price + 50
        put_strike = strike_price - 50
        total_premium_strangle = premium * quantity
        max_profit = "Unlimited"
        max_loss = f"₹{total_premium_strangle:,.2f}"
        breakeven_up = f"₹{call_strike + premium:,.2f}"
        breakeven_down = f"₹{put_strike - premium:,.2f}"
        breakeven = f"{breakeven_up} & {breakeven_down}"
        
    elif strategy_name == "Iron Condor":
        net_credit = (premium/4 + premium/4) - (premium/8 + premium/8)  # Net credit received
        max_profit = f"₹{net_credit * quantity:,.2f}"
        max_loss = f"₹{(50 - net_credit) * quantity:,.2f}"  # Assuming 50 point wing width
        breakeven = f"₹{strike_price - 25 + net_credit:,.2f} & ₹{strike_price + 25 - net_credit:,.2f}"
        
    elif strategy_name == "Butterfly":
        net_premium = premium - (premium/2)  # Net premium paid
        max_profit = f"₹{(50 - net_premium) * quantity:,.2f}"
        max_loss = f"₹{net_premium * quantity:,.2f}"
        breakeven = f"₹{strike_price - 50 + net_premium:,.2f} & ₹{strike_price + 50 - net_premium:,.2f}"
        
    else:
        max_profit = "Varies"
        max_loss = "Varies"
        breakeven = "Varies"
    
    # Add strategy-specific metrics
    summary['Strategy Type'] = strategy_name
    summary['Strike Price'] = f"₹{strike_price:,.2f}"
    summary['Premium'] = f"₹{premium:.2f}"
    summary['Quantity'] = str(quantity)
    summary['Max Profit'] = max_profit
    summary['Max Loss'] = max_loss
    summary['Breakeven'] = breakeven
    
    # Risk metrics
    summary['Capital Required'] = f"₹{abs(total_premium):,.2f}"
    
    # Time decay impact
    theta_impact = greeks.get('call_theta', greeks.get('put_theta', 0))
    if theta_impact != 0:
        days_to_lose_10pct = abs(total_premium * 0.1 / (theta_impact * quantity)) if theta_impact != 0 else float('inf')
        if days_to_lose_10pct < 365:
            summary['Days to Lose 10%'] = f"{days_to_lose_10pct:.0f} days"
        else:
            summary['Days to Lose 10%'] = "N/A"
    
    return summary

# ----------------- ADDITIONAL UTILITY FUNCTIONS -----------------

def calculate_option_fair_value(strike, spot_price, iv, risk_free_rate, days_to_expiry, option_type='call'):
    """
    Calculate theoretical option fair value using Black-Scholes model.
    """
    try:
        T = days_to_expiry / 365.0
        iv = iv / 100 if iv > 1 else iv  # Handle percentage input
        
        if T <= 0 or iv <= 0 or strike <= 0 or spot_price <= 0:
            return 0.0
        
        d1 = (np.log(spot_price / strike) + (risk_free_rate + (iv ** 2) / 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        
        if option_type.lower() == 'call':
            fair_value = spot_price * norm.cdf(d1) - strike * np.exp(-risk_free_rate * T) * norm.cdf(d2)
        else:  # put
            fair_value = strike * np.exp(-risk_free_rate * T) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        
        return max(fair_value, 0.0)
        
    except Exception as e:
        print(f"Error calculating fair value: {e}")
        return 0.0

def implied_volatility_newton_raphson(market_price, strike, spot_price, risk_free_rate, days_to_expiry, option_type='call', tolerance=1e-6, max_iterations=100):
    """
    Calculate implied volatility using Newton-Raphson method.
    """
    try:
        T = days_to_expiry / 365.0
        
        if T <= 0 or strike <= 0 or spot_price <= 0 or market_price <= 0:
            return 0.0
        
        # Initial guess
        iv = 0.3
        
        for _ in range(max_iterations):
            # Calculate theoretical price and vega
            d1 = (np.log(spot_price / strike) + (risk_free_rate + (iv ** 2) / 2) * T) / (iv * np.sqrt(T))
            d2 = d1 - iv * np.sqrt(T)
            
            if option_type.lower() == 'call':
                theoretical_price = spot_price * norm.cdf(d1) - strike * np.exp(-risk_free_rate * T) * norm.cdf(d2)
            else:  # put
                theoretical_price = strike * np.exp(-risk_free_rate * T) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
            
            # Calculate vega (sensitivity to volatility)
            vega = spot_price * norm.pdf(d1) * np.sqrt(T)
            
            if abs(vega) < tolerance:
                break
            
            # Newton-Raphson update
            price_diff = theoretical_price - market_price
            iv = iv - price_diff / vega
            
            # Ensure IV stays positive
            iv = max(iv, 0.001)
            
            if abs(price_diff) < tolerance:
                break
        
        return iv * 100  # Return as percentage
        
    except Exception as e:
        print(f"Error calculating implied volatility: {e}")
        return 0.0

def generate_strategy_recommendations(df, analytics, spot_price):
    """
    Generate intelligent strategy recommendations based on market conditions.
    """
    recommendations = []
    
    if not analytics:
        return recommendations
    
    pcr = analytics.get('pcr', 1.0)
    iv_skew = analytics.get('iv_skew', 0.0)
    iv_atm = analytics.get('iv_atm', 20.0)
    directional_bias = analytics.get('directional_bias', 'Neutral')
    
    # Market direction-based strategies
    if pcr > 1.3:
        recommendations.append({
            'strategy': 'Bull Call Spread',
            'reason': f'High PCR ({pcr:.2f}) indicates strong bullish sentiment',
            'confidence': 'High',
            'risk_level': 'Medium'
        })
    elif pcr < 0.7:
        recommendations.append({
            'strategy': 'Bear Put Spread',
            'reason': f'Low PCR ({pcr:.2f}) indicates bearish sentiment',
            'confidence': 'High',
            'risk_level': 'Medium'
        })
    
    # Volatility-based strategies
    if iv_atm > 30:
        recommendations.append({
            'strategy': 'Short Straddle',
            'reason': f'High IV ({iv_atm:.2f}%) suggests volatility selling opportunity',
            'confidence': 'Medium',
            'risk_level': 'High'
        })
    elif iv_atm < 15:
        recommendations.append({
            'strategy': 'Long Straddle',
            'reason': f'Low IV ({iv_atm:.2f}%) suggests volatility buying opportunity',
            'confidence': 'Medium',
            'risk_level': 'Medium'
        })
    
    # Skew-based strategies
    if abs(iv_skew) > 3:
        if iv_skew > 0:
            recommendations.append({
                'strategy': 'Put Spread',
                'reason': f'Positive skew ({iv_skew:.2f}%) suggests puts are expensive relative to calls',
                'confidence': 'Low',
                'risk_level': 'Medium'
            })
        else:
            recommendations.append({
                'strategy': 'Call Spread',
                'reason': f'Negative skew ({iv_skew:.2f}%) suggests calls are expensive relative to puts',
                'confidence': 'Low',
                'risk_level': 'Medium'
            })
    
    # Range-bound market strategies
    if directional_bias == 'Neutral' and iv_atm > 20:
        recommendations.append({
            'strategy': 'Iron Condor',
            'reason': 'Neutral market with elevated volatility favors premium selling strategies',
            'confidence': 'Medium',
            'risk_level': 'Medium'
        })
    
    return recommendations

def calculate_portfolio_greeks(positions):
    """
    Calculate portfolio-level Greeks for multiple option positions.
    
    Parameters:
    - positions: List of dictionaries with position details
    """
    portfolio_greeks = {
        'total_delta': 0.0,
        'total_gamma': 0.0,
        'total_theta': 0.0,
        'total_vega': 0.0,
        'total_rho': 0.0
    }
    
    for position in positions:
        try:
            greeks = option_greeks(
                position.get('strike', 0),
                position.get('premium', 0),
                position.get('spot_price', 0),
                position.get('iv', 0.2),
                position.get('risk_free_rate', 0.05),
                position.get('days_to_expiry', 30)
            )
            
            quantity = position.get('quantity', 0)
            option_type = position.get('type', 'call').lower()
            
            if option_type == 'call':
                portfolio_greeks['total_delta'] += greeks['call_delta'] * quantity
                portfolio_greeks['total_theta'] += greeks['call_theta'] * quantity
                portfolio_greeks['total_rho'] += greeks['call_rho'] * quantity
            else:
                portfolio_greeks['total_delta'] += greeks['put_delta'] * quantity
                portfolio_greeks['total_theta'] += greeks['put_theta'] * quantity
                portfolio_greeks['total_rho'] += greeks['put_rho'] * quantity
            
            portfolio_greeks['total_gamma'] += greeks['gamma'] * quantity
            portfolio_greeks['total_vega'] += greeks['vega'] * quantity
            
        except Exception as e:
            print(f"Error calculating position Greeks: {e}")
            continue
    
    return portfolio_greeks