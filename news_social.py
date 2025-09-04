# [file name]: news_social.py
# [file content begin]
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import re
import plotly.graph_objects as go  # Added for plotting

def create_news_social_tab():
    """Create a news and social media analysis tab."""
    st.markdown("### 📰 News & Social Media Analysis")
    
    # Fetch news and social media data
    with st.spinner("Gathering latest news and social media trends..."):
        news_data = fetch_news_data()
        social_data = fetch_social_data()
    
    # News analysis
    st.markdown("##### 📊 News Sentiment Analysis")
    
    if news_data:
        # Calculate sentiment scores
        positive_news = sum(1 for item in news_data if item.get('sentiment', 0) > 0)
        negative_news = sum(1 for item in news_data if item.get('sentiment', 0) < 0)
        neutral_news = sum(1 for item in news_data if item.get('sentiment', 0) == 0)
        
        total_news = len(news_data)
        sentiment_score = (positive_news - negative_news) / total_news if total_news > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total News", total_news)
        with col2:
            st.metric("Positive News", positive_news)
        with col3:
            st.metric("Negative News", negative_news)
        with col4:
            st.metric("Sentiment Score", f"{sentiment_score:.2f}")
        
        # Display news with sentiment
        st.markdown("##### 📋 Latest News with Sentiment")
        
        for item in news_data[:10]:
            sentiment = item.get('sentiment', 0)
            if sentiment > 0:
                sentiment_icon = "🟢"
            elif sentiment < 0:
                sentiment_icon = "🔴"
            else:
                sentiment_icon = "🟡"
            
            with st.expander(f"{sentiment_icon} {item.get('title', '')} - {item.get('source', '')}"):
                st.write(f"**Published:** {item.get('time', '')}")
                st.write(item.get('summary', ''))
                if item.get('url'):
                    st.markdown(f"[Read full article]({item.get('url')})")
    else:
        st.info("News data not available at the moment")
    
    # Social media analysis
    st.markdown("##### 📱 Social Media Trends")
    
    if social_data:
        # Top trending topics
        st.markdown("**🔥 Top Trending Topics**")
        
        topics = social_data.get('trending_topics', [])
        if topics:
            for i, topic in enumerate(topics[:10]):
                st.write(f"{i+1}. {topic.get('topic', '')} ({topic.get('mentions', 0)} mentions)")
        else:
            st.info("No trending topics data available")
        
        # Social media sentiment by platform
        st.markdown("**📊 Platform-wise Sentiment**")
        
        platforms = social_data.get('platform_sentiment', {})
        if platforms:
            platform_df = pd.DataFrame.from_dict(platforms, orient='index', columns=['Sentiment'])
            platform_df = platform_df.sort_values('Sentiment', ascending=False)
            
            # Display sentiment bars
            for platform, sentiment in platform_df.iterrows():
                sentiment_value = sentiment['Sentiment']
                color = "green" if sentiment_value > 0 else "red" if sentiment_value < 0 else "gray"
                
                st.write(f"{platform}:")
                st.progress(abs(sentiment_value), text=f"{sentiment_value:.2f}")
        else:
            st.info("Platform sentiment data not available")
        
        # Influencer opinions
        st.markdown("**👑 Influencer Opinions**")
        
        influencers = social_data.get('influencers', [])
        if influencers:
            for influencer in influencers[:5]:
                st.write(f"**{influencer.get('name', '')}** ({influencer.get('followers', 0):,} followers)")
                st.write(f"*{influencer.get('opinion', '')}*")
                st.write(f"Sentiment: {influencer.get('sentiment', 0):.2f}")
                st.markdown("---")
        else:
            st.info("Influencer data not available")
    else:
        st.info("Social media data not available at the moment")
    
    # Combined sentiment analysis
    st.markdown("##### 📈 Combined Market Sentiment")
    
    # Calculate overall sentiment
    news_sentiment = sum(item.get('sentiment', 0) for item in news_data) / len(news_data) if news_data else 0
    social_sentiment = social_data.get('overall_sentiment', 0) if social_data else 0
    
    overall_sentiment = (news_sentiment + social_sentiment) / 2
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("News Sentiment", f"{news_sentiment:.2f}")
    with col2:
        st.metric("Social Sentiment", f"{social_sentiment:.2f}")
    with col3:
        st.metric("Overall Sentiment", f"{overall_sentiment:.2f}")
    
    # Sentiment timeline (mock data)
    st.markdown("##### 📅 Sentiment Timeline")
    
    dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
    sentiment_values = np.random.uniform(-1, 1, 7)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=sentiment_values,
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="7-Day Sentiment Trend",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        height=300
    )
    
    st.plotly_chart(fig, width='stretch')  # Updated for Streamlit deprecation
    
    # Market impact prediction
    st.markdown("##### 🔮 Predicted Market Impact")
    
    if overall_sentiment > 0.3:
        st.success("📈 Strong positive sentiment detected. Likely bullish impact on the market.")
    elif overall_sentiment > 0.1:
        st.info("📈 Moderate positive sentiment detected. Potential slight upward movement.")
    elif overall_sentiment < -0.3:
        st.error("📉 Strong negative sentiment detected. Likely bearish impact on the market.")
    elif overall_sentiment < -0.1:
        st.warning("📉 Moderate negative sentiment detected. Potential slight downward movement.")
    else:
        st.info("➡️ Neutral sentiment detected. Market likely to remain range-bound.")

def fetch_news_data():
    """Fetch news data from various sources."""
    # Mock implementation - in a real app, you would use news APIs
    
    news_items = [
        {
            'title': 'RBI Maintains Accommodative Stance, Markets React Positively',
            'summary': 'The Reserve Bank of India has decided to maintain its accommodative monetary policy stance, which has been well received by the markets.',
            'source': 'Economic Times',
            'time': '2 hours ago',
            'url': '#',
            'sentiment': 0.8
        },
        {
            'title': 'Corporate Earnings Season Begins with Mixed Results',
            'summary': 'The Q2 earnings season has kicked off with some companies exceeding expectations while others have reported disappointing results.',
            'source': 'Business Standard',
            'time': '4 hours ago',
            'url': '#',
            'sentiment': 0.1
        },
        {
            'title': 'Global Economic Slowdown Concerns Weigh on Markets',
            'summary': 'Growing concerns about a potential global economic slowdown have led to increased volatility in financial markets worldwide.',
            'source': 'Reuters',
            'time': '6 hours ago',
            'url': '#',
            'sentiment': -0.7
        },
        {
            'title': 'Government Announces New Infrastructure Projects',
            'summary': 'The government has unveiled a new package of infrastructure projects aimed at boosting economic growth and creating jobs.',
            'source': 'Financial Express',
            'time': '8 hours ago',
            'url': '#',
            'sentiment': 0.6
        },
        {
            'title': 'IT Sector Faces Headwinds from Global Uncertainty',
            'summary': 'The IT sector is experiencing challenges due to global economic uncertainty and reduced technology spending by clients.',
            'source': 'Moneycontrol',
            'time': '10 hours ago',
            'url': '#',
            'sentiment': -0.5
        },
        {
            'title': 'Monsoon Progress Better Than Expected, Rural Stocks Rally',
            'summary': 'Better-than-expected monsoon progress has boosted agricultural prospects, leading to a rally in rural-focused stocks.',
            'source': 'Bloomberg',
            'time': '12 hours ago',
            'url': '#',
            'sentiment': 0.7
        },
        {
            'title': 'Oil Prices Volatile Amid Supply Concerns',
            'summary': 'Crude oil prices have been volatile due to ongoing supply concerns and changing demand forecasts.',
            'source': 'CNBC',
            'time': 'Yesterday',
            'url': '#',
            'sentiment': -0.3
        },
        {
            'title': 'Banking Sector Shows Resilience in Stress Tests',
            'summary': 'Recent stress tests have shown that the banking sector remains resilient despite economic challenges.',
            'source': 'LiveMint',
            'time': 'Yesterday',
            'url': '#',
            'sentiment': 0.4
        }
    ]
    
    return news_items

def fetch_social_data():
    """Fetch social media data and trends."""
    # Mock implementation - in a real app, you would use social media APIs
    
    social_data = {
        'trending_topics': [
            {'topic': '#RBIpolicy', 'mentions': 12500},
            {'topic': '#Q2Earnings', 'mentions': 9800},
            {'topic': '#InfrastructureBoost', 'mentions': 7600},
            {'topic': '#MarketVolatility', 'mentions': 6500},
            {'topic': '#Monsoon2023', 'mentions': 5400},
            {'topic': '#OilPrices', 'mentions': 4300},
            {'topic': '#BankingSector', 'mentions': 3200},
            {'topic': '#ITStocks', 'mentions': 2100}
        ],
        'platform_sentiment': {
            'Twitter': 0.6,
            'Reddit': 0.3,
            'LinkedIn': 0.7,
            'Facebook': 0.4,
            'YouTube': 0.5
        },
        'influencers': [
            {
                'name': 'Market Expert',
                'followers': 250000,
                'opinion': 'The RBI policy decision is a positive move that will support economic growth in the medium term.',
                'sentiment': 0.8
            },
            {
                'name': 'Finance Guru',
                'followers': 180000,
                'opinion': 'Q2 earnings are showing mixed signals, suggesting caution is warranted in the current market environment.',
                'sentiment': -0.2
            },
            {
                'name': 'Investment Analyst',
                'followers': 120000,
                'opinion': 'Infrastructure stocks look attractive given the government\'s renewed focus on capital expenditure.',
                'sentiment': 0.7
            },
            {
                'name': 'Economic Strategist',
                'followers': 95000,
                'opinion': 'Global economic headwinds could impact export-oriented sectors in the coming quarters.',
                'sentiment': -0.5
            },
            {
                'name': 'Technical Analyst',
                'followers': 80000,
                'opinion': 'The Nifty is showing resilience around key support levels, suggesting limited downside from current levels.',
                'sentiment': 0.4
            }
        ],
        'overall_sentiment': 0.35
    }
    
    return social_data
# [file content end]
