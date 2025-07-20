import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime
import os
import requests
from textblob import TextBlob
import time
import json
import traceback

def get_last_valid(series, fallback):
    """Return the last valid (non-NaN) scalar value of a Series, or fallback if none."""
    if series.empty:
        return fallback
    last_valid = series[~series.isna()]
    if not last_valid.empty:
        value = last_valid.iloc[-1]
        # If value is a numpy scalar, convert to Python scalar
        if hasattr(value, 'item'):
            return value.item()
        return value
    return fallback

# --- Configuration ---
symbols = ['NFL.NS', 'IREDA.NS']
bot_token = os.getenv('BOT_TOKEN')
print("bot_token", bot_token)
chat_id = os.getenv('CHAT_ID')
print("chat_id", chat_id)
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
print("NEWS_API_KEY", NEWS_API_KEY)

# File to store hourly price data
PRICE_HISTORY_FILE = 'hourly_price_history.json'

# Domain-specific keywords
STOCK_KEYWORDS = {
    'NFL.NS': ['fertilizer', 'fertiliser', 'national fertilizer', 'agriculture', 'farming', 'crop', 'urea', 'nitrogen', 'potash'],
    'IREDA.NS': ['renewable energy', 'solar', 'wind', 'green energy', 'clean energy', 'IREDA', 'Indian Renewable Energy']
}

def load_price_history():
    """Load historical hourly price data"""
    try:
        with open(PRICE_HISTORY_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_price_history(history):
    """Save hourly price data"""
    with open(PRICE_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)

def get_current_market_time():
    """Get current Indian market time"""
    ist_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    return ist_time

def is_market_open():
    """Check if Indian market is open (9:15 AM to 3:30 PM IST)"""
    ist_time = get_current_market_time()
    market_start = ist_time.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = ist_time.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_start <= ist_time <= market_end

def fetch_latest_price_data(symbol):
    """Fetch latest price data with intraday details"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get today's 1-minute data
        today_data = ticker.history(period='1d', interval='1m')
        
        if today_data.empty:
            return None
        
        # Get latest price
        latest = today_data.tail(1)
        current_price = latest['Close'].values[0]
        current_time = latest.index[-1]
        
        # Calculate intraday metrics
        day_open = today_data.iloc[0]['Open']
        day_high = today_data['High'].max()
        day_low = today_data['Low'].min()
        day_volume = today_data['Volume'].sum()
        
        # Calculate price movement patterns
        price_changes = today_data['Close'].pct_change().dropna()
        volatility = price_changes.std()
        
        # Detect momentum shifts
        recent_prices = today_data['Close'].tail(30)  # Last 30 minutes
        if len(recent_prices) >= 10:
            short_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[-10]) / recent_prices.iloc[-10]
        else:
            short_momentum = 0
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'day_open': day_open,
            'day_high': day_high,
            'day_low': day_low,
            'day_volume': day_volume,
            'current_time': current_time,
            'intraday_change': ((current_price - day_open) / day_open) * 100,
            'volatility': volatility,
            'short_momentum': short_momentum,
            'price_data': today_data
        }
    
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def analyze_people_mindset(price_data, symbol):
    """Analyze people's mindset based on price movements and patterns"""
    if not price_data:
        return {}
    
    # Get price movements
    prices = price_data['price_data']['Close']
    volumes = price_data['price_data']['Volume']
    
    # Calculate mindset indicators
    mindset_analysis = {}
    
    # 1. Panic/FOMO Detection
    recent_volatility = prices.tail(15).pct_change().std()
    volume_spike = volumes.tail(5).mean() / volumes.mean() if volumes.mean() > 0 else 1
    
    if recent_volatility > 0.02 and volume_spike > 1.5:
        mindset_analysis['market_emotion'] = 'PANIC/FOMO'
        mindset_analysis['emotion_score'] = -0.8 if price_data['intraday_change'] < 0 else 0.8
    elif recent_volatility > 0.015:
        mindset_analysis['market_emotion'] = 'UNCERTAINTY'
        mindset_analysis['emotion_score'] = -0.3
    else:
        mindset_analysis['market_emotion'] = 'STABLE'
        mindset_analysis['emotion_score'] = 0.1
    
    # 2. Momentum Analysis
    if price_data['short_momentum'] > 0.01:
        mindset_analysis['momentum'] = 'STRONG_BUYING'
        mindset_analysis['momentum_score'] = 0.7
    elif price_data['short_momentum'] > 0.005:
        mindset_analysis['momentum'] = 'BUYING'
        mindset_analysis['momentum_score'] = 0.3
    elif price_data['short_momentum'] < -0.01:
        mindset_analysis['momentum'] = 'STRONG_SELLING'
        mindset_analysis['momentum_score'] = -0.7
    elif price_data['short_momentum'] < -0.005:
        mindset_analysis['momentum'] = 'SELLING'
        mindset_analysis['momentum_score'] = -0.3
    else:
        mindset_analysis['momentum'] = 'NEUTRAL'
        mindset_analysis['momentum_score'] = 0
    
    # 3. Support/Resistance Psychology
    day_range = price_data['day_high'] - price_data['day_low']
    current_position = (price_data['current_price'] - price_data['day_low']) / day_range if day_range > 0 else 0.5
    
    if current_position > 0.8:
        mindset_analysis['resistance_psychology'] = 'TESTING_RESISTANCE'
        mindset_analysis['psychology_score'] = -0.2
    elif current_position < 0.2:
        mindset_analysis['resistance_psychology'] = 'TESTING_SUPPORT'
        mindset_analysis['psychology_score'] = 0.2
    else:
        mindset_analysis['resistance_psychology'] = 'MID_RANGE'
        mindset_analysis['psychology_score'] = 0
    
    # 4. Overall Mindset Score
    total_mindset_score = (
        mindset_analysis.get('emotion_score', 0) * 0.4 +
        mindset_analysis.get('momentum_score', 0) * 0.4 +
        mindset_analysis.get('psychology_score', 0) * 0.2
    )
    mindset_analysis['total_mindset_score'] = total_mindset_score
    
    return mindset_analysis

def fetch_news_sentiment(symbol, company_name, keywords):
    """Fetch news and analyze sentiment"""
    try:
        news_data = []
        
        # Try Yahoo Finance news first
        try:
            yf_ticker = yf.Ticker(symbol)
            news = yf_ticker.news
            for article in news[:5]:  # Get latest 5 news
                news_data.append({
                    'title': article.get('title', ''),
                    'description': article.get('summary', ''),
                    'source': 'Yahoo Finance',
                    'publishedAt': datetime.datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M')
                })
        except:
            pass
        
        # Analyze sentiment
        sentiments = []
        for news in news_data:
            text = f"{news['title']} {news['description']}"
            sentiment = TextBlob(text).sentiment.polarity
            sentiments.append(sentiment)
        
        weighted_sentiment = np.mean(sentiments) if sentiments else 0
        
        return {
            'sentiment_score': weighted_sentiment,
            'news_count': len(news_data),
            'recent_news': news_data[:2]
        }
    
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return {
            'sentiment_score': 0,
            'news_count': 0,
            'recent_news': []
        }

def predict_with_mindset(symbol, price_data, mindset_data, news_data):
    """Make prediction incorporating people's mindset"""
    try:
        # Get historical data for technical analysis
        ticker = yf.Ticker(symbol)
        hist = yf.download(symbol, start=datetime.datetime.now() - datetime.timedelta(days=30), end=datetime.datetime.now(), progress=False, auto_adjust=True)
        
        if hist.empty:
            return None
        
        # Calculate basic technical indicators
        hist['SMA_5'] = hist['Close'].rolling(window=5).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])
        hist['Price_Momentum'] = hist['Close'].pct_change(5)
        
        # Use robust helper to get last valid values
        sma_5_val = get_last_valid(hist['SMA_5'], price_data['current_price'])
        rsi_val = get_last_valid(hist['RSI'], 50)
        price_momentum_val = get_last_valid(hist['Price_Momentum'], 0)
        vol_mean = hist['Volume'].mean()
        if isinstance(vol_mean, pd.Series):
            # If mean returns a Series (shouldn't for a normal Series, but just in case)
            vol_mean = vol_mean.iloc[-1]
        if hist['Volume'].empty or pd.isna(vol_mean):
            volume_mean = 1
        else:
            volume_mean = vol_mean
        volume_ratio = price_data['day_volume'] / volume_mean if volume_mean > 0 else 1
        
        features = pd.DataFrame({
            'current_price': [price_data['current_price']],
            'intraday_change': [price_data['intraday_change']],
            'volatility': [price_data['volatility']],
            'short_momentum': [price_data['short_momentum']],
            'mindset_score': [mindset_data['total_mindset_score']],
            'news_sentiment': [news_data['sentiment_score']],
            'volume_ratio': [volume_ratio],
            'sma_5': [sma_5_val],
            'rsi': [rsi_val],
            'price_momentum': [price_momentum_val]
        })
        
        # Simple prediction model (you can enhance this)
        base_prediction = price_data['current_price']
        
        # Adjust based on mindset
        mindset_adjustment = mindset_data['total_mindset_score'] * 0.02  # 2% adjustment per unit mindset score
        news_adjustment = news_data['sentiment_score'] * 0.01  # 1% adjustment per unit sentiment
        momentum_adjustment = price_data['short_momentum'] * 0.5  # 50% of short momentum
        
        predicted_price = base_prediction * (1 + mindset_adjustment + news_adjustment + momentum_adjustment)
        
        # Calculate confidence based on data quality
        confidence = min(0.9, 0.5 + abs(mindset_data['total_mindset_score']) * 0.2 + abs(news_data['sentiment_score']) * 0.1)
        
        print("DEBUG: sma_5_val", sma_5_val, type(sma_5_val))
        print("DEBUG: rsi_val", rsi_val, type(rsi_val))
        print("DEBUG: price_momentum_val", price_momentum_val, type(price_momentum_val))
        print("DEBUG: volume_ratio", volume_ratio, type(volume_ratio))
        print("DEBUG: features", features)
        # Use the time of the latest price in the intraday data
        prediction_time = price_data['current_time']

        return {
            'predicted_price': predicted_price,
            'confidence': confidence,
            'mindset_adjustment': mindset_adjustment,
            'news_adjustment': news_adjustment,
            'momentum_adjustment': momentum_adjustment,
            'prediction_time': prediction_time.strftime('%Y-%m-%d %H:%M:%S IST') if hasattr(prediction_time, 'strftime') else str(prediction_time)
        }
    
    except Exception as e:
        print(f"Error in prediction for {symbol}: {e}")
        traceback.print_exc()
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def send_hourly_update(all_data):
    """Send hourly update to Telegram"""
    if not all_data or bot_token == 'YOUR_BOT_TOKEN':
        return
    
    ist_time = get_current_market_time()
    
    summary_lines = []
    summary_lines.append(f"🕐 *HOURLY MARKET MINDSET UPDATE*")
    summary_lines.append(f"⏰ Time: {ist_time.strftime('%Y-%m-%d %H:%M IST')}")
    summary_lines.append("=" * 50)
    
    for data in all_data:
        if data:
            symbol = data['symbol']
            price_data = data['price_data']
            mindset = data['mindset_analysis']
            news = data['news_data']
            prediction = data['prediction']
            
            summary_lines.append(f"\n📈 *{symbol}*")
            summary_lines.append(f"💰 Current: ₹{price_data['current_price']:.2f}")
            summary_lines.append(f"📊 Intraday: {price_data['intraday_change']:+.2f}%")
            summary_lines.append(f"🧠 Mindset: {mindset['market_emotion']} ({mindset['total_mindset_score']:+.2f})")
            summary_lines.append(f"⚡ Momentum: {mindset['momentum']}")
            summary_lines.append(f"📰 News Sentiment: {news['sentiment_score']:+.2f}")
            
            if prediction:
                summary_lines.append(f"🎯 Predicted: ₹{prediction['predicted_price']:.2f}")
                summary_lines.append(f"🕒 Prediction Time: {prediction['prediction_time']}")

                summary_lines.append(f"🎯 Confidence: {prediction['confidence']:.1%}")
    
    summary_message = "\n".join(summary_lines)
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": summary_message,
        "parse_mode": "Markdown"
    }
    
    try:
        resp = requests.post(url, data=payload)
        if resp.status_code == 200:
            print("Hourly update sent to Telegram!")
        else:
            print(f"Failed to send Telegram message: {resp.text}\nCheck your BOT_TOKEN and CHAT_ID.")
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def run_hourly_analysis():
    """Run the complete hourly analysis"""
    print(f"\n🕐 Starting Hourly Analysis at {get_current_market_time().strftime('%H:%M IST')}")
    print("=" * 60)
    
    all_data = []
    
    for symbol in symbols:
        try:
            print(f"\n📊 Analyzing {symbol}...")
            
            # 1. Fetch latest price data
            price_data = fetch_latest_price_data(symbol)
            if not price_data:
                print(f"No data available for {symbol}")
                continue
            
            # 2. Analyze people's mindset
            mindset_analysis = analyze_people_mindset(price_data, symbol)
            
            # 3. Fetch news sentiment
            ticker = yf.Ticker(symbol)
            company_name = ticker.info.get('shortName', symbol)
            keywords = STOCK_KEYWORDS.get(symbol, [company_name])
            news_data = fetch_news_sentiment(symbol, company_name, keywords)
            
            # 4. Make prediction with mindset
            prediction = predict_with_mindset(symbol, price_data, mindset_analysis, news_data)
            
            # 5. Store data
            all_data.append({
                'symbol': symbol,
                'price_data': price_data,
                'mindset_analysis': mindset_analysis,
                'news_data': news_data,
                'prediction': prediction
            })
            
            # Print summary
            print(f"  Current Price: ₹{price_data['current_price']:.2f}")
            print(f"  Intraday Change: {price_data['intraday_change']:+.2f}%")
            print(f"  Mindset: {mindset_analysis['market_emotion']} ({mindset_analysis['total_mindset_score']:+.2f})")
            print(f"  Momentum: {mindset_analysis['momentum']}")
            if prediction:
                print(f"  Predicted: ₹{prediction['predicted_price']:.2f} (Confidence: {prediction['confidence']:.1%})")
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    # Send update to Telegram
    send_hourly_update(all_data)
    
    # Save price history
    history = load_price_history()
    current_time = get_current_market_time().strftime('%Y-%m-%d %H:%M')
    history[current_time] = {data['symbol']: data['price_data'] for data in all_data}
    save_price_history(history)
    
    print("\n✅ Hourly analysis complete!")

if __name__ == "__main__":
    # Check if market is open
    if not is_market_open():
        print("Market is closed. Running analysis anyway...")
    
    run_hourly_analysis() 