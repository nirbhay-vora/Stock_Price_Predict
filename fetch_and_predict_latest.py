import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime
import os
import requests

symbols = ['NFL.NS', 'IREDA.NS']

# --- Telegram Bot Config ---
bot_token = os.getenv('BOT_TOKEN')
chat_id = os.getenv('CHAT_ID')

results = []

for symbol in symbols:
    print(f'\nFetching latest price for {symbol}...')
    ticker = yf.Ticker(symbol)
    # Get company name
    try:
        company_name = ticker.info.get('shortName', symbol)
    except Exception:
        company_name = symbol
    # Fetch today's 1-minute interval data
    data = ticker.history(period='1d', interval='1m')
    if data.empty:
        print(f'No data found for {symbol}')
        continue
    latest = data.tail(1)
    latest_price = latest['Close'].values[0]
    latest_date = latest.index[-1]
    print(f'Latest price for {symbol}: {latest_price:.2f} (as of {latest_date})')

    # Fetch last 1 year of daily data for prediction
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=365)
    hist = yf.download(symbol, start=start, end=end, progress=False)
    if hist.empty or len(hist) < 30:
        print(f'Not enough historical data for {symbol} to make a prediction.')
        continue
    # Technical indicators
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI_14'] = 100 - (100 / (1 + rs))
    exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = exp1 - exp2
    features = hist[['SMA_20', 'RSI_14', 'MACD']].dropna()
    if features.empty:
        print(f'Not enough features for {symbol} to make a prediction.')
        continue
    target = hist['Close'].shift(-1).reindex(features.index)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    if len(X_train) < 10:
        print(f'Not enough training data for {symbol} to make a prediction.')
        continue
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    last_row = features.iloc[[-1]]
    predicted_price = model.predict(last_row)[0]
    print(f'Predicted next closing price for {symbol}: {predicted_price:.2f}')
    results.append({
        'Symbol': symbol,
        'Company Name': company_name,
        'Latest Price': latest_price,
        'Latest Date': latest_date,
        'Predicted Next Close': predicted_price
    })

# Save or append to CSV
csv_file = 'latest_predictions.csv'
results_df = pd.DataFrame(results)
if not results_df.empty:
    if os.path.exists(csv_file):
        results_df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(csv_file, mode='w', header=True, index=False)
    print(f"Results saved/appended to {csv_file}")
else:
    print("No results to save.")

# --- Send summary to Telegram ---
if not results_df.empty and bot_token != 'YOUR_BOT_TOKEN' and chat_id != 'YOUR_CHAT_ID':
    summary_lines = [
        f"{row['Company Name']} ({row['Symbol']}):\n  Latest Price: {row['Latest Price']:.2f} (as of {row['Latest Date']})\n  Predicted Next Close: {row['Predicted Next Close']:.2f}"
        for _, row in results_df.iterrows()
    ]
    summary_message = "Stock Prediction Summary:\n\n" + "\n\n".join(summary_lines)
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": summary_message
    }
    try:
        resp = requests.post(url, data=payload)
        if resp.status_code == 200:
            print("Summary sent to Telegram!")
        else:
            print(f"Failed to send Telegram message: {resp.text}")
    except Exception as e:
        print(f"Error sending Telegram message: {e}") 