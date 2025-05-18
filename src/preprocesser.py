import pandas as pd


def generate_rsi_macd_signals(df, short_window=7, long_window=35, trend_window=70):
    df = df.copy()

    # Calculate SMAs
    df['SMA_Fast'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_Slow'] = df['Close'].rolling(window=long_window).mean()
    df['SMA_Trend'] = df['Close'].rolling(window=trend_window).mean()

    # --- RSI (14) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # --- Filtered Advice ---
    advice_list = []

    for i in range(len(df)):
        row = df.iloc[i]

        # Ensure all indicators are available
        if pd.isna(row['SMA_Fast']) or pd.isna(row['SMA_Slow']) or pd.isna(row['SMA_Trend']) or pd.isna(row['RSI']) or pd.isna(row['MACD']) or pd.isna(row['MACD_Signal']):
            advice_list.append('Hold')
            continue

        trend_up = row['Close'] > row['SMA_Trend']
        trend_down = row['Close'] < row['SMA_Trend']
        rsi_up = row['RSI'] > 50
        rsi_down = row['RSI'] < 50
        macd_up = row['MACD'] > row['MACD_Signal']
        macd_down = row['MACD'] < row['MACD_Signal']

        if row['SMA_Fast'] > row['SMA_Slow'] and trend_up and rsi_up and macd_up:
            advice_list.append('Buy Call')
        elif row['SMA_Fast'] < row['SMA_Slow'] and trend_down and rsi_down and macd_down:
            advice_list.append('Buy Put')
        else:
            advice_list.append('Hold')

    df['Advice'] = advice_list

    # Optional: Filter to weekly Monday 9am for option entry timing
    df = df.set_index('Time')
    df = df[df.index.dayofweek == 0]
    df = df[df.index.hour == 9]

    return df[['Close', 'SMA_Fast', 'SMA_Slow', 'RSI', 'MACD', 'MACD_Signal', 'Advice']]

def evaluate_weekly_signals(df, signals_df):
    """
    Evaluates a weekly SMA crossover strategy using hourly price data.

    Parameters:
        df (pd.DataFrame): Historical hourly price data. Must have 'Time' and 'Close'.
        signals_df (pd.DataFrame): Weekly signal output. Must have 'Time' index and 'Advice'.

    Returns:
        result_df (pd.DataFrame): Weekly evaluation with prediction accuracy.
    """
    # Ensure datetime index
    df = df.copy()
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    df.sort_index(inplace=True)

    signals_df = signals_df.copy()
    signals_df['Time'] = pd.to_datetime(signals_df.index)
    signals_df.set_index('Time', inplace=True)
    signals_df.sort_index(inplace=True)

    results = []

    for signal_time, signal_row in signals_df.iterrows():
        advice = signal_row['Advice']

        # Define the week range (from signal time to 5 trading days later)
        start_time = signal_time
        end_time = start_time + pd.Timedelta(days=5)

        # Filter that week's data
        week_data = df[(df.index >= start_time) & (df.index <= end_time)]

        if week_data.empty:
            continue

        start_price = week_data.iloc[0]['Close']
        end_price = week_data.iloc[-1]['Close']

        # Determine if advice was directionally correct
        if advice == 'Buy Call':
            correct = end_price > start_price
        elif advice == 'Buy Put':
            correct = end_price < start_price
        else:
            correct = None

        results.append({
            'Week Start': signal_time.date(),
            'Advice': advice,
            'Start Price': round(start_price, 2),
            'End Price': round(end_price, 2),
            'Correct': correct
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.dropna(subset=['Correct'])

    return result_df

def label_signals(signals_df, historical_df, lookahead_hours=35, threshold=0.001):
    signals_df = signals_df.copy()
    historical_df = historical_df.copy()

    # Ensure time is datetime
    signals_df['Time'] = signals_df.index  # Assume signal_df is time-indexed
    signals_df = signals_df.reset_index(drop=True)
    signals_df['Time'] = pd.to_datetime(signals_df['Time'])
    historical_df['Time'] = pd.to_datetime(historical_df['Time'])

    # Floor signals to the hour to align with historical hourly data
    signals_df['Time'] = signals_df['Time'].dt.floor('h')

    # Merge Close prices into signal_df
    historical_df = historical_df.sort_values('Time')
    signals_df = signals_df.sort_values('Time')

    signals_df = pd.merge_asof(
        signals_df,
        historical_df[['Time', 'Close']],
        on='Time',
        direction='backward',
        suffixes=('', '_Hist')
    )

    # Get future price after N hours
    historical_df.set_index('Time', inplace=True)
    future_prices = historical_df['Close'].shift(-lookahead_hours)
    historical_df['Future_Close'] = future_prices

    # Re-merge to get Future_Close aligned to signal time
    signals_df = pd.merge_asof(
        signals_df,
        historical_df[['Future_Close']].reset_index(),
        on='Time',
        direction='backward'
    )

    # Compute return
    signals_df['Return'] = (signals_df['Future_Close'] - signals_df['Close']) / signals_df['Close']

    # Classify based on advice and return
    def classify(row):
        if row['Advice'] == 'Buy Call' and row['Return'] >= threshold:
            return 1
        elif row['Advice'] == 'Buy Put' and row['Return'] <= -threshold:
            return 1
        else:
            return 0

    signals_df['Target'] = signals_df.apply(classify, axis=1)

    # Drop rows only if they still have NaNs (e.g. at the end of the dataset)
    signals_df.dropna(subset=['Return', 'Future_Close'], inplace=True)

    return signals_df

