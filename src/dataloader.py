import pandas as pd
import requests
import time 
import src.constants as C

def create_df_from_response(response):
    data = response.json()
    results = data.get("results", [])
    df = pd.DataFrame(results)
    df['t'] = pd.to_datetime(df['t'], unit='ms')  # Convert timestamp
    df.set_index('t', inplace=True)
    df.reset_index(inplace = True)
    print('Compiled DF Chunk!')
    df.rename(columns={
        't': 'Time', 
        'o': 'Open', 
        'h': 'High', 
        'l': 'Low', 
        'c': 'Close', 
        'v': 'Volume',
        'vw': 'Volume Weighted Average Price',
        'n': 'Number of transactions',
    }, inplace=True)
    return df

def make_request_with_retry(url, params=None):
    retries = 0
    while retries < C.MAX_RETRIES:
        try:
            if params:
                response = requests.get(url, params=params)
            else:
                response = requests.get(url)

            if response.status_code == 200:
                return response
            else:
                print(f"Attempt {retries+1} failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Request error: {e}")

        retries += 1
        wait = C.INITIAL_WAIT * (2 ** (retries - 1))
        print(f"Retrying in {wait:.1f} seconds...")
        time.sleep(wait)

    print("Max retries reached. Request failed.")
    return None

def get_historical_data(ticker, start_date, end_date):

    dfs = []
    # API endpoint
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": C.POLYGON_API_KEY
    }

    # Make the request
    response = make_request_with_retry(url, params=params)

    # Check for success
    if response.status_code == 200:
        df = create_df_from_response(response)
        dfs.append(df)
    else:
        print("Failed to fetch data:", response.status_code, response.text)

    while 'next_url' in response.json().keys():
        response = make_request_with_retry(response.json()['next_url'], params=params)
        if response.status_code == 200:
            if response.json()['resultsCount'] > 0:
                df = create_df_from_response(response)
                dfs.append(df)
        else:
            print("Failed to fetch data:", response.status_code, response.text)

    return pd.concat(dfs)