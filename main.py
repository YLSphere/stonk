import pandas as pd
from datetime import datetime, timedelta

import src.constants as C
import src.utils as utils
import src.dataloader as dataloader

import src.preprocesser as preprocesser
import src.model as ClassificationModel
import src.evaluation as ev
import pywhatkit

# Define date range (past 1 year)
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 4)

# Format dates
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

def evaluate_stocks():
    accuracy_dict = {}
    num_records_dict = {}

    for ticker in C.TICKERS:
        df = pd.read_csv(f'./moment_data/MOMENT_HISTORICAL_DATA_{ticker}.csv')
        df['Time'] = df['Time'].apply(lambda x: datetime.strptime(x, C.DATE_FORMAT))

        signals = preprocesser.generate_rsi_macd_signals(df)
        results = preprocesser.evaluate_weekly_signals(df, signals)

        labeled = preprocesser.label_signals(signals, df)
        model = ClassificationModel.train_and_evaluate_model(labeled)
        final = ClassificationModel.apply_model(signals, model)
        eval_df, accuracy_with_prices, num_records = ev.evaluate_with_price_data(final, df, 7)

        accuracy_dict[ticker] = accuracy_with_prices
        num_records_dict[ticker] = num_records
        
    accuracy_df = pd.DataFrame.from_dict(accuracy_dict, orient = 'index').reset_index()
    accuracy_df.columns = ['Ticker', 'Accuracy']
    accuracy_df['Number of Records'] = num_records_dict.values()

    upper_accuracy = []
    lower_accuracy = []

    for i, row in accuracy_df.iterrows():
        lower_bound, upper_bound = utils.accuracy_confidence_interval(row['Accuracy'], row['Number of Records'])
        lower_accuracy.append(lower_bound)
        upper_accuracy.append(upper_bound)

    accuracy_df['Upper Bound'] = upper_accuracy
    accuracy_df['Lower Bound'] = lower_accuracy
    accuracy_df['Accuracy Range'] = abs(accuracy_df['Upper Bound'] - accuracy_df['Lower Bound'])
    return accuracy_df

def main(send_eval = True, send_message = True):
    for ticker in C.TICKERS:
        # Grab data from API and save to local folder
        df = dataloader.get_historical_data(ticker, start_str, end_str)
        df.to_csv(f'./moment_data/MOMENT_HISTORICAL_DATA_{ticker}.csv', index = False)
    last_bd_today = utils.get_last_business_day(None)
    print(f"Last business day: {last_bd_today}")

    msg_string = ""

    for ticker in C.TICKERS:

        df = pd.read_csv(f'./moment_data/MOMENT_HISTORICAL_DATA_{ticker}.csv')
        df['Time'] = df['Time'].apply(lambda x: datetime.strptime(x, C.DATE_FORMAT))

        signals = preprocesser.generate_rsi_macd_signals(df)
        results = preprocesser.evaluate_weekly_signals(df, signals)

        labeled = preprocesser.label_signals(signals, df)
        model = ClassificationModel.train_and_evaluate_model(labeled)
        final = ClassificationModel.apply_model(signals, model).reset_index()

        recent_day = final[final['Time'] >= pd.Timestamp(last_bd_today)]
        previous_days = final[final['Time'] < pd.Timestamp(last_bd_today)]
        msg_string += '------------------------------------------------------------\n'
        msg_string += f"Last alert for {ticker}: {previous_days[previous_days['Filtered_Advice'] != 'Hold'].iloc[-1].Time.strftime('%Y-%m-%d')}\n"
        if recent_day.shape[0] > 0 and recent_day['Filtered_Advice'].values[0] != 'Hold':
            recent_advice = recent_day[recent_day['Filtered_Advice'] != 'Hold'].iloc[-1]
            msg_string += 'ALERT FOR ' + ticker + '\n'
            msg_string += '----------------\n'
            msg_string +=f"TIME OF ADVICE: {recent_advice.Time.strftime('%Y-%m-%d')}\n"
            msg_string +=f"ADVICE: {recent_advice['Filtered_Advice']}\n"
            msg_string +=f"OPEN PRICE: {str(df[df['Time'] == recent_day.Time.values[0]].Open.values[0])}\n"
        else:
            msg_string += 'No Alerts for: ' + ticker + '\n'

    if send_eval:
        accuracy_df = evaluate_stocks()
        msg_string += '*****************EVALUATION*****************\n'
        msg_string += accuracy_df.to_string(
            index=False,
            header=True,
            float_format='%.2f'
        ) + '\n'
        msg_string += '*****************EVALUATION*****************'
    if send_message:
        pywhatkit.sendwhatmsg_instantly(C.WHATSAPP_CONTACT_NUMBER, 
            msg_string,
            tab_close = True
        )

