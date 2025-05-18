import pandas as pd
import numpy as np

def evaluate_with_price_data(final_results, historical_df, contract_holding_length = 2):
    filtered_final_results = final_results[
        final_results['Filtered_Advice']!= 'Hold'
    ][['Advice', 'Filtered_Advice']].reset_index()
    merged_results = historical_df.merge(filtered_final_results, on = 'Time', how = 'right')

    end_prices = []
    dropped_index = []
    for i, row in merged_results.iterrows():
        if historical_df[historical_df['Time'] == row.Time + pd.Timedelta(days = contract_holding_length)].shape[0] > 0:
            end_prices.append(historical_df[historical_df
                    ['Time'] == row.Time + pd.Timedelta(days = contract_holding_length)
                ].Close.values[0]
            )
        else:
            dropped_index.append(i)
    merged_results = merged_results.drop(index = dropped_index)
    merged_results['End Price'] = end_prices
    merged_results['relative_diff'] = merged_results['End Price'] - merged_results['Open']
    merged_results['Analysis'] = merged_results.apply(
        lambda row: 'Correct' if (row['relative_diff'] < 0 and row['Filtered_Advice'] == 'Buy Put') or 
        (row['relative_diff'] > 0 and row['Filtered_Advice'] == 'Buy Call') else 'Not Correct',axis=  1
    )
    merged_results['stock_move_%'] = 100 * merged_results['relative_diff'].apply(abs)/merged_results['Open']
    accuracy_with_prices = merged_results[merged_results['Analysis'] == 'Correct'].shape[0]/merged_results.shape[0]
    return merged_results, accuracy_with_prices, merged_results.shape[0]