from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate_model(labeled_df):
    features = ['SMA_Fast', 'SMA_Slow', 'RSI', 'MACD', 'MACD_Signal']
    X = labeled_df[features]
    y = labeled_df['Target']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def apply_model(signals_df, model):
    features = ['SMA_Fast', 'SMA_Slow', 'RSI', 'MACD', 'MACD_Signal']
    signals_df['Prediction'] = model.predict(signals_df[features])
    signals_df['Filtered_Advice'] = signals_df.apply(
        lambda row: row['Advice'] if row['Prediction'] == 1 else 'Hold', axis=1
    )
    return signals_df[['Advice', 'Filtered_Advice']]