def build_lstm_prediction(series, future_steps=7, seq_len=14, epochs=6, batch_size=8):
    series = series.dropna()
    if len(series) < seq_len + 2:
        return np.array([])  # ← properly indented inside the if block

    # Use last ~3 sequence lengths of data to reduce model load
    series = series[-(seq_len * 3):]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([LSTM(32, input_shape=(seq_len, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    preds = []
    last_seq = X[-1]
    for _ in range(future_steps):
        pred = model.predict(last_seq.reshape(1, seq_len, 1), verbose=0)[0, 0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
