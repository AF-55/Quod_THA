import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


df = pd.read_csv("data/all_transactions.csv", index_col=0)


# Preparing sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# Preparing data based on a reference date
def prepare_data(reference_date, seq_length=6):
    """
    Filters data up to a reference date and prepares sequences
    """

    # Creating normalized monthly customer transactions
    formatted_dates = df['date'].apply(lambda x:x[:10])
    df_filtered = df[formatted_dates <= reference_date].copy()
    customer_monthly = df_filtered.groupby(['customer_id', 'year_month']).size().reset_index(name='transactions')
    scaler = MinMaxScaler()
    customer_monthly['transactions_scaled'] = scaler.fit_transform(customer_monthly[['transactions']])

    # List of training data
    X_train_list, y_train_list = [], []
    for customer_id in customer_monthly['customer_id'].unique():
        temp_data = customer_monthly[customer_monthly['customer_id'] == customer_id]['transactions_scaled'].values
        # Ensuring enough data points
        if len(temp_data) > seq_length:
            X_temp, y_temp = create_sequences(temp_data, seq_length)
            X_train_list.append(X_temp)
            y_train_list.append(y_temp)

    # Combining sequences
    if len(X_train_list) == 0:
        print("Not enough data to train the model.")
        return None, None, None, None, scaler

    # Converting training data to PyTorch tensors
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # (samples, time_steps, 1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    # Creating DataLoader
    batch_size = 16
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, customer_monthly, reference_date, seq_length, scaler


# Defining an LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]  # Take the last output
        return self.fc(last_time_step)


# Training the model and predicting transactions
def train_and_predict(reference_date, model_file=None):
    """
    Trains an LSTM model using data up to a reference date and predicts transactions for the next 3 months.
    """
    train_loader, customer_monthly, reference_date, seq_length, scaler = prepare_data(reference_date)
    if train_loader is None:
        return None

    # Initializing the model
    model = LSTMModel()

    # Take pre-trained model or retraining
    if model_file:
        model.load_state_dict(torch.load(model_file))
    else:
        # Initializing a loss and an optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Training the model
        num_epochs = 20
        for epoch in range(num_epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Predicting transactions for each customer over the next 3 months
    future_predictions = {}
    model.eval()
    with torch.no_grad():
        for customer_id in customer_monthly['customer_id'].unique():
            temp_data = customer_monthly[customer_monthly['customer_id'] == customer_id]['transactions_scaled'].values
            if len(temp_data) < seq_length:  # Skip customers with insufficient history
                continue

            last_seq = torch.tensor(temp_data[-seq_length:].reshape(1, seq_length, 1), dtype=torch.float32)

            future_preds = []
            for _ in range(3):
                next_pred = model(last_seq).item()
                future_preds.append(next_pred)
                last_seq = torch.cat((last_seq[:, 1:, :], torch.tensor([[[next_pred]]], dtype=torch.float32)), dim=1)

            # Converting back to original scale
            future_predictions[customer_id] = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

    # Converting predictions to DataFrame
    pred_df = pd.DataFrame.from_dict(future_predictions, orient='index', columns=['Month 1', 'Month 2', 'Month 3'])

    return pred_df, model    # Returning both items that we want to save



# Predicting transactions for Feb-Apr 2019
reference_date = "2019-01-31"
#predictions, model = train_and_predict(reference_date)                                 # use if you want to train a new model
predictions, model = train_and_predict(reference_date, model_file="lstm_model.pth")     # use if you don't want to train a new model

# Saving predictions and model
predictions.to_csv("data/customer_behavior_predictions.csv")
torch.save(model.state_dict(), "lstm_model.pth")

# Visualizing predictions
plt.figure(figsize=(12, 8))
for customer_id in predictions.index[:5]:  # Plot for first 5 customers
    plt.plot(range(1, 4), predictions.loc[customer_id].values, label=f'Customer {customer_id}', linestyle='--', marker='o')
    plt.xticks(range(1,4))
plt.grid(linestyle='--')
plt.xlabel("Future Months")
plt.ylabel("Predicted Transactions")
plt.title(f"Transactions Forecast for Next 3 Months (Up to {reference_date})")
plt.legend()
plt.savefig('results/customer_behavior_prediction.png')
