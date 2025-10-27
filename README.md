# Stock Market Price Prediction (LSTM Model)

This project fetches historical stock market data using Yahoo Finance, trains a deep learning model (LSTM-based neural network) to learn stock price trends, and predicts the next 30 days of close prices for multiple major companies. The model automatically updates and continues training if an existing model is found, allowing long-term incremental improvement.

---

## Project 
<img width="1005" height="547" alt="image" src="https://github.com/user-attachments/assets/263df978-e19e-4506-a989-f1299b188707" />
<img width="1005" height="547" alt="image" src="https://github.com/user-attachments/assets/fad81f6f-ad28-4900-aad2-90c1fbbe16fc" />
<img width="1005" height="547" alt="image" src="https://github.com/user-attachments/assets/80425a74-f3e9-45bd-94c8-d3dda448c48b" />


## Features

- Downloads full history of stock data via `yfinance`
- Normalizes pricing data using MinMax Scaling
- Uses stacked LSTM layers for long-term trend capture
- Supports incremental model training
- Predicts future 30-day close prices
- Saves predictions as JSON files
- Visualizes:
  - Actual vs Predicted Prices
  - Training Loss vs Validation Loss

---

## Dependencies

Ensure these libraries are installed:

```bash
pip install numpy yfinance tensorflow matplotlib scikit-learn
