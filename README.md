# Fundamental-Forecast-by-Ticker

## Introduction
This script is part of a larger project, "Stock Analyzer," which includes a Streamlit dashboard with multiple analysis pages. It focuses on forecasting stock fundamentals using Long Short-Term Memory (LSTM) models in Python. The forecasting model clusters similar stocks based on their sector, performs sentiment analysis on 10Q & 10K filings using an LLM, and predicts future values of a company's income statement items.

The dashboard can be viewed at: [https://hxiao.cloud/](https://hxiao.cloud/)

## Features

- **Data Processing:** Pulls stock data and fundamentals, including quarterly revenue, gross profit, and net income.
- **Database Integration:** Uses Azure MSSQL RDS for data storage.
- **Sentiment Analysis:** Extracts and analyzes sentiments from 10Q & 10K files using an LLM.
- **LSTM Modeling:** Forecasts income statement values considering factors like sector similarity and sentiment analysis.
- **Dashboard Integration:** Part of a Streamlit dashboard for dynamic data visualization and analysis.

## Requirements

- Python 3.9
- Required Libraries: see requirements.txt

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/h-xiao/Fundamental-Forecast-by-Ticker.git

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
 
 
## How It Works

- **1. Data Collection and Preprocessing:** Gathers data from various sources, including SimFin and an Azure MSSQL database. It preprocesses data for LSTM modeling.
- **2. Clustering and Feature Engineering:** Clusters stocks from the same sector and adds features from sentiment analysis of financial documents.
- **3. LSTM Training:** Trains an LSTM model on the processed and clustered data.
- **4. Prediction and Visualization:** Predicts future values of selected income statement items and visualizes them in the Streamlit dashboard.



