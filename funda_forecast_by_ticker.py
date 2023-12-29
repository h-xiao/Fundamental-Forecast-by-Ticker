import pandas as pd
from pandas import Timestamp, Timedelta
import numpy as np
import pyodbc
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking
import matplotlib.pyplot as plt
import seaborn as sns
import simfin as sf
from simfin.names import *
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from kneed import KneeLocator
from textblob import TextBlob
import configparser

class ConfigManager:
    @staticmethod
    def load_config(file_path):
        config = configparser.ConfigParser()
        config.read(file_path)
        return config

class DatabaseConnector:
    def __init__(self, config):
        self.conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={config['DEFAULT']['server']};"
            f"DATABASE={config['DEFAULT']['database']};"
            f"Trusted_Connection=yes;"
        )
        self.conn = None

    def connect(self):
        self.conn = pyodbc.connect(self.conn_str)

    def query_database(self, query):
        return pd.read_sql(query, self.conn)

    def close(self):
        if self.conn:
            self.conn.close()


class TimeSeriesAnalysis:
    def __init__(self, df, n_clusters, is_value, n_hold_out):
        self.df = df
        self.n_clusters = n_clusters
        self.is_value = is_value
        self.n_hold_out = n_hold_out

    # Function to find the optimal number of clusters using the Elbow method with Kneedle algorithm
    def determine_optimal_clusters_kneedle(self, scaled_data, max_num_clusters=10):
        wcss = []
        cluster_values = list(range(1, max_num_clusters + 1))

        for i in cluster_values:
            kmeans = TimeSeriesKMeans(n_clusters=i, metric="dtw", verbose=False)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)

        kneedle = KneeLocator(cluster_values, wcss, curve='convex', direction='decreasing')
        return kneedle.elbow

    # Adjust optimal_clusters if any cluster has only one time series assigned
    def adjust_clusters_if_necessary(self, n_clusters, data):
        while n_clusters > 1:
            model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=False)
            labels = model.fit_predict(data)
            cluster_sizes = np.bincount(labels[labels >= 0])  # exclude possible -1 labels for noise
            if np.any(cluster_sizes == 1):  # Check if there's any cluster with only one time series
                n_clusters -= 1  # Reduce the number of clusters
            else:
                break
        return n_clusters, labels

    def time_series_cluster_tickers(self):
        # Pivot your dataframe
        df_cluster = self.df.pivot(index='Report Date', columns='Ticker', values=self.is_value)

        # get count of na's in first & last row then drop the row that has more na's
        na_first_row = df_cluster.iloc[0].isna().sum()
        na_last_row = df_cluster.iloc[-1].isna().sum()

        # Drop the row with more NaNs
        if na_first_row > na_last_row:
            df_cluster = df_cluster.iloc[1:]  # Drop first row
        elif na_last_row > na_first_row:
            df_cluster = df_cluster.iloc[:-1]  # Drop last row

        df_cluster.dropna(axis=1, inplace=True)
        # remove the rows that will be in our test set
        df_cluster = df_cluster.iloc[:-self.n_hold_out]

        # Scale the time series data
        scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Standardize to have mean 0 and std 1
        scaled_time_series = scaler.fit_transform(df_cluster.T)  # Transpose because the scaler expects shape (n_samples, n_timestamps)

        if self.n_clusters == False:
            # Determine the optimal number of clusters
            optimal_clusters = self.determine_optimal_clusters_kneedle(scaled_time_series, max_num_clusters=10)

            # Adjust the number of clusters if necessary
            if optimal_clusters is None:
                optimal_clusters = 3  # default number of clusters
            final_clusters, labels = self.adjust_clusters_if_necessary(optimal_clusters, scaled_time_series)

        else:  # user picks n_clusters
            final_clusters = self.n_clusters

        # Perform final clustering
        ts_cluster_model = TimeSeriesKMeans(n_clusters=final_clusters, metric="dtw", verbose=True)
        final_labels = ts_cluster_model.fit_predict(scaled_time_series)

        # Create a DataFrame linking tickers to clusters
        ticker_clusters = pd.DataFrame({'ticker': df_cluster.columns, 'cluster': final_labels})

        # Dictionary to store models for each cluster
        cluster_models = {}

        # Dictionary to store tickers for each cluster
        cluster_tickers = {}

        # Assign tickers to models based on clusters
        for cluster in range(final_clusters):
            cluster_tickers[cluster] = ticker_clusters[ticker_clusters['cluster'] == cluster]['ticker'].tolist()
            cluster_models[cluster] = None  # Placeholder for the model

        return cluster_models, cluster_tickers, ts_cluster_model, scaled_time_series, ticker_clusters


class DataPreprocessor:
    def __init__(self, target_ticker, is_value, df_all, n_hold_out, n_past):
        self.target_ticker = target_ticker
        self.is_value = is_value
        self.df_all = df_all
        self.scaler = StandardScaler()
        self.n_hold_out = n_hold_out
        self.n_past = n_past
        self.is_value = is_value

    def load_data(self):
        self.industry = self.df_all[self.df_all['Ticker'] == self.target_ticker]['industry'].iloc[0]
        df = self.df_all[self.df_all['industry'] == self.industry][['Ticker', 'Report Date', 'Publish Date', self.is_value]]
        return df

    def adjust_and_pivot(self, df_adj):

        is_values_df = df_adj.pivot(index='Report Date', columns='Ticker', values=self.is_value)
        pub_dates_df = df_adj.pivot(index='Report Date', columns='Ticker', values='Publish Date')

        # get count of na's in first & last row then drop the row that has more na's
        na_first_row = is_values_df.iloc[0].isna().sum()
        na_last_row = is_values_df.iloc[-1].isna().sum()

        # Drop the row with more NaNs
        if na_first_row > na_last_row:  # Drop first row
            is_values_df = is_values_df.iloc[1:]
            pub_dates_df = pub_dates_df.iloc[1:]
        elif na_last_row > na_first_row:   # Drop last row
            is_values_df = is_values_df.iloc[:-1]
            pub_dates_df = pub_dates_df.iloc[:-1]
        is_values_df.dropna(axis=1, inplace=True)
        pub_dates_df.dropna(axis=1, inplace=True)

        # add additional feature of target tickers last quarter's value
        is_values_df[f'{self.target_ticker} Previous'] = is_values_df[f'{self.target_ticker}'].shift(1)
        pub_dates_df[f'{self.target_ticker} Previous'] = pub_dates_df[f'{self.target_ticker}'].shift(1)

        # only keep the rows where target_ticker not na
        is_values_df = is_values_df[is_values_df[f'{self.target_ticker}'].notna()]
        pub_dates_df = pub_dates_df[pub_dates_df[f'{self.target_ticker}'].notna()]

        # move target ticker to first column
        cols_without_target = is_values_df.columns.tolist()
        cols_without_target.remove(f'{self.target_ticker}')
        is_values_df = is_values_df[[f'{self.target_ticker}'] + cols_without_target]
        pub_dates_df = pub_dates_df[[f'{self.target_ticker}'] + cols_without_target]

        return is_values_df, pub_dates_df

    def preprocess_data(self, df, pub_dates_df):
        df_for_all = df.copy()

        # hold out the last 2 rows and use as test data
        df_for_all = df_for_all.reset_index()
        train_dates = pd.to_datetime(df_for_all['Report Date'][:-self.n_hold_out])
        test_dates = pd.to_datetime(df_for_all['Report Date'][-self.n_hold_out:])
        df_for_training = df_for_all[df_for_all['Report Date'].isin(train_dates)]
        df_for_training = df_for_training.drop('Report Date', axis=1)

        # LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
        self.scaler = self.scaler.fit(df_for_training)
        df_for_training_scaled = self.scaler.transform(df_for_training)

        # set up test df
        df_for_testing = df_for_all[df_for_all['Report Date'].isin(test_dates.tolist() + train_dates[-self.n_past + 1:].tolist())]
        df_for_testing = df_for_testing.drop('Report Date', axis=1)
        df_for_testing.reset_index(drop=True, inplace=True)
        df_for_testing_scaled = self.scaler.transform(df_for_testing)

        # split pub_dates_df to train and test
        pub_dates_df = pub_dates_df.reset_index()
        pub_dates_df_train = pub_dates_df[pub_dates_df['Report Date'].isin(train_dates)]
        pub_dates_df_train = pub_dates_df_train.drop('Report Date', axis=1)
        pub_dates_df_test = pub_dates_df[pub_dates_df['Report Date'].isin(test_dates.tolist() + train_dates[-self.n_past + 1:].tolist())]
        pub_dates_df_test = pub_dates_df_test.drop('Report Date', axis=1)
        return df_for_training_scaled, df_for_testing_scaled, test_dates, pub_dates_df_train, pub_dates_df_test


    def split_data(self, df_train, df_test, pub_dates_df_train, pub_dates_df_test):

        # Splitting logic from original script
        df_for_training_scaled_X = df_train[:, 1:]
        df_for_training_scaled_Y = df_train[:, 0]

        trainX = []
        trainY = []

        self.n_past = 8  # Number of past days we want to use to predict the future.

        pub_dates_df_train.values
        # Reformat input data into a shape: (n_samples x timesteps x n_features)  & do masking
        for i in range(self.n_past, len(df_train) + 1):
            pub_date = pub_dates_df_train.values[i - 1, 0]
            compare_date_train = pub_dates_df_train.values[i - self.n_past:i:,  1:]
            mask = pub_date > compare_date_train
            mask_train=mask*df_for_training_scaled_X[i - self.n_past:i, 0:df_for_training_scaled_X.shape[1]]
            mask_train[np.isnan(mask_train)] = 0

            trainX.append(mask_train)
            trainY.append(df_for_training_scaled_Y[i - 1:i])

        trainX, trainY = np.array(trainX), np.array(trainY)

        # set up test df
        df_for_testing_scaled_X = df_test[:, 1:]
        df_for_testing_scaled_Y = df_test[:, 0]

        # Empty lists to be populated using formatted training data
        testX = []
        testY = []

        # Reformat input data into a shape: (n_samples x timesteps x n_features)  & do masking
        for i in range(self.n_past, len(df_test) + 1):
            pub_date = pub_dates_df_test.values[i - 1, 0]
            compare_date_test = pub_dates_df_test.values[i - self.n_past:i:,  1:]
            mask = pub_date > compare_date_test
            mask_test=mask*df_for_testing_scaled_X[i - self.n_past:i, 0:df_for_testing_scaled_X.shape[1]]
            mask_test[np.isnan(mask_test)] = 0

            testX.append(mask_test)
            testY.append(df_for_testing_scaled_Y[i - 1:i])

        testX, testY = np.array(testX), np.array(testY)

        return trainX, trainY, testX, testY

    def adjust_revenue_by_publish_date(self, df, target_ticker, is_value):
        # Create a DataFrame for the target ticker
        target_df = df[df['Ticker'] == target_ticker][['Report Date', 'Publish Date']].copy()

        # Rename columns for merging
        target_df.rename(columns={'Publish Date': 'Target Publish Date'}, inplace=True)

        # Sort the DataFrame based on Report Date to ensure proper previous quarter revenue calculation
        df.sort_values(by=['Ticker', 'Report Date'], inplace=True)

        # Shift the revenue column to get the previous quarter's revenue for each ticker
        df[f'Previous Quarter {is_value}'] = df.groupby('Ticker')[is_value].shift(1)

        # Merge the target DataFrame with the original DataFrame on Report Date
        merged_df = df.merge(target_df, on='Report Date', how='left')

        # Initialize the adjusted revenue column with existing revenue values
        merged_df[f'Adjusted {is_value}'] = merged_df[is_value]

        # For each ticker in the dataframe other than the target, find the closest Publish Date that is before the Target Publish Date
        for ticker in merged_df['Ticker'].unique():
            if ticker != target_ticker:
                ticker_rows = merged_df[merged_df['Ticker'] == ticker]
                for index, row in ticker_rows.iterrows():
                    # Get the last valid (non-NaN) Previous Quarter Revenue before the Target Publish Date
                    previous_revenues = df[(df['Ticker'] == ticker) &
                                           (df['Publish Date'] < row['Target Publish Date'])][f'Previous Quarter {is_value}']

                    # If there are no such entries, this means we cannot have a valid comparison and should set to NaN
                    if previous_revenues.empty or previous_revenues.last_valid_index() is None:
                        adjusted_revenue = pd.NA
                    else:
                        adjusted_revenue = df.loc[previous_revenues.last_valid_index(), is_value]

                    merged_df.at[index, f'Adjusted {is_value}'] = adjusted_revenue

        # Drop the columns that are not needed anymore
        merged_df.drop(columns=['Target Publish Date', f'Previous Quarter {is_value}', is_value], inplace=True)
        # Rename 'Adjusted Revenue' to 'Revenue'
        merged_df.rename(columns={f'Adjusted {is_value}': is_value}, inplace=True)

        return merged_df

    def pivot_revenues(self, adjusted_df, target_ticker, is_value):
        # Pivot the table with Report Date as index and Tickers as columns
        # Values are from the 'Revenue' column of adjusted_df
        pivoted_df = adjusted_df.pivot(index='Report Date', columns='Ticker', values=is_value)

        # Ensure the target ticker's revenue is the first column
        # By default, pandas will sort the columns alphabetically, so we reorder them
        column_order = [target_ticker] + [ticker for ticker in pivoted_df.columns if ticker != target_ticker]
        pivoted_df = pivoted_df[column_order]
        pivoted_df = pivoted_df.sort_values('Report Date')
        pivoted_df[f'{target_ticker} Previous'] = pivoted_df[f'{target_ticker}'].shift(1)
        rest_col = pivoted_df.columns.tolist()
        rest_col.remove(f'{target_ticker}')
        rest_col.remove(f'{target_ticker} Previous')
        reorder_col = [f'{target_ticker}', f'{target_ticker} Previous'] + rest_col
        pivoted_df = pivoted_df[reorder_col]
        return pivoted_df

    # If Report Date is not the standard month-ends of Dec, Mar, June, Sept need to adjust to closest one
    @staticmethod
    def adjust_report_date(row):
        # Find the closest month end to the quarter ends of Mar, Jun, Sep, Dec
        month_end = pd.date_range(start=row['Report Date'] - pd.offsets.MonthEnd(1),
                                  end=row['Report Date'] + pd.offsets.MonthEnd(1),
                                  freq='M')
        # Filter month_end to get only the quarter ends
        quarter_ends = month_end[month_end.month.isin([3, 6, 9, 12])]

        # Find the nearest quarter end date to the report date
        nearest_quarter_end = quarter_ends[np.argmin(np.abs((quarter_ends - row['Report Date']).days))]

        return nearest_quarter_end


class FeatureEngineering:
    def __init__(self, target_ticker, add_com_ind_px_tickers, add_econ_inds, db_connector):
        self.target_ticker = target_ticker
        self.add_com_ind_px_tickers = add_com_ind_px_tickers
        self.add_econ_inds = add_econ_inds
        self.db_connector = db_connector

    def add_sentiment_features(self, is_values_df, pub_dates_df):
        # list the columns to run sentiment analysis on:
        sent_cols = ['risks', 'risk_mitigations', 'management_challenges', 'management_opportunities',
                     'business_growth',
                     'research_investments', 'long_term_goals', 'competitive_advantage', 'competitive_disadvantage',
                     'legal_regulatory_issues', 'legal_regulatory_impact', 'dividend_policy', 'esg_approach',
                     'sustainability_initiatives']

        query1 = f"SELECT * FROM [sec_files_llm_answer] WHERE ticker='{self.target_ticker}';"
        llm_df = self.db_connector.query_database(query1)

        for col in sent_cols:
            llm_df[col + '_sentiment'] = llm_df[col].apply(self.calculate_sentiment)

        # convert boolean columns to 0,1
        llm_df['new_risks_int'] = llm_df['new_risks'].astype(int)
        llm_df['planned_acquisitions_int'] = llm_df['planned_acquisitions'].astype(int)

        # merge to is_values_df, then lag all the sentiment / newly added boolean columns
        merge_cols = ['new_risks_int', 'planned_acquisitions_int'] + [x + '_sentiment' for x in sent_cols]
        is_values_df = is_values_df.reset_index()
        is_values_df = is_values_df.merge(llm_df[['date'] + merge_cols], left_on='Report Date', right_on='date', how='left')

        # drop date column used in merge
        is_values_df = is_values_df.drop('date', axis=1)

        # lag all the sentiment / newly added boolean columns
        for x in merge_cols:
            is_values_df[x] = is_values_df[x].shift(1)
            pub_dates_df[x] = pub_dates_df.index

        # make Report Date index
        is_values_df = is_values_df.set_index('Report Date')

        return is_values_df, pub_dates_df

    # Function to calculate sentiment
    @staticmethod
    def calculate_sentiment(text):
        if pd.isna(text):
            return 0
        return TextBlob(text).sentiment.polarity

    def query_px_add_to_data(self, is_values_df, pub_dates_df, ticker_list, table='index_px'):
        st_date = pd.to_datetime(pub_dates_df.index[0]) - Timedelta(days=30)
        end_date = pd.to_datetime(pub_dates_df.index[-1])
        st_date_fmt = st_date.strftime('%Y-%m-%d')
        end_date_fmt = end_date.strftime('%Y-%m-%d')

        ticker_tuple = str(tuple(ticker_list))  # Convert the list to a tuple for the SQL query
        if len(ticker_list) == 1:  # remove extra comma at the end
            ticker_tuple = str(ticker_tuple).replace(',', '')

        query1 = f"SELECT ticker, Date, Adj_Close FROM {table} WHERE ticker IN {ticker_tuple} and Date >= '{st_date_fmt}' and Date <= '{end_date_fmt}';"
        com_ind_df = self.db_connector.query_database(query1)
        com_ind_df = com_ind_df.pivot(index='Date', columns='ticker', values='Adj_Close')

        for ticker in ticker_list:
            com_ind_df[f'{ticker}_5_day_MA'] = com_ind_df[f'{ticker}'].rolling(window=5).mean()
            com_ind_df[f'{ticker}_10_day_MA'] = com_ind_df[f'{ticker}'].rolling(window=10).mean()
            com_ind_df[f'{ticker}_20_day_MA'] = com_ind_df[f'{ticker}'].rolling(window=20).mean()

        # merge to is_values_df on report date
        is_values_df = is_values_df.reset_index()
        is_values_df = pd.merge_asof(is_values_df, com_ind_df, left_on='Report Date', right_on='Date', direction='backward')
        is_values_df = is_values_df.set_index('Report Date')

        for n in range(len(com_ind_df.columns)):
            col = com_ind_df.columns[n]
            pub_dates_df[col] = pub_dates_df.index
        return is_values_df, pub_dates_df

    def query_econ_add_to_data(self, is_values_df, pub_dates_df):
        st_date = pd.to_datetime(pub_dates_df.index[0]) - Timedelta(days=30)
        end_date = pd.to_datetime(pub_dates_df.index[-1])
        st_date_fmt = st_date.strftime('%Y-%m-%d')
        end_date_fmt = end_date.strftime('%Y-%m-%d')

        ticker_tuple = str(tuple(self.add_econ_inds))  # Convert the list to a tuple for the SQL query
        if len(self.add_econ_inds) == 1:  # remove extra comma at the end
            ticker_tuple = str(ticker_tuple).replace(',', '')

        query1 = f"SELECT [Key], Date, Value FROM econ_data WHERE [Key] IN {ticker_tuple} and Date >= '{st_date_fmt}' and Date <= '{end_date_fmt}';"
        com_ind_df = self.db_connector.query_database(query1)
        com_ind_df = com_ind_df.pivot(index='Date', columns='Key', values='Value')

        # merge to is_values_df on report date
        is_values_df = is_values_df.reset_index()
        is_values_df = pd.merge_asof(is_values_df, com_ind_df, left_on='Report Date', right_on='Date', direction='backward')
        is_values_df = is_values_df.set_index('Report Date')

        for n in range(len(com_ind_df.columns)):
            col = com_ind_df.columns[n]
            pub_dates_df[col] = pub_dates_df.index
        return is_values_df, pub_dates_df

    def add_more_features(self, is_values_df, pub_dates_df):
        # Adjust is_values_df & pub_dates_df to include additional features (addditional col for pub_dates_df will just be report date (pub_dates_df.index) )
        if len(self.add_com_ind_px_tickers) > 0:
            is_values_df, pub_dates_df = self.query_px_add_to_data(is_values_df, pub_dates_df, self.add_com_ind_px_tickers, table='index_px')

        if len(self.add_econ_inds) > 0:
            is_values_df, pub_dates_df = self.query_econ_add_to_data(is_values_df, pub_dates_df, self.add_econ_inds)

        return is_values_df, pub_dates_df


class LSTMModel:
    def __init__(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY
        self.input_shape = (trainX.shape[1], trainX.shape[2])
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=self.input_shape))
        model.add(LSTM(64, activation='relu', input_shape=self.input_shape, return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(self.input_shape[0]))
        model.add(Dense(self.trainY.shape[1]))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self):
        history = self.model.fit(self.trainX, self.trainY, epochs=10, batch_size=1, validation_split=0.1, verbose=1)
        return history


class Predictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def make_predictions(self, testX):
        prediction = self.model.predict(testX)
        prediction_copies = np.repeat(prediction, testX.shape[2]+1, axis=-1)
        y_pred_future = self.scaler.inverse_transform(prediction_copies)[:, 0]
        return y_pred_future

    @staticmethod
    def plot_results(original, forecast, target_ticker, is_value):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=original, x='Report Date', y=f'{target_ticker}', label=f'Actual -- {target_ticker}')
        sns.lineplot(data=forecast, x='Report Date', y=is_value, label=f'Predicted -- {target_ticker}')
        plt.legend()
        plt.show()


def run_ticker_funda_forecast(target_ticker, is_value='Revenue', n_clusters=False, n_hold_out=4, n_past=8, add_com_ind_px_tickers=[], add_econ_inds=[]):
    # Load configuration and connect to the database
    config = ConfigManager.load_config('config.ini')
    db_connector = DatabaseConnector(config)
    db_connector.connect()

    # Load data from SimFin and database
    sf.set_api_key(config['DEFAULT']['simfin_api_key'])
    sf.set_data_dir(config['DEFAULT']['simfin_path'])
    is_qt = sf.load_income(variant='quarterly', market='us')
    is_qt = is_qt.reset_index()

    stock_info = db_connector.query_database("SELECT * FROM stock_info")
    df_all = is_qt.merge(stock_info[['ticker', 'industry', 'sector']], left_on='Ticker', right_on='ticker')

    # Initialize DataPreprocessor and FeatureEngineering
    preprocessor = DataPreprocessor(target_ticker, is_value, df_all, n_hold_out, n_past)

    # Data preprocessing - get all tickers in target_ticker's industry and use the tickers in same industry's value as features
    df = preprocessor.load_data()
    industry_tickers = df['Ticker'].unique().tolist()

    # Apply the adjusted function to each row of the DataFrame
    df_adj = df.copy()
    df_adj['Report Date_unadj'] = df_adj['Report Date']
    df_adj['Report Date'] = df_adj.apply(DataPreprocessor.adjust_report_date, axis=1)

    # Initialize TimeSeriesAnalysis to cluster tickers - using time series kmean clustering
    ts_analysis = TimeSeriesAnalysis(df_adj, n_clusters, is_value, n_hold_out)
    cluster_models, cluster_tickers, ts_cluster_model, scaled_time_series, ticker_clusters = ts_analysis.time_series_cluster_tickers()

    # DataFrames to collect the plotting data
    plot_data_training = {}
    plot_data_pred = {}

    # training loop: (train on all target ticker and other ticker combinations for an industry)
    for i in range(len(industry_tickers)):
        target_ticker = industry_tickers[i]
        if target_ticker not in ticker_clusters['ticker'].tolist():
            continue
        cluster = ticker_clusters[ticker_clusters['ticker'] == f'{target_ticker}']['cluster'].iloc[0]

        # Data preparation for target ticker
        is_values_df, pub_dates_df = preprocessor.adjust_and_pivot(df_adj)

        # Add more features like commodities prices, index prices, stock prices, economic indicators
        feature_engineering = FeatureEngineering(target_ticker, add_com_ind_px_tickers, add_econ_inds, db_connector)
        is_values_df, pub_dates_df = feature_engineering.add_more_features(is_values_df, pub_dates_df)

        # Add sec file sentiment analysis as features for target ticker
            # Add func to read in sec_files_llm_ans table for target ticker, do sent analysis on selected cols, merge cols to is_values_df (need to lag! 1 period lag)
        is_values_df, pub_dates_df = feature_engineering.add_sentiment_features(is_values_df, pub_dates_df)

        # preprocess - split to test set to hold out from training
        preprocessed_df_train, preprocessed_df_test, test_dates, pub_dates_df_train, pub_dates_df_test, = preprocessor.preprocess_data(is_values_df, pub_dates_df)

        # split data, generate sequences & do masking
        trainX, trainY, testX, testY = preprocessor.split_data(preprocessed_df_train, preprocessed_df_test, pub_dates_df_train, pub_dates_df_test)

        # Check if the model for this cluster has been created
        if cluster_models[cluster] is None:
            lstm_model = LSTMModel(trainX, trainY)
            cluster_models[cluster] = lstm_model
        else:
            lstm_model = cluster_models[cluster]
        history = lstm_model.train()

        # Instead of plotting, collect the loss data into a DataFrame
        history_df = pd.DataFrame(history.history)
        history_df['ticker'] = target_ticker
        plot_data_training[f'{target_ticker}'] = history_df

    # prediction loop
    for i in range(len(industry_tickers)):
        target_ticker = industry_tickers[i]
        if target_ticker not in ticker_clusters['ticker'].tolist():
            continue

        # Find the cluster the ticker belongs to
        for cluster, tickers in cluster_tickers.items():
            if target_ticker in tickers:
                lstm_model = cluster_models[cluster]
                break

        # Data preparation for target ticker
        is_values_df, pub_dates_df = preprocessor.adjust_and_pivot(df_adj)

        # Add more features like commodities prices, index prices, stock prices, economic indicators
        feature_engineering = FeatureEngineering(target_ticker, add_com_ind_px_tickers, add_econ_inds)
        is_values_df, pub_dates_df = feature_engineering.add_more_features(is_values_df, pub_dates_df)

        # Add sec file sentiment analysis as features for target ticker
            # Add func to read in sec_files_llm_ans table for target ticker, do sent analysis on selected cols, merge cols to is_values_df (need to lag! 1 period lag)
        is_values_df, pub_dates_df = feature_engineering.add_sentiment_features(is_values_df, pub_dates_df)

        # Preprocess & split for prediction
        preprocessed_df_train, preprocessed_df_test, test_dates, pub_dates_df_train, pub_dates_df_test, = preprocessor.preprocess_data(is_values_df, pub_dates_df)
        trainX, trainY, testX, testY = preprocessor.split_data(preprocessed_df_train, preprocessed_df_test, pub_dates_df_train, pub_dates_df_test)

        # Make predictions
        predictor = Predictor(lstm_model.model, preprocessor.scaler)
        y_pred_future = predictor.make_predictions(testX)

        # # Plotting results
        forecast = pd.DataFrame({'Report Date': test_dates, is_value: y_pred_future})
        forecast['Report Date'] = pd.to_datetime(forecast['Report Date'])
        original = is_values_df.reset_index()[['Report Date', f'{target_ticker}']]
        original['Report Date'] = pd.to_datetime(original['Report Date'])
        predictor.plot_results(original, forecast, target_ticker, is_value)

        # Instead of plotting, collect the prediction and original data into DataFrames
        pred_df = original.merge(forecast, on='Report Date', how='left')
        pred_df = pred_df[pred_df[f'{target_ticker}'].notna()]
        pred_df = pred_df.rename(columns={f'{target_ticker}': 'Actual', f'{is_value}': 'Pred'})
        pred_df = pred_df[['Report Date', 'Actual', 'Pred']]
        plot_data_pred[f'{target_ticker}'] = pred_df

    return plot_data_training, plot_data_pred, ts_cluster_model, scaled_time_series, ticker_clusters, df


if __name__ == "__main__":
    plot_data_training, plot_data_pred, ts_cluster_model, scaled_time_series, ticker_clusters, df = run_ticker_funda_forecast(industry, is_value)
