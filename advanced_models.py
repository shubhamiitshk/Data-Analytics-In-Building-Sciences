import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import os

sns.set_theme(style="whitegrid")

class AdvancedBuildingAnalytics:
    def __init__(self, processed_data_path):
        print("Loading processed data...")
        # Load Enriched Data
        try:
            self.df = pd.read_csv('Enriched_Building_Data.csv', parse_dates=['Datetime'])
        except FileNotFoundError:
            self.df = pd.read_csv(processed_data_path, parse_dates=['Datetime'])
        self.df.set_index('Datetime', inplace=True)
        # Drop rows with missing critical values
        self.df.dropna(subset=['Total_Energy', 'CO2_indoor', 'Temp_indoor', 'RH_indoor'], inplace=True)
        
    def time_series_decomposition(self, output_dir="outputs"):
        """Decompose Total Energy into trend, seasonal, and residual components."""
        print("Performing Time-Series Decomposition...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Resample to hourly data to smooth out noise and make decomposition cleaner
        hourly_energy = self.df['Total_Energy'].resample('h').mean().ffill()
        
        # Decompose assuming a daily seasonality (24 hours)
        decomposition = seasonal_decompose(hourly_energy, model='additive', period=24)
        
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        fig.suptitle('Time-Series Decomposition of Hourly Energy Consumption', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_series_decomposition.png")
        plt.close()
        print("Time-series decomposition plot saved.")

    def occupancy_clustering(self, output_dir="outputs"):
        """Use K-Means clustering to detect occupancy states based on CO2, Temp, and Energy."""
        print("Performing K-Means Clustering for Occupancy Detection...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Select features for clustering
        features = ['CO2_indoor', 'Temp_indoor', 'Total_Energy']
        X = self.df[features].copy()
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-Means (Assuming 3 states: Unoccupied, Low Occupancy, High Occupancy)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.df['Occupancy_Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Sort cluster labels so that 0 = Low, 1 = Medium, 2 = High Energy/CO2 (roughly)
        cluster_means = self.df.groupby('Occupancy_Cluster')['Total_Energy'].mean().sort_values()
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_means.index)}
        self.df['Occupancy_Cluster'] = self.df['Occupancy_Cluster'].map(label_mapping)
        
        # Visualize Clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.df, x='CO2_indoor', y='Total_Energy', 
            hue='Occupancy_Cluster', palette='viridis', alpha=0.6
        )
        plt.title('Occupancy Pattern Clusters (K-Means)')
        plt.xlabel('Indoor CO₂ (ppm)')
        plt.ylabel('Total Energy (kW)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/occupancy_clusters.png")
        plt.close()
        print("Occupancy clustering plot saved.")

    def predictive_modeling(self, output_dir='outputs'):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_squared_error
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        from statsmodels.tsa.arima.model import ARIMA
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from sklearn.preprocessing import MinMaxScaler
        
        print("\nTraining Predictive Models with Optuna Parameters and API Weather Data...")
        os.makedirs(output_dir, exist_ok=True)
        
        self.df['Hour'] = self.df.index.hour
        self.df['DayOfWeek'] = self.df.index.dayofweek
        
        # Define Features including new API features
        features = ['CO2_indoor', 'Temp_indoor', 'RH_indoor', 'CO2_outdoor', 'Temp_outdoor', 'Hour', 'DayOfWeek', 'Solar_Radiation_W/m2', 'Wind_Speed_kmh']
        
        # Ensure all feature columns exist, drop NaNs
        # Only use features that exist in the dataframe (in case someone runs it on the old dataset)
        available_features = [f for f in features if f in self.df.columns]
        model_df = self.df.dropna(subset=available_features + ['Total_Energy'])
        X = model_df[available_features]
        y = model_df['Total_Energy']
        
        # Train, Eval, Test Split (70% Train, 15% Eval, 15% Test)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=False) # Sequential split for time series
        X_train, X_eval, y_train, y_eval = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, shuffle=False)
        
        print(f"Data Split -> Train: {len(X_train)} | Eval: {len(X_eval)} | Test: {len(X_test)}")
        
        # 1. Random Forest Regressor with Optuna Best Parameters
        rf_model = RandomForestRegressor(
            n_estimators=51,
            max_depth=28,
            min_samples_split=10,
            min_samples_leaf=6,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        y_pred_test_rf = rf_model.predict(X_test)
        rf_test_r2 = r2_score(y_test, y_pred_test_rf)
        rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_rf))
        
        print(f"\nRandom Forest Results (Optimized):")
        print(f"Test R²: {rf_test_r2:.4f} | Test RMSE: {rf_test_rmse:.4f} kW")
        
        # 2. ARIMA Time-Series Forecasting
        print("\nTraining ARIMA model...")
        # Train on the same y_train sequence
        arima_model = ARIMA(y_train, order=(5,1,0))
        arima_fit = arima_model.fit()
        # Predict on Test set length
        y_pred_arima = arima_fit.forecast(steps=len(y_test))
        arima_rmse = np.sqrt(mean_squared_error(y_test, y_pred_arima))
        print(f"ARIMA Test RMSE: {arima_rmse:.4f} kW")
        
        # 3. LSTM Deep Learning Forecasting
        print("\nTraining LSTM Neural Network...")
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        
        # Reshape for LSTM [samples, time steps, features]
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mse', optimizer='adam')
        
        # Train quickly (epochs=10 for demonstration)
        lstm_model.fit(X_train_lstm, y_train_scaled, epochs=10, batch_size=32, verbose=0, shuffle=False)
        
        y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
        y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled)
        lstm_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
        print(f"LSTM Test RMSE: {lstm_rmse:.4f} kW")
        
        print("\nPredictive modeling successfully completed!")       # Feature Importance Plot (Random Forest)
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Random Forest Feature Importances (Predicting Energy)")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rf_feature_importance.png")
        plt.close()
        print("\nPredictive modeling plots saved.")
        
if __name__ == "__main__":
    PROCESSED_FILE = 'Processed_Building_Data.csv'
    
    if os.path.exists(PROCESSED_FILE):
        analytics = AdvancedBuildingAnalytics(PROCESSED_FILE)
        
        # Section 3: Time Series Decomposition
        analytics.time_series_decomposition()
        
        # Section 4A: Occupancy Pattern Detection
        analytics.occupancy_clustering()
        
        # Section 4B: Regression & Machine Learning
        analytics.predictive_modeling()
        
        print("\nAll advanced analytics completed successfully!")
    else:
        print(f"Error: {PROCESSED_FILE} not found.")
        print("Please run data_analysis_pipeline.py first to generate the processed dataset.")
