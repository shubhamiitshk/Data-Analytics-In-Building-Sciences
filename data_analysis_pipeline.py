import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import os

# Set plotting style for professional look
sns.set_theme(style="whitegrid")

class BuildingDataAnalyzer:
    def __init__(self, indoor_file, outdoor_file, energy_file):
        """Initialize with file paths for the datasets."""
        self.indoor_file = indoor_file
        self.outdoor_file = outdoor_file
        self.energy_file = energy_file
        
        self.indoor_df = None
        self.outdoor_df = None
        self.energy_df = None
        self.merged_df = None

    def load_and_preprocess_indoor(self):
        """Loads and cleans Indoor Air Quality data."""
        print("Loading Indoor Data...")
        df = pd.read_excel(self.indoor_file, sheet_name='PAoffice')
        df.columns = ['Date', 'Time', 'CO2_indoor', 'Temp_indoor', 'Pressure_indoor', 
                      'RH_indoor', 'DewPoint_indoor', 'AbsHumidity_indoor']
        
        # Create a unified Datetime column
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
        
        # Convert measurements to numeric, forcing errors to NaN
        cols_to_numeric = ['CO2_indoor', 'Temp_indoor', 'Pressure_indoor', 'RH_indoor', 'DewPoint_indoor', 'AbsHumidity_indoor']
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.replace("###", np.nan, inplace=True)
        df.ffill(inplace=True)
        
        self.indoor_df = df.sort_values('Datetime').dropna(subset=['Datetime'])
        return self.indoor_df

    def load_and_preprocess_outdoor(self):
        """Loads and cleans Outdoor Air Quality data."""
        print("Loading Outdoor Data...")
        # Skip header rows
        df = pd.read_excel(self.outdoor_file, sheet_name='OUTDOOR DATA - Air quality', skiprows=8, header=None)
        df.columns = ['Datetime', 'Time', 'CO2_outdoor', 'Temp_outdoor', 'Pressure_outdoor', 
                      'RH_outdoor', 'DewPoint_outdoor', 'AbsHumidity_outdoor']
        
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        
        cols_to_numeric = ['CO2_outdoor', 'Temp_outdoor', 'Pressure_outdoor', 'RH_outdoor', 'DewPoint_outdoor', 'AbsHumidity_outdoor']
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.ffill(inplace=True)
        self.outdoor_df = df.sort_values('Datetime').dropna(subset=['Datetime'])
        return self.outdoor_df

    def load_and_preprocess_energy(self):
        """Loads and cleans Energy Consumption data."""
        print("Loading Energy Data...")
        df = pd.read_excel(self.energy_file, sheet_name='ENERGYDATA')
        df.columns = ['Datetime', 'Computer', 'Plug_Load', 'AC_Load', 'Light_Fan']
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        
        df.replace("No CT", np.nan, inplace=True)
        
        cols_to_numeric = ['Computer', 'Plug_Load', 'AC_Load', 'Light_Fan']
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.fillna(0, inplace=True)
        
        # Calculate Total Energy
        df['Total_Energy'] = df['Computer'] + df['Plug_Load'] + df['AC_Load'] + df['Light_Fan']
        
        self.energy_df = df.sort_values('Datetime').dropna(subset=['Datetime'])
        return self.energy_df

    def merge_datasets(self):
        """Merges indoor, outdoor, and energy datasets using asof merge (nearest timestamps)."""
        print("Merging Datasets...")
        if self.indoor_df is None or self.outdoor_df is None or self.energy_df is None:
            raise ValueError("Data not loaded. Run load methods first.")
            
        # Merge indoor and outdoor
        merged = pd.merge_asof(self.indoor_df, self.outdoor_df, on='Datetime', direction='nearest', tolerance=pd.Timedelta('5min'))
        # Merge with energy
        self.merged_df = pd.merge_asof(merged, self.energy_df, on='Datetime', direction='nearest', tolerance=pd.Timedelta('5min'))
        return self.merged_df

    def feature_engineering(self):
        """Adds comfort indicators and anomaly detection features."""
        print("Performing Feature Engineering...")
        # Comfort Metrics
        self.merged_df['Comfort_CO2'] = self.merged_df['CO2_indoor'].apply(lambda x: 'Acceptable' if x < 1000 else 'Poor')
        self.merged_df['Comfort_Temp'] = self.merged_df['Temp_indoor'].apply(lambda x: 'Comfortable' if 22 <= x <= 26 else 'Uncomfortable')
        self.merged_df['Comfort_RH'] = self.merged_df['RH_indoor'].apply(lambda x: 'Comfortable' if 40 <= x <= 70 else 'Uncomfortable')
        
        # Outlier Detection (Z-Score)
        self.merged_df['Z_Total_Energy'] = zscore(self.merged_df['Total_Energy'], nan_policy='omit')
        self.merged_df['Is_Energy_Outlier'] = np.abs(self.merged_df['Z_Total_Energy']) > 3
        
        return self.merged_df

    def generate_visualizations(self, output_dir="outputs"):
        """Generates and saves key EDA visualizations."""
        print("Generating Visualizations...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Correlation Heatmap
        plt.figure(figsize=(12, 8))
        cols_for_corr = ['CO2_indoor', 'Temp_indoor', 'RH_indoor', 'CO2_outdoor', 'Temp_outdoor', 'Total_Energy']
        corr_matrix = self.merged_df[cols_for_corr].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix: Air Quality vs Energy')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()

        # 2. Scatter plot: CO2 vs Total Energy colored by comfort
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=self.merged_df, x='CO2_indoor', y='Total_Energy', hue='Comfort_CO2', palette='coolwarm', alpha=0.7)
        plt.title('Indoor CO₂ vs Total Energy Consumption')
        plt.xlabel('Indoor CO₂ (ppm)')
        plt.ylabel('Total Energy (kW)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/co2_vs_energy.png")
        plt.close()

        # 3. Energy Time Series with Outliers Highlighted
        plt.figure(figsize=(14, 6))
        plt.plot(self.merged_df['Datetime'], self.merged_df['Total_Energy'], label='Energy Consumption', alpha=0.6)
        outliers = self.merged_df[self.merged_df['Is_Energy_Outlier']]
        plt.scatter(outliers['Datetime'], outliers['Total_Energy'], color='red', label='Outliers (|z| > 3)', zorder=5)
        plt.title('Energy Consumption Over Time with Outliers')
        plt.xlabel('Datetime')
        plt.ylabel('Total Energy (kW)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/energy_outliers_timeseries.png")
        plt.close()

        print(f"Visualizations saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    # Define file paths
    INDOOR_FILE = 'Indoor Air Quality.xlsx'
    OUTDOOR_FILE = 'Outdoor Air Quality.xlsx'
    ENERGY_FILE = 'ENERGYDATA.xlsx'
    
    # Initialize and run pipeline
    try:
        analyzer = BuildingDataAnalyzer(INDOOR_FILE, OUTDOOR_FILE, ENERGY_FILE)
        analyzer.load_and_preprocess_indoor()
        analyzer.load_and_preprocess_outdoor()
        analyzer.load_and_preprocess_energy()
        
        analyzer.merge_datasets()
        analyzer.feature_engineering()
        
        analyzer.generate_visualizations()
        
        # Save final processed dataset
        analyzer.merged_df.to_csv("Processed_Building_Data.csv", index=False)
        print("Data processing complete. Final dataset saved as 'Processed_Building_Data.csv'.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the .xlsx data files are in the same directory as this script.")
