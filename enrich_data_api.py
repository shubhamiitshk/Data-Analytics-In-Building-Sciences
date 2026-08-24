import pandas as pd
import requests
import json
from datetime import datetime

def fetch_historical_weather(start_date, end_date, lat=40.7128, lon=-74.0060):
    """
    Fetches historical solar radiation and wind speed data from the free Open-Meteo API.
    No API key required for non-commercial archive data.
    """
    print(f"Fetching external API data from {start_date} to {end_date}...")
    
    # Open-Meteo Historical Weather API endpoint
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["direct_radiation", "wind_speed_10m"],
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Parse JSON into Pandas DataFrame
        hourly_data = data['hourly']
        df_api = pd.DataFrame({
            'Datetime': pd.to_datetime(hourly_data['time']),
            'Solar_Radiation_W/m2': hourly_data['direct_radiation'],
            'Wind_Speed_kmh': hourly_data['wind_speed_10m']
        })
        
        print(f"Successfully fetched {len(df_api)} rows of external weather data.")
        return df_api
    else:
        print(f"Failed to fetch data: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    print("--- Starting Data Enrichment Pipeline ---")
    
    # 1. Load the existing processed dataset
    print("Loading local processed data...")
    df_local = pd.read_csv('Processed_Building_Data.csv', parse_dates=['Datetime'])
    
    # 2. Extract date range to query the API
    start_date = df_local['Datetime'].min().strftime('%Y-%m-%d')
    end_date = df_local['Datetime'].max().strftime('%Y-%m-%d')
    
    # 3. Hit the API
    df_api = fetch_historical_weather(start_date, end_date)
    
    if df_api is not None:
        # 4. Merge the local data with the external API data based on Datetime
        # Since API data is hourly and local data is 15-min, we use merge_asof
        print("Merging API data with local dataset...")
        
        # Sort both dataframes by Datetime for merge_asof
        df_local = df_local.sort_values('Datetime')
        df_api = df_api.sort_values('Datetime')
        
        # Merge allowing for up to 1 hour tolerance (matching the nearest hour's weather)
        df_enriched = pd.merge_asof(
            df_local, 
            df_api, 
            on='Datetime', 
            direction='nearest',
            tolerance=pd.Timedelta('1h')
        )
        
        # 5. Save the enriched dataset
        output_file = 'Enriched_Building_Data.csv'
        df_enriched.to_csv(output_file, index=False)
        print(f"\nSUCCESS! Enriched dataset saved to '{output_file}'.")
        print("New columns added: 'Solar_Radiation_W/m2' and 'Wind_Speed_kmh'")
        print(df_enriched[['Datetime', 'Total_Energy', 'Solar_Radiation_W/m2', 'Wind_Speed_kmh']].head())
    else:
        print("Enrichment failed.")
