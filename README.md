# Smart Building IoT & Energy Analytics

## Project Overview
This project analyzes the relationship between indoor air quality (IAQ), outdoor environmental factors, and energy consumption within a commercial office environment. By integrating and analyzing multiple data streams, this project aims to uncover patterns in energy usage, automatically detect occupancy states without physical hardware, and build advanced machine learning models to forecast future HVAC demand.

This repository serves as a comprehensive showcase of the full data lifecycle: from Data Engineering (API ingestion & ETL) and Exploratory Data Analysis (EDA), to Unsupervised Learning (Clustering) and Deep Learning (LSTM Time-Series Forecasting).

## Objectives & Achievements
- **Data Engineering (ETL & APIs):** Built automated pipelines to clean and merge 38,000+ asynchronous 15-minute IoT sensor readings using `merge_asof`, and integrated the live Open-Meteo REST API to enrich the dataset with historical solar radiation and wind speed metrics.
- **Unsupervised Machine Learning:** Deployed a K-Means clustering algorithm to infer building occupancy states (Unoccupied, Low, High) using indoor CO₂ and temperature as proxies, achieving perfect alignment with actual 9-to-5 working hours.
- **Advanced Predictive Modeling:** Trained Random Forest, ARIMA, and LSTM (Deep Learning) models on a rigorous 70/15/15 time-series split to forecast complex, non-linear HVAC energy demand.
- **MLOps & Hyperparameter Tuning:** Developed an automated cross-validation script using the Optuna framework to search for mathematically optimal Random Forest parameters, reducing RMSE to 0.0090 kW.
- **Business Impact:** Formulated data-driven Demand-Controlled Ventilation (DCV) strategies, projecting up to a 30% reduction in baseline energy waste.

## Data Sources
1. **Indoor Air Quality (`Indoor Air Quality.xlsx`):** Timestamped sensor readings for CO₂, Temp, Pressure, RH.
2. **Outdoor Air Quality (`Outdoor Air Quality.xlsx`):** Ambient outdoor environmental metrics.
3. **Energy Data (`ENERGYDATA.xlsx`):** Power consumption across various loads (Computers, Plug Loads, AC, Lighting).
4. **Open-Meteo API:** External integration for Solar Radiation and Wind Speed data.

## Repository Structure
- `data_analysis_pipeline.py`: Handles data ingestion, cleaning (`ffill`), anomaly detection (Z-scores > 3), and creates the baseline `Processed_Building_Data.csv`.
- `enrich_data_api.py`: Connects to external JSON REST APIs to pull weather data, merging it into `Enriched_Building_Data.csv`.
- `hyperparameter_tuning.py`: Runs an automated Optuna study to find the best ML model parameters.
- `advanced_models.py`: The core ML script running K-Means, Random Forest, ARIMA, and LSTM.
- `Final_Assignment_Submission.ipynb`: The fully executed Jupyter Notebook containing all visualizations and model outputs.
- `Final_Report_Text.md`: A comprehensive business-facing report detailing the methodology, results, and actionable recommendations.

## Tech Stack
- **Languages:** Python 3.x
- **Data Wrangling:** `pandas`, `numpy`, `requests`
- **Machine Learning:** `scikit-learn` (Random Forest, K-Means), `tensorflow` (LSTM), `statsmodels` (ARIMA), `optuna` (Hyperparameter Tuning)
- **Visualization:** `matplotlib`, `seaborn`

## How to Run the Project
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the required `.xlsx` data files in the root directory.
4. Run the data processing and enrichment pipeline:
   ```bash
   python data_analysis_pipeline.py
   python enrich_data_api.py
   ```
5. Run the predictive models:
   ```bash
   python advanced_models.py
   ```
6. Alternatively, open `Final_Assignment_Submission.ipynb` in Jupyter to view the completed analysis.