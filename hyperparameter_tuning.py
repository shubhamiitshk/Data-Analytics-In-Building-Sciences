import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import optuna
import warnings
warnings.filterwarnings('ignore')

def objective(trial):
    # 1. Load Data
    df = pd.read_csv('Processed_Building_Data.csv', parse_dates=['Datetime'])
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    # 2. Feature Engineering
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    features = ['CO2_indoor', 'Temp_indoor', 'RH_indoor', 'CO2_outdoor', 'Temp_outdoor', 'Hour', 'DayOfWeek']
    
    model_df = df.dropna(subset=features + ['Total_Energy'])
    X = model_df[features]
    y = model_df['Total_Energy']
    
    # Train/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Define Hyperparameter Search Space
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    # 4. Initialize Model
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    
    # 5. Evaluate using Cross-Validation on the training set
    # Using Negative Mean Squared Error for scoring (Optuna minimizes, so we return RMSE)
    scores = cross_val_score(rf, X_train, y_train, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse = np.mean(np.sqrt(-scores))
    
    return rmse

if __name__ == "__main__":
    print("Starting Optuna Hyperparameter Optimization for Random Forest...")
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize', study_name="HVAC_Energy_Optimization")
    
    # Run for 15 trials (in a real scenario, this would be 100+)
    study.optimize(objective, n_trials=15)
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best Trial Score (RMSE): {study.best_value:.4f} kW")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save results to a text file for the CV portfolio
    with open('Optuna_Optimization_Results.txt', 'w') as f:
        f.write("Optuna Hyperparameter Optimization Results\n")
        f.write("="*40 + "\n")
        f.write(f"Best RMSE: {study.best_value:.4f} kW\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
    print("\nResults saved to 'Optuna_Optimization_Results.txt'.")
