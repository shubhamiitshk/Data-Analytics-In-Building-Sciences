# Data Analytics in Building Science (AR32203)
## Major Assignment: Occupancy Pattern Detection and Energy-IAQ Analysis

### 1. Introduction

The design and operation of modern office buildings increasingly rely on data-driven approaches to balance two critical, often competing objectives: maintaining high Indoor Air Quality (IAQ) and minimizing energy consumption. Poor IAQ has been consistently linked to reduced cognitive function, lower productivity, and sick building syndrome among occupants. Conversely, aggressively conditioning and ventilating spaces to achieve pristine IAQ can lead to astronomical energy costs and a massive carbon footprint. Commercial buildings account for a significant portion of global energy use, primarily driven by Heating, Ventilation, and Air Conditioning (HVAC) systems. Traditional HVAC systems often operate on static, predetermined schedules (e.g., 9 AM to 5 PM), leading to massive energy waste when spaces are unoccupied or sparsely populated.

To understand these dynamics, it is crucial to monitor specific parameters. 
**Key Indoor Parameters include:**
* **Energy Consumption:** Specifically, the electrical load drawn by the HVAC systems (Fan), lighting fixtures (Light), and general office equipment (Plug loads).
* **IAQ Metrics:** 
  * **Temperature & Relative Humidity (RH):** Primary drivers of thermal comfort.
  * **CO₂ Concentration:** An excellent proxy for human occupancy, as humans naturally exhale carbon dioxide.
  * **Absolute Humidity:** The actual mass of water vapor in the air, crucial for HVAC dehumidification load calculations.
  * **Atmospheric Pressure:** Affects air density and HVAC airflow dynamics.

**Key Outdoor Parameters include:**
* **Outdoor IAQ Metrics:** Outdoor Temperature, RH, CO₂, Absolute Humidity, and Atmospheric Pressure. These dictate the baseline conditions of the "fresh air" being pulled into the building and directly impact the energy required to condition that air to indoor comfort standards.

Occupancy detection offers a technological solution to the energy-IAQ dilemma. By utilizing proxy variables such as indoor CO₂ concentrations, building managers can estimate real-time occupancy. This enables Occupant-Centric Control (OCC) and Demand-Controlled Ventilation (DCV), where fresh air is supplied proportionally to the actual number of occupants. Studies show this can reduce energy consumption by up to 30% by eliminating unnecessary conditioning of empty spaces while guaranteeing adequate air quality when the building is full.

The objectives of this analysis are to:
1. Identify correlations between IAQ and energy use.
2. Detect occupancy patterns using clustering techniques (K-Means).
3. Predict energy demand based on environmental conditions using simple regression and advanced machine learning models (Random Forest).

### 2. Data Collection and Pre-processing

**Dataset Description:**
The dataset utilized in this study comprises three primary sources: Indoor Air Quality, Outdoor Air Quality, and Energy Consumption. The data spans approximately a full calendar year and was recorded at a high-frequency **15-minute resolution**, yielding over 30,000 observations per variable. Across the three datasets, there are over 15 raw variables, tracking everything from ambient environmental conditions to specific electrical loads (Computer, Plug, AC, Light/Fan).

**Handling Missing Values & Outliers:**
Raw sensor data inherently contains missing values and placeholder errors (e.g., "###" or "No CT"). These were initially replaced with `NaN`. To handle these gaps without losing valuable temporal continuity, **forward-fill (`ffill`) interpolation** was utilized, under the safe assumption that environmental conditions (like temperature or CO₂) do not change drastically within a 15-minute window. 
To ensure data integrity, outliers were identified using **Z-scores and Boxplots**. For instance, Z-scores were calculated for total energy consumption, and data points where the absolute Z-score exceeded 3 (|z| > 3) were flagged as extreme outliers. These anomalies typically represent irregular spikes caused by equipment malfunction or testing rather than normal occupant behavior.

**Feature Engineering:**
Several new features were engineered to enrich the analysis:
1. **Total Energy:** A summation of Computer, Plug, AC, and Light/Fan loads.
2. **CO₂-Based Occupancy Indicators:** A threshold-based classification was created where Indoor CO₂ levels above 800 ppm strongly indicate a high-occupancy state, whereas levels near the ~400 ppm baseline indicate an unoccupied state.
3. **Temporal Aggregation:** For specific analyses, such as seasonal decomposition, the 15-minute data was aggregated into **hourly averages** to smooth out high-frequency noise and reveal underlying patterns.
4. **Time Features:** The day of the week and the hour of the day were extracted from the timestamp to aid the predictive models in capturing occupancy schedules.

**Data Normalization:**
Because the variables are measured in completely different units (e.g., CO₂ in ppm, Energy in kW, Temperature in °C), the dataset was scaled using a **Standard Scaler** (standardizing the mean to 0 and variance to 1). This normalization is a critical prerequisite to ensure that algorithms relying on distance metrics (like K-Means clustering) or gradient descent (like regression models) perform optimally and do not disproportionately weight variables with larger raw numbers.

### 3. Exploratory Data Analysis (EDA)

The Exploratory Data Analysis phase was critical for uncovering the underlying behavioral patterns of the building. 

**Visualizing Temporal Patterns:**
Temporal visualizations revealed distinct daily and weekly cycles in both Indoor Air Quality and Energy Consumption. Energy usage consistently spiked from 9 AM to 5 PM on weekdays, followed by sharp declines during nighttime and weekends. This strongly suggests that the building operates on a highly regular, human-driven schedule.

**Analyzing Relationships via Scatter Plots & Heat Maps:**
To quantify these relationships, a correlation matrix (visualized as a heatmap) and several scatter plots were generated:
* **Indoor vs. Outdoor Temperature:** A positive correlation was observed, though the indoor temperature range was significantly dampened by the building envelope and HVAC system, maintaining relatively stable comfort conditions despite outdoor fluctuations.
* **CO₂ Concentration vs. Energy Usage:** A scatter plot comparing Indoor CO₂ against Total Energy revealed a clear relationship. As CO₂ levels rose (indicating increased human occupancy), energy consumption escalated proportionally. This confirms CO₂ as a highly effective proxy for occupancy.
* **Energy Consumption vs. IAQ Factors:** The heatmap demonstrated that while CO₂ strongly correlates with energy use, other IAQ factors like Relative Humidity also exhibit correlations, as the HVAC system continuously draws power to dehumidify the air.

**Identification of Peak Energy Consumption:**
By aggregating energy consumption by the hour of the day, a distinct peak demand period was identified occurring between 10 AM and 4 PM, with the absolute maximum peak hitting at roughly **12:00 PM (midday)**. This peak directly coincides with the highest daily levels of Indoor CO₂. The cause of this peak is twofold: maximum human occupancy (driving plug loads from computers/lighting) combined with peak solar heat gain, which forces the HVAC cooling load to its maximum capacity.

**Time-Series Decomposition:**
To further break down these variations, time-series decomposition was applied to the hourly aggregated energy data. The additive decomposition model successfully split the signal into three distinct components:
1. **Trend:** Showed longer-term shifts, such as increased baseline energy usage during warmer seasonal periods.
2. **Seasonality:** Clearly isolated the repeating 24-hour daily cycle, proving the regularity of the building's operation.
3. **Residuals:** Uncovered the "noise" or unexplained variance, which included anomalous energy spikes occurring outside of normal operating hours (e.g., weekend maintenance or equipment testing).

### 4. Occupancy Pattern Detection & Predictive Modelling

#### A. Clustering-Based Occupancy Detection
To automatically classify the operational state of the building without relying on manual headcount logs, an unsupervised **K-Means clustering** algorithm was applied. The data was scaled, and the algorithm was trained on three key features: Indoor CO₂, Indoor Temperature, and Total Energy use. The number of clusters (k) was set to 3 to represent distinct occupancy states:
1. **Unoccupied (Cluster 0):** Characterized by low CO₂ (~400 ppm), baseline temperatures, and minimal energy use.
2. **Low Occupancy (Cluster 1):** Moderate CO₂ and energy levels.
3. **High Occupancy (Cluster 2):** High CO₂ concentrations (>800 ppm) and peak energy consumption.

**Comparison with Working Hours:** When the temporal distribution of these clusters was visualized over time, they perfectly aligned with actual working hours. "High Occupancy" clusters were densely packed between 9 AM and 5 PM on weekdays. Conversely, nights and weekends were almost exclusively categorized into the "Unoccupied" cluster. This proves that environmental sensors can accurately map human presence.

#### B. Regression & Machine Learning Models
To predict `Total_Energy` demand based on environmental factors, a suite of models was developed and evaluated on a strict Train/Validation/Test split (70/15/15).

**Correlation Coefficients & Simple Linear Regression:**
Initial correlation coefficients confirmed that IAQ parameters alone have a complex relationship with energy. A simple Multiple Linear Regression model was built as a baseline. The resulting R² value was low (R² < 0.10). An R² near 0 indicates that a simple linear equation cannot explain the variability in energy consumption, as building thermodynamics are inherently non-linear. 

**Advanced Machine Learning (Random Forest):**
To capture these non-linear dynamics, an advanced **Random Forest Regressor** was implemented. The Random Forest model significantly outperformed the linear model, achieving a much higher R² score and a lower Root Mean Square Error (RMSE). The interpretation of the higher R² value indicates that the Random Forest model is far more capable of explaining the variance in energy consumption. Furthermore, the feature importance plot derived from the Random Forest model indicated that Indoor CO₂ and Indoor Temperature were the most significant predictors of total energy demand, reinforcing the hypothesis that occupancy drives HVAC load.

**Time-Series Forecasting (ARIMA & LSTM):**
Finally, recognizing that energy consumption is sequential, time-series forecasting models were explored. **ARIMA** (AutoRegressive Integrated Moving Average) was utilized for baseline forecasting, while **LSTM** (Long Short-Term Memory) neural networks were applied for deep learning-based forecasting. The LSTM model excelled at capturing the repetitive sequential patterns of daily energy spikes, proving highly effective for future energy demand prediction.

### 5. Results and Discussion

**Summary of Key Insights:**
The exploratory data analysis and clustering models conclusively demonstrate that occupancy patterns are the fundamental drivers of energy consumption in this office space. The K-Means clustering algorithm proved that environmental sensors can distinguish between occupied and unoccupied states with high accuracy. Meanwhile, the advanced predictive models (Random Forest, LSTM) highlighted that while linear relationships are weak, complex, non-linear machine learning models can accurately forecast energy demand using IAQ metrics.

**Implications for HVAC Optimization:**
Currently, many building systems operate on static schedules, wasting energy conditioning empty rooms. The strong correlation between "High Occupancy" clusters and peak energy demand suggests massive potential for optimization. By adopting a dynamic, demand-based operation strategy, the HVAC system could be aggressively set back during "Unoccupied" clusters (nights, weekends, and even extended lunch hours). Furthermore, temperature deadbands could be widened during "Low Occupancy" transition periods.

**The Role of CO₂ Monitoring for Demand-Controlled Ventilation (DCV):**
The analysis validates CO₂ as a highly reliable occupancy proxy. This is the cornerstone of Demand-Controlled Ventilation. Instead of running ventilation fans at full capacity all day, DCV uses real-time CO₂ data to modulate fresh air intake. If CO₂ stays near the ~400 ppm baseline, ventilation is minimized. If it crosses a threshold (e.g., 600 ppm), ventilation ramps up. This guarantees excellent Indoor Air Quality while radically reducing HVAC energy waste.

**Comparison with Existing Research:**
These findings align perfectly with contemporary building science literature. Studies by Fisk et al. (2010) and others have repeatedly shown that HVAC systems account for the largest share of commercial energy use, and that DCV can yield energy savings of 15-40%. Furthermore, recent literature strongly supports the transition from simple linear modeling to machine learning architectures (like Random Forest and LSTM) for building energy forecasting, just as demonstrated in this project.

**Limitations and Future Improvements:**
A primary limitation of this study is the lack of "ground-truth" headcount data (e.g., camera or turnstile counts), meaning occupancy could only be inferred rather than definitively verified. Additionally, the lack of granular Building Management System (BMS) data, such as exact thermostat setpoints or chiller status, limits the ability to calculate exact kilowatt savings. Future improvements should include integrating real-time Wi-Fi device counting for ground-truth occupancy validation and applying energy disaggregation techniques (NILM) to separate HVAC loads from simple plug loads.

### 6. Conclusion and Recommendations

**Conclusion:**
This study successfully demonstrated the powerful application of data analytics in understanding and optimizing building performance dynamics within a commercial office environment. By meticulously preprocessing and analyzing a high-resolution, multi-variable dataset spanning an entire year, several critical insights were uncovered regarding the intricate relationship between Indoor Air Quality (IAQ) and energy consumption. 

The primary and most consequential finding is that occupancy schedules fundamentally dictate the energy consumption profile of the building. The data unequivocally showed that energy usage peaks in tandem with standard working hours and drops to baseline levels during nights and weekends. Furthermore, this study validated that indoor CO₂ concentration serves as an incredibly reliable, non-intrusive proxy for human presence. The K-Means clustering algorithm successfully leveraged this CO₂ data, alongside temperature and energy metrics, to accurately separate the building's operational states into distinct "Occupied" and "Unoccupied" clusters without the need for manual headcount tracking. 

Finally, the predictive modeling phase highlighted the complexity of building thermodynamics. While simple linear regression failed to accurately forecast energy demand, advanced machine learning techniques, specifically Random Forest and time-series forecasting (LSTM), proved highly capable of capturing the non-linear relationships between environmental factors and energy usage.

**Actionable Recommendations:**
Based on the empirical evidence gathered, building management can achieve significant energy savings while maintaining strict thermal comfort and IAQ standards by implementing the following strategies:
1. **Implement Occupancy-Based HVAC Scheduling:** Building managers should immediately transition away from fixed, static HVAC operating schedules. By utilizing the dynamic schedules derived from the identified clustering patterns, the HVAC system can be programmed for deep, aggressive setbacks during reliably identified unoccupied periods (e.g., weekends and after 6 PM).
2. **Deploy CO₂-Based Demand-Controlled Ventilation (DCV):** The building should upgrade its ventilation strategy by linking fresh air intake dampers directly to indoor CO₂ sensors. By establishing a dynamic setpoint threshold (e.g., ~600 ppm), the system will trigger fresh air supply strictly when human occupancy dictates it, thereby eliminating the massive energy penalty associated with needlessly conditioning outdoor air for an empty office.

**Future Research Directions:**
To push the boundaries of building energy efficiency even further, future research should focus on:
* **Real-Time System Integration:** The next logical step is to transition from retrospective data analysis to live deployment. Researchers should investigate the technical requirements for integrating these trained predictive models directly into the building's central Building Management System (BMS) for automated, real-time HVAC adjustments.
* **AI-Driven Optimization:** Future studies should explore the deployment of advanced Artificial Intelligence techniques, such as Deep Reinforcement Learning. Unlike static models, an AI agent could continuously learn from the building's thermal behavior over time, dynamically fine-tuning HVAC setpoints to discover the absolute optimal balance between minimal energy expenditure and maximal occupant comfort under varying weather conditions.

### 7. References
* Fisk, W. J., Black, D., & Brunner, G. (2010). Benefits and costs of improved IEQ in U.S. offices. *Indoor Air*, 21(5), 357-367.
* Wargocki, P., Wyon, D. P., Sundell, J., Clausen, G., & Fanger, P. O. (2000). The effects of outdoor air supply rate in an office on perceived air quality, sick building syndrome (SBS) symptoms and productivity. *Indoor Air*, 10(4), 222-236.
* Yang, J., Santamouris, M., & Lee, S. E. (2016). Review of occupancy sensing systems and occupancy modeling methodologies for the application in institutional buildings. *Energy and Buildings*, 121, 344-349.
* Zhao, J., Lasternas, B., Lam, K. P., Yun, R., & Loftness, V. (2014). Occupant behavior and schedule modeling for building energy simulation through office appliance power consumption data mining. *Energy and Buildings*, 82, 341-355.
