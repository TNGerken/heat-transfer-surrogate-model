
## Overview  
This project develops a machine learning-based surrogate model to predict the heat transfer coefficient of cylinders in crossflow under a variety of geometric configurations. The model is trained on 456 CFD simulation results generated in Ansys Fluent and anchored to first-principles heat transfer correlations. By using a surrogate model, engineers can obtain accurate predictions without the need for costly CFD simulations.  

---

## Methodology  

### 1. Problem Definition  
The objective is to predict the average heat transfer coefficient (h) for:
- **Single cylinder**
- **Series arrangement**
- **Parallel arrangement**
- **Matrix arrangement**

### 2. Data Generation  
- **Simulation Tool:** Ansys Fluent  
- **Flow Type:** Crossflow over cylinders  
- **Parameters:** Cylinder diameter, spacing ratios, Reynolds number, flow velocity, and arrangement type.  
- **Output:** Heat transfer coefficient values for each configuration.

### 3. Feature Engineering  
- Computed Reynolds number (Re) for each case. 
- Encoded geometry type as categorical variables.
- Scaled numerical features for training.

### 4. Machine Learning Approach  
- **Model:** Gradient Boosting Regressor (scikit-learn)  
- **Training Data:** 80% of CFD dataset  
- **Test Data:** 20% of CFD dataset  
- **Performance Metric:** Mean Absolute Percentage Error (MAPE)  

**Results:**  
- Prediction error: ±6% compared to CFD results  
- Significant reduction in computation time (instant vs. hours per simulation)  

---

## Installation  

```bash
# Clone the repository
git clone https://github.com/yourusername/heat-transfer-surrogate-model.git
cd heat-transfer-surrogate-model

# Install required packages
pip install -r requirements.txt
```

---

## Usage  

### 1. Generate CFD Dataset (Optional)  
Use Ansys Fluent to run simulations and export results to `.csv`.  

### 2. Train the Model  
```bash
python train_model.py
```

### 3. Make Predictions  
```bash
python predict.py --diameter 0.05 --velocity 1.2 --geometry "series"
```

---

## Results  

| Configuration | CFD h (W/m²·K) | Predicted h (W/m²·K) | % Error |
|---------------|---------------|----------------------|---------|
| Single        | 48.1          | 47.5                 | -1.25%  |
| Series        | 36.8          | 37.9                 | +2.99%  |
| Parallel      | 51.3          | 49.6                 | -3.31%  |
| Matrix        | 44.2          | 46.5                 | +5.19%  |

---

## File Structure  

```
heat-transfer-surrogate-model/
│── data/                    # CFD-generated dataset
│── src/                     # Python scripts
│   ├── Ridge_regression.py  # Model training (Gradient Boosting)
│   └── Gradient_Boosting.py # Model training (Ridge Regression)
|── documentation/           # Complete report with technical documentation
│── README.md                 # Project documentation
```

---

## License  
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.  
