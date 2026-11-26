# House Price Prediction System

An end-to-end machine-learning project that predicts a property's estimated price from basic features using **Linear Regression**, with a **Streamlit** web interface.

## Features
- Explore and clean housing data with Pandas and NumPy.
- Visualize price relationships with Matplotlib.
- Encode the categorical location feature using OneHotEncoder.
- Train and evaluate a Scikit-learn Linear Regression pipeline.
- Save and reload the trained model with joblib.
- Enter property details in a Streamlit application to get a prediction.

## Repository structure
```
house-price-prediction-system/
├── app.py
├── train_model.py
├── house_price_prediction.ipynb
├── requirements.txt
├── data/
│   └── house_prices.csv
├── models/                 # Created after training
├── .gitignore
└── README.md
```

## Setup
Use Python 3.10 or newer.

```bash
git clone https://github.com/KageFlow/house-price-prediction-system.git
cd house-price-prediction-system
python -m venv .venv
```

Activate the environment:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install packages:
```bash
pip install -r requirements.txt
```

## Train the model
```bash
python train_model.py
```
The script prints MAE, RMSE, and R² on a held-out test set and saves the pipeline to `models/house_price_model.joblib`.

## Launch the Streamlit app
After training:
```bash
streamlit run app.py
```
Open the local URL shown in the terminal.

## Google Colab
Upload `house_price_prediction.ipynb` to [Google Colab](https://colab.research.google.com/) and run its cells in order. The notebook expects the repository's `data/house_prices.csv` to be available in the Colab working directory. You can upload the CSV if needed.

## Dataset and limitations
The included `data/house_prices.csv` is **synthetic demo data** created to make the project runnable. It is not real estate market data and should not be used for actual buying, selling, lending, or valuation decisions. Replace it with a suitable real dataset, document its source and usage terms, and retrain the model before presenting real-world performance.

The model uses area, bedrooms, bathrooms, property age, and location. Actual prices depend on many additional factors such as neighborhood, property condition, amenities, legal status, and market conditions. Metrics from synthetic data are only a software demonstration.

## Evaluation metrics
- **MAE:** average absolute prediction error.
- **RMSE:** penalizes larger errors more strongly.
- **R²:** proportion of target variance explained by the model on the test set.

## Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit, joblib, Google Colab.
