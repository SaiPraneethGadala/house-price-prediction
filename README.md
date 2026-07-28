# Hyderabad House Price Prediction (Linear Regression + Streamlit)

Predicts house prices across Hyderabad localities using a Linear Regression
model trained on a synthetic dataset (no real dataset was provided, so
realistic sample data was generated for the areas you listed).

## Files

| File | Purpose |
|---|---|
| `generate_data.py` | Creates the synthetic dataset (`data/house_prices.csv`) |
| `train_model.py` | Trains the Linear Regression pipeline, prints metrics, saves the model |
| `app.py` | Streamlit app for interactive predictions |
| `data/house_prices.csv` | Generated dataset |
| `model/house_price_model.joblib` | Trained model pipeline |
| `requirements.txt` | Python dependencies |

## Dataset Columns

- `area` — locality (Gachibowli, Madhapur, Kondapur, etc.)
- `sqft` — built-up area in square feet
- `bhk` — number of bedrooms
- `bathrooms` — number of bathrooms
- `age_years` — property age in years
- `price_lakhs` — price in INR Lakhs (target)

## Setup

```bash
pip install -r requirements.txt

# (Optional) regenerate the dataset
python generate_data.py

# Train the model
python train_model.py

# Launch the app
streamlit run app.py
```

## Model

- **Algorithm**: Linear Regression (Scikit-learn)
- **Preprocessing**: `StandardScaler` for numeric features, `OneHotEncoder` for `area`, combined via `ColumnTransformer` inside a single `Pipeline`
- **Test performance** (on this synthetic data): R² ≈ 0.95, MAE ≈ 10.8 Lakhs

## Using Your Real Dataset

To swap in real data, replace `data/house_prices.csv` with your own CSV using
the same column names (or edit `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` in
`train_model.py` and `app.py` to match your actual columns), then re-run
`train_model.py`.

## Notes

This is a portfolio/demo project. The dataset is synthetic and price
relationships are illustrative, not based on real Hyderabad market data.
