"""
Generates a synthetic but realistic house price dataset for Hyderabad
localities, since no real dataset was provided.

Columns:
    area          - locality name
    sqft          - built-up area in square feet
    bhk           - number of bedrooms
    bathrooms     - number of bathrooms
    age_years     - age of the property in years
    price_lakhs   - price in INR Lakhs (target variable)

Run:
    python generate_data.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# Approximate relative price-per-sqft tiers across Hyderabad localities
# (illustrative synthetic values, not real market data)
AREA_PRICE_PER_SQFT = {
    "Jubilee Hills": 9500,
    "Madhapur": 7800,
    "Gachibowli": 7200,
    "Hitech City": 7400,
    "Kondapur": 6900,
    "Begumpet": 6600,
    "Ameerpet": 6000,
    "Manikonda": 5800,
    "Kukatpally": 5500,
    "Miyapur": 5200,
    "Nizampet": 5000,
    "Bachupally": 4800,
    "Tellapur": 4700,
    "Nallagandla": 5100,
    "Pocharam": 4300,
    "Kompally": 4200,
    "LB Nagar": 4000,
    "Nagole": 3800,
    "Attapur": 4100,
    "Shamshabad": 3500,
    "Bowenpally": 4400,
    "Patancheru": 3200,
}

AREAS = list(AREA_PRICE_PER_SQFT.keys())


def generate_synthetic_dataset(n_samples: int = 1200) -> pd.DataFrame:
    areas = np.random.choice(AREAS, size=n_samples)
    sqft = np.random.randint(600, 4000, size=n_samples)
    bhk = np.random.randint(1, 6, size=n_samples)
    bathrooms = np.clip(bhk + np.random.randint(-1, 2, size=n_samples), 1, 6)
    age_years = np.random.randint(0, 30, size=n_samples)

    price_per_sqft = np.array([AREA_PRICE_PER_SQFT[a] for a in areas])

    base_price = sqft * price_per_sqft
    bhk_premium = bhk * 150_000
    bathroom_premium = bathrooms * 80_000
    age_depreciation = base_price * (age_years * 0.004)
    noise = np.random.normal(0, 250_000, size=n_samples)

    price = base_price + bhk_premium + bathroom_premium - age_depreciation + noise
    price = np.clip(price, 800_000, None)  # floor so no negative/absurdly low prices
    price_lakhs = np.round(price / 100_000, 2)

    df = pd.DataFrame(
        {
            "area": areas,
            "sqft": sqft,
            "bhk": bhk,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "price_lakhs": price_lakhs,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset(1200)
    df.to_csv("data/house_prices.csv", index=False)
    print(f"Generated {len(df)} rows -> data/house_prices.csv")
    print(df.head())
