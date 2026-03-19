"""
Car Price Prediction Model
==========================
Inspired by the Cardeko Car Price Prediction approach.
Implements data preprocessing, model training, evaluation, and prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET
#    (Replace this section with your real CSV)
# ─────────────────────────────────────────────
def generate_dataset(n=1500, seed=42):
    """Generate a realistic synthetic car dataset for demonstration."""
    np.random.seed(seed)

    brands = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford',
              'BMW', 'Audi', 'Mercedes', 'Volkswagen', 'Tata']
    fuel_types = ['Petrol', 'Diesel', 'CNG', 'Electric']
    transmissions = ['Manual', 'Automatic']
    seller_types = ['Individual', 'Dealer', 'Trustmark Dealer']
    owners = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']

    # Base prices per brand (in lakhs INR)
    brand_base = {
        'Maruti': 5, 'Hyundai': 7, 'Honda': 9, 'Toyota': 12, 'Ford': 8,
        'BMW': 35, 'Audi': 40, 'Mercedes': 45, 'Volkswagen': 15, 'Tata': 6
    }

    data = []
    for _ in range(n):
        brand = np.random.choice(brands)
        year = np.random.randint(2005, 2024)
        km = np.random.randint(5000, 200000)
        fuel = np.random.choice(fuel_types, p=[0.45, 0.35, 0.1, 0.1])
        trans = np.random.choice(transmissions, p=[0.6, 0.4])
        seller = np.random.choice(seller_types, p=[0.5, 0.35, 0.15])
        owner = np.random.choice(owners, p=[0.55, 0.30, 0.10, 0.05])
        mileage = round(np.random.uniform(10, 25), 1)  # km/l
        engine = np.random.choice([800, 1000, 1197, 1248, 1498, 1598, 1968, 2993, 3498])
        seats = np.random.choice([5, 7, 8], p=[0.7, 0.25, 0.05])

        # Price calculation with realistic factors
        age = 2024 - year
        base = brand_base[brand]
        price = base * 100000  # Convert to rupees

        # Depreciation
        price *= max(0.25, 1 - (age * 0.08))

        # Km driven penalty
        price *= max(0.6, 1 - (km / 500000))

        # Fuel & transmission bonus
        if fuel == 'Diesel':
            price *= 1.1
        elif fuel == 'Electric':
            price *= 1.3
        if trans == 'Automatic':
            price *= 1.08

        # Owner penalty
        owner_factors = {'First Owner': 1.0, 'Second Owner': 0.85,
                         'Third Owner': 0.70, 'Fourth & Above Owner': 0.55}
        price *= owner_factors[owner]

        # Engine size bonus
        price *= (1 + engine / 100000)

        # Add noise
        price *= np.random.uniform(0.88, 1.12)
        price = round(price, -3)  # Round to nearest 1000

        data.append({
            'name': brand,
            'year': year,
            'km_driven': km,
            'fuel': fuel,
            'seller_type': seller,
            'transmission': trans,
            'owner': owner,
            'mileage': mileage,
            'engine': engine,
            'seats': seats,
            'selling_price': price
        })

    return pd.DataFrame(data)


# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    """Clean and encode the dataset."""
    df = df.copy()

    # Drop rows with missing target
    df.dropna(subset=['selling_price'], inplace=True)

    # Remove extreme outliers using IQR
    Q1 = df['selling_price'].quantile(0.01)
    Q3 = df['selling_price'].quantile(0.99)
    df = df[(df['selling_price'] >= Q1) & (df['selling_price'] <= Q3)]

    # Feature engineering
    df['car_age'] = 2024 - df['year']
    df['km_per_year'] = df['km_driven'] / (df['car_age'] + 1)
    df['engine_power_ratio'] = df['engine'] / 1000

    # Encode categorical features
    le = LabelEncoder()
    cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
    encoders = {}
    for col in cat_cols:
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Select features
    feature_cols = [
        'car_age', 'km_driven', 'km_per_year', 'engine', 'engine_power_ratio',
        'mileage', 'seats',
        'name_enc', 'fuel_enc', 'seller_type_enc', 'transmission_enc', 'owner_enc'
    ]

    X = df[feature_cols]
    y = df['selling_price']

    return X, y, encoders, feature_cols


# ─────────────────────────────────────────────
# 3. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(X, y):
    """Train multiple models and return the best one."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=15,
                                               min_samples_split=4, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                        max_depth=5, random_state=42)
    }

    results = {}
    trained_models = {}

    print("\n" + "="*60)
    print("  CAR PRICE PREDICTION — MODEL EVALUATION")
    print("="*60)

    for name, model in models.items():
        # Use scaled data for Linear Regression only
        X_tr = X_train_sc if name == 'Linear Regression' else X_train
        X_te = X_test_sc if name == 'Linear Regression' else X_test

        model.fit(X_tr, y_train)
        preds = model.predict(X_te)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
        trained_models[name] = model

        print(f"\n  {name}")
        print(f"    RMSE : ₹{rmse:>12,.0f}")
        print(f"    MAE  : ₹{mae:>12,.0f}")
        print(f"    R²   :  {r2:.4f}")

    # Best model by R²
    best_name = max(results, key=lambda k: results[k]['R²'])
    print(f"\n  ✅ Best Model: {best_name} (R² = {results[best_name]['R²']:.4f})")
    print("="*60)

    # Feature importance (Random Forest)
    rf = trained_models['Random Forest']
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    print("\n  🔍 Top Feature Importances (Random Forest):")
    for feat, imp in importances.nlargest(6).items():
        bar = "█" * int(imp * 60)
        print(f"    {feat:<22} {bar} {imp:.3f}")

    return trained_models[best_name], scaler, results


# ─────────────────────────────────────────────
# 4. PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_price(model, encoders, feature_cols, car_details):
    """Predict the price of a single car."""
    df = pd.DataFrame([car_details])
    df['car_age'] = 2024 - df['year']
    df['km_per_year'] = df['km_driven'] / (df['car_age'] + 1)
    df['engine_power_ratio'] = df['engine'] / 1000

    for col in ['name', 'fuel', 'seller_type', 'transmission', 'owner']:
        le = encoders[col]
        val = df[col].astype(str).values[0]
        if val in le.classes_:
            df[col + '_enc'] = le.transform([val])
        else:
            df[col + '_enc'] = 0  # Unknown category fallback

    X_pred = df[feature_cols]
    price = model.predict(X_pred)[0]
    return max(0, price)


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n  Loading dataset...")
    df = generate_dataset(n=1500)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Price range: ₹{df['selling_price'].min():,.0f} — ₹{df['selling_price'].max():,.0f}")

    print("\n  Preprocessing data...")
    X, y, encoders, feature_cols = preprocess(df)
    print(f"  Features: {list(X.columns)}")

    best_model, scaler, results = train_and_evaluate(X, y)

    # Sample prediction
    sample_car = {
        'name': 'Honda', 'year': 2019, 'km_driven': 45000,
        'fuel': 'Petrol', 'seller_type': 'Dealer', 'transmission': 'Manual',
        'owner': 'First Owner', 'mileage': 17.5, 'engine': 1498, 'seats': 5
    }
    predicted = predict_price(best_model, encoders, feature_cols, sample_car)
    print(f"\n  🚗 Sample Prediction")
    print(f"     Car    : {sample_car['name']} {sample_car['year']} ({sample_car['fuel']})")
    print(f"     KM     : {sample_car['km_driven']:,}")
    print(f"     Predicted Price: ₹{predicted:,.0f}")
    print()
