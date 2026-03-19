#  Car Price Prediction Model
> A machine learning pipeline to predict used car prices — inspired by the Cardeko Car Price Prediction approach.

##  Introduction
This project builds a **used car price prediction system** using supervised machine learning. Given features like brand, age, fuel type, and mileage, the model estimates the fair market value of a used car. It demonstrates the full ML workflow: data collection → preprocessing → model training → evaluation → prediction.

##  Dataset
| Property | Detail |
| **Source** | Synthetic dataset modeled on real Indian used-car market data (e.g., CarDekho / Cardeko CSV) |
| **Records** | 1,500 cars |
| **Target** | `selling_price` (in ₹ INR) |

### Features Used
| Feature | Type | Description |
| `name` | Categorical | Car brand (Maruti, Honda, BMW, etc.) |
| `year` | Numeric | Manufacturing year |
| `km_driven` | Numeric | Total kilometers driven |
| `fuel` | Categorical | Petrol / Diesel / CNG / Electric |
| `seller_type` | Categorical | Individual / Dealer / Trustmark Dealer |
| `transmission` | Categorical | Manual / Automatic |
| `owner` | Categorical | First / Second / Third / Fourth+ Owner |
| `mileage` | Numeric | Fuel efficiency (km/l) |
| `engine` | Numeric | Engine displacement (CC) |
| `seats` | Numeric | Number of seats |

### Engineered Features
| Feature | Formula | Rationale |
| `car_age` | `2024 - year` | More intuitive than raw year |
| `km_per_year` | `km_driven / car_age` | Usage intensity |
| `engine_power_ratio` | `engine / 1000` | Normalized engine size |

### Preprocessing Steps
1. **Drop rows** with missing `selling_price`
2. **Remove outliers** using 1st–99th percentile IQR clipping
3. **Label Encoding** for all categorical columns
4. **Feature Engineering** (3 derived features added)
5. **StandardScaler** applied for Linear Regression baseline

##  Models Implemented
| Model | RMSE | MAE | R² Score |
| Linear Regression | ₹6,12,342 | ₹4,61,392 | 0.231 |
| Random Forest | ₹1,66,039 | ₹97,301 | **0.944** |
| **Gradient Boosting**  | **₹1,22,842** | **₹67,815** | **0.969** |

**Gradient Boosting Regressor** was selected as the best model with an R² of **0.969**.

##  Setup & Run
### 1. Clone the Repository
```
git clone https://github.com/monikag-creator/car-price-prediction-.git
cd car-price-prediction
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run the Model
```
python car_price_model.py
```

### 4. (Optional) Open the Interactive Demo
Open `demo_app.html` in any browser — no server required.

---

##  Results & Insights

### Model Performance Summary
- **Gradient Boosting** achieves R² = 0.969, meaning the model explains ~97% of price variance.
- **RMSE of ₹1.2 lakh** on an average car price range of ₹3–30 lakhs represents strong accuracy.
- **Linear Regression** underperforms (R² = 0.23), confirming non-linear relationships in pricing.

### Key Findings
1. **Brand (name)** is the single most important feature — luxury brands command exponentially higher prices.
2. **Car Age** is the second strongest predictor — depreciation is rapid in the first 3–5 years.
3. **Fuel type** matters: Diesel and Electric cars fetch premiums; CNG cars depreciate faster.
4. **Transmission**: Automatic cars are priced ~8% higher than equivalent manuals.
5. **Owner history**: Price drops ~15% per additional owner.

### Future Improvements
- [ ] Use the real CarDekho dataset (~8,000+ rows) for production accuracy
- [ ] Add `max_power` (bhp) as a feature — strong price predictor
- [ ] Experiment with XGBoost / LightGBM for further gains
- [ ] Deploy as a Flask/FastAPI web service
- [ ] Add SHAP values for explainability
- [ ] Cross-validation with k=10 folds for robust metric estimation

---

##  Project Structure

```
car_price_prediction/
│
├── car_price_model.py     # Main ML pipeline
├── requirements.txt       # Python dependencies
├── demo_app.html          # Interactive browser demo
└── README.md              # This file
```

---

## 🛠 Requirements

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
```

---

*Built as part of a machine learning course project. Inspired by the Cardeko Car Price Prediction tutorial.*
=======
# car-price-prediction-
 car price prediction using Machine Learning
>>>>>>> 5a35417d5f572722e306b7bc0947476637d5d633
