import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("food_delivery_time.csv").drop_duplicates()
df = df.drop(columns=["order_id", "distance_miles"])

X, y = df.drop(columns="delivery_time_min"), df["delivery_time_min"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "linear": LinearRegression(),
    "gbr": GradientBoostingRegressor(random_state=42),
}
for name, m in models.items():
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", m)])
    cv = -cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="neg_mean_absolute_error")
    print(name, "CV MAE:", cv.mean().round(2))