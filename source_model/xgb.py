import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline




df = pd.read_csv('data\sale.csv')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract useful date features
df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

# Select features and target variable
features = ['day_of_week', 'month', 'year', 'Category', 'Location', 'Platform','Price','Revenue', 'Discount', 'Units Returned']
target = 'Units Sold'

X = df[features]
y = df[target]

# Define column types
categorical_cols = ['Category', 'Location', 'Platform']
numerical_cols = ['day_of_week', 'month', 'year','Price', 'Revenue', 'Discount', 'Units Returned']

# Pipelines for preprocessing
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])



preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ]
)

# Gradient Boosting
gbr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=10, random_state=42))
])

# XGBoost
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=10, random_state=42, objective='reg:squarederror'))
])