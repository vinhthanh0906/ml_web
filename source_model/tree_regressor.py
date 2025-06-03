
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor


# Load dataset
df = pd.read_csv('source_model\tree_regressor.py')

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

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ]
)

# Complete pipeline with Decision Tree Regressor
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(max_depth=14, min_samples_leaf=10, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model_pipeline.fit(X_train, y_train)

