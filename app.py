import pandas as pd
import numpy as np

#Extra
import altair as alt
from datetime import datetime

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


#Model
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#UI
import streamlit as st
from streamlit_option_menu import option_menu





#Sidebar Design

with st.sidebar:
    page = option_menu(
        menu_title="Projects",  # Title of the menu
        options=["📅 Time Forecasting","📊 Data Preview","💊 Supplement Recommendation","📦 Revenue Prediction"],  # Menu options
        icons=["bar-chart-line", "robot"],  # Bootstrap icon names
        menu_icon="cast",  # Icon next to 'Projects'
        default_index=0,  # Default selected index
        styles={
            "container": {"padding": "5px", "background-color": "#2c3e50"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#34495e"},
            "nav-link-selected": {"background-color": "#1abc9c"},
        }
    )
    


# Load and sample data show
@st.cache_data
def load_data():
    df = pd.read_csv("data/sale.csv")
    return df

df = load_data()
raw_df = load_data()


# Data Preprocessing
# Label Data
categorical_cols = df.select_dtypes(include='object').columns

if 'Date' in categorical_cols:
    categorical_cols = categorical_cols.drop('Date')

onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
onehot_encoded_data = onehot_encoder.fit_transform(df[categorical_cols])
onehot_encoded_df = pd.DataFrame(onehot_encoded_data, columns=onehot_encoder.get_feature_names_out(categorical_cols))
df = df.drop(categorical_cols, axis=1)
df = pd.concat([df, onehot_encoded_df], axis=1)


#Scale the Data
numerical_cols = df.select_dtypes(include=np.number).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])




#Data Preview 
def eda_section(df):
    st.title("📊 Exploratory Data Analysis")

    st.subheader("🔍 Dataset Overview")
    st.write("Here’s a preview of the dataset:")
    st.dataframe(raw_df.head(20))

    st.markdown("---")
    st.subheader("📈 Summary Statistics")
    st.write(raw_df.describe(include='all'))

    st.markdown("---")
    st.subheader("🧱 Missing Values")
    missing = raw_df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.write("Columns with missing values:")
        st.dataframe(missing.to_frame("Missing Count"))
    else:
        st.success("No missing values detected!")

    st.markdown("---")
    st.subheader("📊 Distribution Plots")
    selected_column = st.selectbox("Select a numerical column to plot", raw_df.select_dtypes(include='number').columns)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_column], kde=True, ax=ax, bins=30, color='skyblue')
    ax.set_title(f'Distribution of {selected_column}')
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔗 Correlation Heatmap (Numerical Features Only)")
    corr = raw_df.select_dtypes(include='number').corr()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("🗂️ Category Counts")
    selected_cat = st.selectbox("Select a categorical column", raw_df.select_dtypes(include='object').columns)
    st.bar_chart(raw_df[selected_cat].value_counts())

if page == "📊 Data Preview":
    eda_section(raw_df)






if page == "📅 Time Forecasting":

    #Time Forecasting System

    features = [col for col in df.columns if col not in ['Date', 'Units Sold']]
    X = df[features]
    y = df['Units Sold']

    # Time-based train/test split
    split_date = '2020-10-05'
    X_train = X[df['Date'] < split_date]
    y_train = y[df['Date'] < split_date]
    X_test = X[df['Date'] >= split_date]
    y_test = y[df['Date'] >= split_date]
    dates_test = df[df['Date'] >= split_date]['Date']

    # Train both models once
    @st.cache_resource
    def train_models(X_train, y_train, X_test):
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)

        return xgb_pred, lgb_pred

    xgb_pred, lgb_pred = train_models(X_train, y_train, X_test)

    # Streamlit UI
    st.title("📈 Sales Forecasting Dashboard")

    # Model selector
    model_choice = st.selectbox("Select Model", ["XGBoost", "LightGBM"])

    # Display corresponding results
    if model_choice == "XGBoost":
        selected_pred = xgb_pred
    else:
        selected_pred = lgb_pred

    # Prepare results DataFrame
    results = pd.DataFrame({
        'Date': dates_test,
        'Actual': y_test.values,
        'Predicted': selected_pred
    })

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results['Date'], results['Actual'], label='Actual Sales', color='black')
    ax.plot(results['Date'], results['Predicted'], label=f'{model_choice} Predicted', color='teal')
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.set_title(f"Forecast vs Actual Sales ({model_choice})")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Show data table
    with st.expander("📋 Show Forecast Data Table"):
        st.dataframe(results)


    # Optional: Add animation note
    st.write("")
    st.markdown("🔁 Chart updates automatically when you rerun or adjust model parameters.")






if page == "💊 Supplement Recommendation":
    raw_df['Product_Category'] = raw_df['Product Name'] + " (" + raw_df['Category'] + ")"

    # 👉 Step 2: Simulate transaction IDs by grouping every N rows into a basket
    N = 5  # Number of rows per pseudo-transaction
    raw_df['BasketID'] = raw_df.index // N

    # 👉 Step 3: Build a transaction matrix (Basket x Product_Category)
    basket = raw_df.groupby(['BasketID', 'Product_Category'])['Product_Category'] \
            .count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

    # 👉 Step 4: Apply Apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)

    # 👉 Step 5: Generate association rules
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
    rules = rules[(rules['confidence'] >= 0.5) & (rules['lift'] >= 1)]

    # 👉 Step 6: Define recommendation function
    def recommend_with(product_name, category=None, top_n=5):
        if category:
            key = f"{product_name} ({category})"
        else:
            matches = [col for col in basket.columns if product_name.lower() in col.lower()]
            if not matches:
                return f"Product '{product_name}' not found."
            key = matches[0]

        filtered_rules = rules[rules['antecedents'].apply(lambda x: key in x)]
        recommendations = (
            filtered_rules.sort_values('confidence', ascending=False)
            .consequents.apply(lambda x: list(x)).explode().unique()
        )
        return recommendations[:top_n]




    product_names = raw_df['Product Name'].dropna().unique()
    product_names.sort()  # Optional: sort alphabetically

    # App title
    st.title("Recommendation System")

    # Create a dropdown list of products
    selected_product = st.selectbox("Select a Product:", product_names)

    # Recommend button
    if st.button("Recommend"):
        if selected_product:
            # Replace with your actual recommendation function
            recommendations = recommend_with(selected_product)
            st.subheader("Top Recommendations:")
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.warning("Please select a product.")
            

    st.write(raw_df.head())        
            


# Display selected page
if page == "📦 Revenue Prediction":
    st.title("Supplement Revenue Predictor")

    # Step 0: Model selection
    model_options = {
        "Linear Regression": "linear_regression_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
        "XGBoost": "xgb.pkl"
    }

    selected_model_name = st.selectbox("Select a prediction model", list(model_options.keys()))
    model_path = f"model/{model_options[selected_model_name]}"
    model = joblib.load(model_path)

    # Step 1: User input
    selected_date = st.date_input("Choose a date", datetime.today())
    day_of_week = selected_date.weekday()
    month = selected_date.month
    year = selected_date.year

    selected_category = st.selectbox("Select Supplement Category", raw_df['Category'].unique())
    selected_location = st.selectbox("Select Location", raw_df['Location'].unique())
    selected_platform = st.selectbox("Select Platform", raw_df['Platform'].unique())

    # Step 2: Lookup hidden features
    match = raw_df[
        (raw_df['Category'] == selected_category) &
        (raw_df['Location'] == selected_location) &
        (raw_df['Platform'] == selected_platform)
    ]

    if not match.empty:
        price = match['Price'].mean()
        discount = match['Discount'].mean()
        units_returned = match['Units Returned'].mean()
    else:
        st.warning("No matching historical data found — using default values.")
        price = 0
        discount = 0
        units_returned = 0

    # Step 3: Construct input DataFrame
    input_data = {
        'day_of_week': [day_of_week],
        'month': [month],
        'year': [year],
        'Category': [selected_category],
        'Location': [selected_location],
        'Platform': [selected_platform],
        'Price': [price],
        'Discount': [discount],
        'Units Returned': [units_returned]
    }

    input_df = pd.DataFrame(input_data)

    # Step 4: Predict
    if st.button("Predict Revenue"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Revenue ({selected_model_name}): ${prediction:.2f}")

        # Step 5: Plot comparison with historical actual revenue
        # Filter raw_df for matching records
        historical_data = match.copy()
        
        if not historical_data.empty:
            avg_actual_revenue = historical_data['Revenue'].mean()

            # Create a bar chart
            fig, ax = plt.subplots()
            bars = ax.bar(['Actual Avg Revenue', 'Predicted Revenue'], [avg_actual_revenue, prediction], color=['skyblue', 'orange'])
            ax.set_ylabel("Revenue")
            ax.set_title("Predicted vs. Actual Average Revenue")

            # Add value labels
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"${yval:.2f}", ha='center', va='bottom')

            st.pyplot(fig)
        else:
            st.info("No historical revenue data available to plot comparison.")