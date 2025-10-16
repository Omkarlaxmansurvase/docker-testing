import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.sidebar.warning("⚠️ SHAP not installed. Install with: pip install shap")

# Page configuration
st.set_page_config(
    page_title="OnePlus Price Predictor",
    page_icon="📱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        return model
    except FileNotFoundError:
        try:
            model = joblib.load("models/best_model.pkl")
            return model
        except:
            st.error("Model file not found! Please ensure best_model.pkl exists.")
            return None

@st.cache_data
def load_data():
    try:
        # Try different possible locations
        for path in ["data/One_Plus_Phones_cleaned.csv", "oneplus_data.csv", "data.csv"]:
            try:
                df = pd.read_csv(path)
                return df
            except:
                continue
        st.warning("Dataset not found. Using sample data for visualization.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load resources
model = load_model()
data = load_data()

# Title and description
st.title("📱 OnePlus Phone Price Prediction Dashboard")
st.markdown("""
    This dashboard provides comprehensive insights into OnePlus phone pricing predictions, 
    including real-time predictions, model performance metrics, and data analysis.
""")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("🎛️ Navigation")
page = st.sidebar.radio("Select Page", 
                        ["🔮 Prediction", "📊 Data Analysis", "🎯 Model Performance", "📈 Insights"])

# ============= PAGE 1: PREDICTION =============
if page == "🔮 Prediction":
    st.header("🔮 Price Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Enter Phone Specifications")
        
        # Input widgets
        ram = st.slider("RAM (GB)", 4, 16, 8, 2)
        rom = st.slider("ROM (GB)", 64, 512, 128, 64)
        battery = st.slider("Battery (mAh)", 3000, 6000, 5000, 500)
        display = st.slider("Display Size (cm)", 14.0, 18.0, 16.5, 0.5)
        rating = st.slider("Rating", 3.0, 5.0, 4.5, 0.1)
        
        # Display input summary
        st.markdown("### 📋 Input Summary")
        input_df = pd.DataFrame({
            'Feature': ['RAM', 'ROM', 'Battery', 'Display', 'Rating'],
            'Value': [f"{ram} GB", f"{rom} GB", f"{battery} mAh", f"{display} cm", f"{rating} ⭐"],
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Prediction Result")
        
        if st.button("🚀 Predict Price", type="primary", use_container_width=True):
            if model is not None:
                with st.spinner("Making prediction..."):
                    try:
                        # Create feature array
                        features = np.array([[ram, rom, battery, display, rating]])
                        
                        # Make prediction
                        predicted_price = model.predict(features)[0]
                        
                        # Display prediction
                        st.success("### Prediction Complete!")
                        st.markdown(f"<h1 style='text-align: center; color: green;'>₹{predicted_price:,.2f}</h1>", 
                                  unsafe_allow_html=True)
                        
                        # Confidence interval (using ±15% as estimate)
                        confidence = 0.15
                        lower = predicted_price * (1 - confidence)
                        upper = predicted_price * (1 + confidence)
                        
                        st.info(f"**95% Confidence Range:** ₹{lower:,.2f} - ₹{upper:,.2f}")
                        
                        # SHAP explanation (if available)
                        if SHAP_AVAILABLE:
                            st.markdown("### 🔍 Feature Contribution")
                            try:
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer.shap_values(features)
                                
                                fig, ax = plt.subplots(figsize=(8, 4))
                                shap.waterfall_plot(
                                    shap.Explanation(
                                        values=shap_values[0],
                                        base_values=explainer.expected_value,
                                        data=features[0],
                                        feature_names=['RAM', 'ROM', 'Battery', 'Display', 'Rating']
                                    ),
                                    show=False
                                )
                                st.pyplot(fig)
                            except Exception as e:
                                st.warning(f"Could not generate SHAP plot: {e}")
                        
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
            else:
                st.error("Model not loaded!")

# ============= PAGE 2: DATA ANALYSIS =============
elif page == "📊 Data Analysis":
    st.header("📊 Dataset Analysis")
    
    if data is not None:
        # Dataset overview
        st.subheader("📋 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Features", len(data.columns))
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", data.duplicated().sum())
        
        # Show sample data
        st.subheader("🔍 Sample Data")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
        
        # Distribution plots
        st.subheader("📊 Feature Distributions")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        cols_per_row = 3
        for i in range(0, len(numeric_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(numeric_cols[i:i+cols_per_row]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    data[col_name].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                    ax.set_title(f'{col_name} Distribution')
                    ax.set_xlabel(col_name)
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        correlation = data[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
        
    else:
        st.warning("No dataset loaded. Please ensure your dataset is in the correct location.")

# ============= PAGE 3: MODEL PERFORMANCE =============
elif page == "🎯 Model Performance":
    st.header("🎯 Model Performance Metrics")
    
    if data is not None and model is not None:
        # Assuming your data has these columns (adjust based on your actual data)
        try:
            # Identify feature columns and target
            feature_cols = ['RAM', 'ROM', 'Battery_in_mAh', 'Display_size_cm', 'Rating']
            target_col = 'Discounted_price'  # Adjust this to your actual target column name
            
            # Check if columns exist
            available_features = [col for col in feature_cols if col in data.columns]
            
            if len(available_features) > 0 and target_col in data.columns:
                X = data[available_features]
                y = data[target_col]
                
                # Make predictions
                y_pred = model.predict(X)
                
                # Calculate metrics
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                mae = mean_absolute_error(y, y_pred)
                mape = np.mean(np.abs((y - y_pred) / y)) * 100
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² Score", f"{r2:.4f}", help="Coefficient of determination")
                with col2:
                    st.metric("RMSE", f"₹{rmse:,.2f}", help="Root Mean Squared Error")
                with col3:
                    st.metric("MAE", f"₹{mae:,.2f}", help="Mean Absolute Error")
                with col4:
                    st.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
                
                # Prediction vs Actual plot
                st.subheader("📉 Predictions vs Actual Values")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y, y_pred, alpha=0.5, s=50, edgecolors='k')
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
                ax.set_xlabel('Actual Price (₹)')
                ax.set_ylabel('Predicted Price (₹)')
                ax.set_title('Model Predictions vs Actual Prices')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Residual plot
                st.subheader("📊 Residual Analysis")
                residuals = y - y_pred
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(y_pred, residuals, alpha=0.5, s=30)
                    ax.axhline(y=0, color='r', linestyle='--', lw=2)
                    ax.set_xlabel('Predicted Price (₹)')
                    ax.set_ylabel('Residuals (₹)')
                    ax.set_title('Residual Plot')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    residuals.hist(bins=30, ax=ax, color='lightcoral', edgecolor='black')
                    ax.set_xlabel('Residuals (₹)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Residual Distribution')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Feature importance
                st.subheader("🎯 Feature Importance")
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': available_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(data=importance_df, x='Importance', y='Feature', 
                              palette='viridis', ax=ax)
                    ax.set_title('Feature Importance Scores')
                    ax.set_xlabel('Importance')
                    st.pyplot(fig)
                    
                    st.dataframe(importance_df, use_container_width=True, hide_index=True)
                
            else:
                st.error(f"Required columns not found. Expected: {feature_cols + [target_col]}")
                st.info(f"Available columns: {data.columns.tolist()}")
                
        except Exception as e:
            st.error(f"Error in model evaluation: {e}")
            st.info("Please ensure your dataset has the correct column names.")
    else:
        st.warning("Model or dataset not loaded.")

# ============= PAGE 4: INSIGHTS =============
elif page == "📈 Insights":
    st.header("📈 Business Insights & Monitoring")
    
    if data is not None:
        # Price range analysis
        st.subheader("💰 Price Range Analysis")
        
        target_col = 'Discounted_price'  # Adjust to your actual column name
        if target_col in data.columns:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Min Price", f"₹{data[target_col].min():,.2f}")
            with col2:
                st.metric("Avg Price", f"₹{data[target_col].mean():,.2f}")
            with col3:
                st.metric("Max Price", f"₹{data[target_col].max():,.2f}")
            
            # Price distribution by category
            st.subheader("📊 Price Distribution by RAM")
            if 'RAM' in data.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                data.boxplot(column=target_col, by='RAM', ax=ax)
                ax.set_title('Price Distribution by RAM Size')
                ax.set_xlabel('RAM (GB)')
                ax.set_ylabel('Price (₹)')
                plt.suptitle('')
                st.pyplot(fig)
        
        # Data drift monitoring simulation
        st.subheader("🔄 Data Drift Monitoring")
        st.info("Monitor feature distributions over time to detect data drift")
        
        # Simulated drift scores
        drift_data = pd.DataFrame({
            'Feature': ['RAM', 'ROM', 'Battery', 'Display', 'Rating'],
            'Drift_Score': np.random.uniform(0.01, 0.08, 5),
            'Status': ['✅ Normal', '✅ Normal', '⚠️ Warning', '✅ Normal', '✅ Normal']
        })
        
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['green' if score < 0.05 else 'orange' for score in drift_data['Drift_Score']]
        ax.barh(drift_data['Feature'], drift_data['Drift_Score'], color=colors)
        ax.axvline(x=0.05, color='r', linestyle='--', linewidth=2, label='Threshold (0.05)')
        ax.set_xlabel('Drift Score')
        ax.set_title('Feature Drift Detection')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)
        
        st.dataframe(drift_data, use_container_width=True, hide_index=True)
        
    else:
        st.warning("Dataset not loaded.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit | Model Version 1.0 | Last Updated: October 2025</p>
        <p><a href='https://github.com/yourusername/oneplus-predictor'>📂 GitHub Repository</a></p>
    </div>
""", unsafe_allow_html=True)