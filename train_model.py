import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 OnePlus Phone Price Prediction Model Training")
print("="*70)

# Load data
print("\n📂 Loading dataset...")
try:
    df = pd.read_csv('data/One_Plus_Phones_cleaned.csv')
    print(f"✓ Dataset loaded from 'data/oneplus_data.csv'")
except FileNotFoundError:
    try:
        df = pd.read_csv('data/One_Plus_Phones_cleaned.csv')
        print(f"✓ Dataset loaded from 'oneplus_data.csv'")
    except:
        print("❌ Dataset not found!")
        print("Please ensure your CSV file is in the correct location")
        exit(1)

print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")

# Display dataset info
print(f"\n📋 Dataset Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i}. {col}")

print(f"\n📊 First 3 rows:")
print(df.head(3))

print(f"\n📈 Dataset Statistics:")
print(df.describe())

# Check for missing values
print(f"\n🔍 Missing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("   ✓ No missing values!")

# Define features and target
# Using numeric features that are relevant for price prediction
feature_cols = [
    'RAM',
    'ROM', 
    'Battery_in_mAh',
    'Display_size_cm',
    'Rating'
]

target_col = 'Discounted_Price'  # Your target column

print(f"\n🎯 Training Configuration:")
print(f"   Features: {feature_cols}")
print(f"   Target: {target_col}")

# Check if all columns exist
missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
if missing_cols:
    print(f"\n❌ ERROR: Following columns not found in dataset:")
    print(f"   {missing_cols}")
    print(f"\n   Available columns: {df.columns.tolist()}")
    exit(1)

print(f"   ✓ All required columns found!")

# Prepare data
print(f"\n🔧 Preprocessing data...")
# Select only required columns
df_model = df[feature_cols + [target_col]].copy()

# Helper to parse memory/storage strings like '8 GB', '128GB', '1 TB' into numeric GB
def parse_size_to_gb(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip().upper()
    # Remove commas and other separators
    s = s.replace(',', '')
    # Match patterns like '8 GB', '128GB', '1 TB', '512 MB'
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(GB|G|TB|T|MB|M)?$", s)
    if not m:
        # Try to extract digits
        digits = re.findall(r"\d+(?:\.\d+)?", s)
        if digits:
            num = float(digits[0])
            return num
        return np.nan
    num = float(m.group(1))
    unit = m.group(2)
    if not unit or unit in ('GB', 'G'):
        return num
    if unit in ('TB', 'T'):
        return num * 1024
    if unit in ('MB', 'M'):
        return num / 1024
    return num

# Convert RAM and ROM to numeric GB values
if 'RAM' in df_model.columns:
    df_model['RAM'] = df_model['RAM'].apply(parse_size_to_gb)
if 'ROM' in df_model.columns:
    df_model['ROM'] = df_model['ROM'].apply(parse_size_to_gb)

# Coerce other numeric-like columns to numeric
for col in ['Battery_in_mAh', 'Display_size_cm', 'Rating', target_col]:
    if col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col].astype(str).str.replace('[^0-9\.\-]','', regex=True), errors='coerce')

# Remove rows with missing values
initial_rows = len(df_model)
df_model = df_model.dropna()
final_rows = len(df_model)

if initial_rows > final_rows:
    print(f"   ⚠️ Removed {initial_rows - final_rows} rows with missing values")
    print(f"   ✓ Clean dataset: {final_rows} rows")
else:
    print(f"   ✓ No missing values to remove")

# Remove duplicates
initial_rows = len(df_model)
df_model = df_model.drop_duplicates()
final_rows = len(df_model)

if initial_rows > final_rows:
    print(f"   ⚠️ Removed {initial_rows - final_rows} duplicate rows")
    print(f"   ✓ Final dataset: {final_rows} rows")

# Separate features and target
X = df_model[feature_cols]
y = df_model[target_col]

print(f"\n📊 Final Data Shape:")
print(f"   Features (X): {X.shape}")
print(f"   Target (y): {y.shape}")

# Split data
print(f"\n✂️ Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Train Random Forest
print(f"\n🤖 Training Random Forest Model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_pred_train = rf_model.predict(X_train)
rf_pred_test = rf_model.predict(X_test)

rf_r2_train = r2_score(y_train, rf_pred_train)
rf_r2_test = r2_score(y_test, rf_pred_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred_test))
rf_mae = mean_absolute_error(y_test, rf_pred_test)

print(f"   ✓ Random Forest trained!")
print(f"      Train R²: {rf_r2_train:.4f}")
print(f"      Test R²:  {rf_r2_test:.4f}")
print(f"      RMSE:     ₹{rf_rmse:,.2f}")

# Train Gradient Boosting
print(f"\n🤖 Training Gradient Boosting Model...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)

# Evaluate Gradient Boosting
gb_pred_train = gb_model.predict(X_train)
gb_pred_test = gb_model.predict(X_test)

gb_r2_train = r2_score(y_train, gb_pred_train)
gb_r2_test = r2_score(y_test, gb_pred_test)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred_test))
gb_mae = mean_absolute_error(y_test, gb_pred_test)

print(f"   ✓ Gradient Boosting trained!")
print(f"      Train R²: {gb_r2_train:.4f}")
print(f"      Test R²:  {gb_r2_test:.4f}")
print(f"      RMSE:     ₹{gb_rmse:,.2f}")

# Select best model
print(f"\n🏆 Model Selection:")
if rf_r2_test > gb_r2_test:
    best_model = rf_model
    model_name = "Random Forest"
    best_r2 = rf_r2_test
    best_rmse = rf_rmse
    best_mae = rf_mae
    print(f"   ✓ Random Forest selected (Better R² Score)")
else:
    best_model = gb_model
    model_name = "Gradient Boosting"
    best_r2 = gb_r2_test
    best_rmse = gb_rmse
    best_mae = gb_mae
    print(f"   ✓ Gradient Boosting selected (Better R² Score)")

# Calculate additional metrics
y_pred = best_model.predict(X_test)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Performance Summary
print(f"\n{'='*70}")
print(f"📊 FINAL MODEL PERFORMANCE - {model_name}")
print(f"{'='*70}")
print(f"R² Score:           {best_r2:.4f}  (Higher is better, max 1.0)")
print(f"RMSE:               ₹{best_rmse:,.2f}  (Lower is better)")
print(f"MAE:                ₹{best_mae:,.2f}  (Lower is better)")
print(f"MAPE:               {mape:.2f}%  (Lower is better)")
print(f"{'='*70}")

# Feature Importance
if hasattr(best_model, 'feature_importances_'):
    print(f"\n📈 Feature Importance:")
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"{'='*70}")
    for idx, row in importance_df.iterrows():
        bar_length = int(row['Importance'] * 50)
        bar = '█' * bar_length
        print(f"{row['Feature']:<20} {bar} {row['Importance']:.4f}")
    print(f"{'='*70}")

# Save model
print(f"\n💾 Saving model...")
joblib.dump(best_model, 'best_model.pkl')
print(f"   ✓ Model saved as 'best_model.pkl'")

# Verify saved model
print(f"\n✅ Verifying saved model...")
loaded_model = joblib.load('best_model.pkl')
print(f"   ✓ Model loaded successfully!")
print(f"   ✓ Model type: {type(loaded_model).__name__}")
print(f"   ✓ Number of features: {loaded_model.n_features_in_}")
print(f"   ✓ Feature names: {feature_cols}")

# Test prediction with sample data
print(f"\n🧪 Testing Model Prediction:")
sample = X_test.iloc[0:1]
prediction = loaded_model.predict(sample)[0]
actual = y_test.iloc[0]
error = abs(prediction - actual)
error_pct = (error / actual) * 100

print(f"{'='*70}")
print(f"Sample Input Features:")
for col in feature_cols:
    print(f"   {col:<20}: {sample[col].values[0]}")
print(f"\nPrediction Results:")
print(f"   Predicted Price:    ₹{prediction:,.2f}")
print(f"   Actual Price:       ₹{actual:,.2f}")
print(f"   Absolute Error:     ₹{error:,.2f}")
print(f"   Percentage Error:   {error_pct:.2f}%")
print(f"{'='*70}")

# Summary
print(f"\n{'='*70}")
print(f"✅ TRAINING COMPLETED SUCCESSFULLY!")
print(f"{'='*70}")
print(f"\n📋 Summary:")
print(f"   • Model Type:       {model_name}")
print(f"   • Features Used:    {len(feature_cols)}")
print(f"   • Training Samples: {len(X_train)}")
print(f"   • Test Samples:     {len(X_test)}")
print(f"   • R² Score:         {best_r2:.4f}")
print(f"   • RMSE:             ₹{best_rmse:,.2f}")
print(f"   • Model File:       best_model.pkl")

print(f"\n💡 Next Steps:")
print(f"   1. Stop current FastAPI server (Ctrl+C)")
print(f"   2. Restart API: python -m uvicorn main:app --reload --port 8000")
print(f"   3. Test prediction in dashboard")
print(f"\n{'='*70}")