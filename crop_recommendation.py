import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import os

warnings.filterwarnings('ignore')

# ============================================================
# SECTION 1: DATA LOADING AND GENERATION
# ============================================================

def generate_sample_data():
    
    np.random.seed(42)
    
    # Define crop types and their optimal conditions
    crops = [
        'Rice', 'Wheat', 'Corn', 'Cotton', 'Sugarcane', 
        'Coffee', 'Tea', 'Banana', 'Apple', 'Grapes',
        'Tomato', 'Potato', 'Onion', 'Carrot', 'Lettuce'
    ]
    
    # Number of samples per crop
    samples_per_crop = 200
    total_samples = len(crops) * samples_per_crop
    
    data = []
    
    for crop in crops:
        # Define optimal ranges for each crop (approximate values)
        if crop == 'Rice':
            n_range, p_range, k_range = (60, 100), (30, 60), (40, 80)
            temp_range, humid_range = (20, 35), (70, 95)
            ph_range, rain_range = (5.5, 7.0), (100, 250)
        elif crop == 'Wheat':
            n_range, p_range, k_range = (40, 80), (20, 50), (30, 60)
            temp_range, humid_range = (10, 25), (40, 70)
            ph_range, rain_range = (6.0, 7.5), (50, 150)
        elif crop == 'Corn':
            n_range, p_range, k_range = (80, 120), (40, 70), (50, 80)
            temp_range, humid_range = (18, 32), (50, 80)
            ph_range, rain_range = (5.8, 7.0), (60, 180)
        elif crop == 'Cotton':
            n_range, p_range, k_range = (50, 90), (30, 60), (40, 70)
            temp_range, humid_range = (20, 35), (40, 70)
            ph_range, rain_range = (5.5, 8.0), (50, 150)
        elif crop == 'Sugarcane':
            n_range, p_range, k_range = (100, 150), (50, 80), (80, 120)
            temp_range, humid_range = (20, 35), (60, 85)
            ph_range, rain_range = (6.0, 7.5), (100, 200)
        elif crop == 'Coffee':
            n_range, p_range, k_range = (60, 100), (30, 50), (40, 70)
            temp_range, humid_range = (15, 28), (60, 80)
            ph_range, rain_range = (5.0, 6.5), (100, 200)
        elif crop == 'Tea':
            n_range, p_range, k_range = (40, 80), (20, 40), (30, 60)
            temp_range, humid_range = (15, 30), (70, 90)
            ph_range, rain_range = (4.5, 6.0), (120, 250)
        elif crop == 'Banana':
            n_range, p_range, k_range = (80, 120), (40, 70), (80, 120)
            temp_range, humid_range = (25, 35), (70, 95)
            ph_range, rain_range = (5.5, 7.0), (100, 250)
        elif crop == 'Apple':
            n_range, p_range, k_range = (40, 70), (20, 40), (40, 60)
            temp_range, humid_range = (10, 25), (40, 60)
            ph_range, rain_range = (6.0, 7.0), (50, 150)
        elif crop == 'Grapes':
            n_range, p_range, k_range = (50, 80), (30, 50), (50, 80)
            temp_range, humid_range = (15, 30), (40, 70)
            ph_range, rain_range = (6.0, 7.5), (50, 120)
        elif crop == 'Tomato':
            n_range, p_range, k_range = (50, 100), (30, 60), (40, 80)
            temp_range, humid_range = (15, 30), (50, 80)
            ph_range, rain_range = (6.0, 6.8), (50, 150)
        elif crop == 'Potato':
            n_range, p_range, k_range = (60, 100), (40, 70), (60, 100)
            temp_range, humid_range = (10, 25), (50, 80)
            ph_range, rain_range = (5.0, 6.5), (50, 150)
        elif crop == 'Onion':
            n_range, p_range, k_range = (40, 80), (20, 50), (40, 70)
            temp_range, humid_range = (15, 28), (50, 75)
            ph_range, rain_range = (6.0, 7.0), (50, 120)
        elif crop == 'Carrot':
            n_range, p_range, k_range = (40, 70), (30, 60), (50, 80)
            temp_range, humid_range = (15, 25), (50, 80)
            ph_range, rain_range = (6.0, 7.0), (40, 100)
        elif crop == 'Lettuce':
            n_range, p_range, k_range = (30, 60), (20, 40), (30, 60)
            temp_range, humid_range = (10, 20), (60, 85)
            ph_range, rain_range = (6.0, 7.0), (30, 80)
        else:
            n_range, p_range, k_range = (50, 100), (30, 60), (40, 80)
            temp_range, humid_range = (15, 30), (50, 80)
            ph_range, rain_range = (5.5, 7.0), (50, 150)
        
        for _ in range(samples_per_crop):
            # Generate values within range with some noise
            n = np.random.uniform(n_range[0], n_range[1])
            p = np.random.uniform(p_range[0], p_range[1])
            k = np.random.uniform(k_range[0], k_range[1])
            temp = np.random.uniform(temp_range[0], temp_range[1])
            humidity = np.random.uniform(humid_range[0], humid_range[1])
            ph = np.random.uniform(ph_range[0], ph_range[1])
            rainfall = np.random.uniform(rain_range[0], rain_range[1])
            
            data.append([n, p, k, temp, humidity, ph, rainfall, crop])
    
    # Create DataFrame
    columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    df = pd.DataFrame(data, columns=columns)
    
    return df


def load_data():
    
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    try:
        # Try to load from UCI ML Repository
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/crop-movement/crop.data"
        # This might not work, so we'll use generated data as fallback
        df = pd.read_csv(url, header=None)
        print("Loaded data from UCI Repository")
    except:
        print("Using generated sample data...")
        df = generate_sample_data()
        print(f"Generated {len(df)} samples for {df['label'].nunique()} crop types")
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nCrop Types: {df['label'].unique()}")
    print(f"Number of Crops: {df['label'].nunique()}")
    
    return df


# ============================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================

def preprocess_data(df):
    
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Features: {list(X.columns)}")
    print(f"Target classes: {list(label_encoder.classes_)}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nData preprocessing completed!")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder


# ============================================================
# SECTION 3: MODEL TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test, label_encoder):
    
    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATING MODELS")
    print("=" * 60)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM (RBF Kernel)': SVC(kernel='rbf', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        trained_models[name] = model
        
        print(f"{name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Find best model
    best_model_name = max(results, key=results.get)
    best_accuracy = results[best_model_name]
    
    print("\n" + "-" * 60)
    print("MODEL COMPARISON RESULTS")
    print("-" * 60)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("\nRank | Model                    | Accuracy")
    print("-" * 60)
    for rank, (model_name, acc) in enumerate(sorted_results, 1):
        print(f"{rank:4d} | {model_name:23s} | {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_model_name} with {best_accuracy*100:.2f}% accuracy")
    print("=" * 60)
    
    return trained_models, results, best_model_name


def evaluate_best_model(trained_models, best_model_name, X_test, y_test, label_encoder):
    """
    Detailed evaluation of the best model.
    """
    print("\n" + "=" * 60)
    print(f"DETAILED EVALUATION: {best_model_name}")
    print("=" * 60)
    
    best_model = trained_models[best_model_name]
    y_pred = best_model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return best_model


# ============================================================
# SECTION 4: PREDICTION FUNCTION
# ============================================================

def predict_crop(model, scaler, label_encoder, features):

    # Convert to numpy array and reshape
    features = np.array(features).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)
    predicted_crop = label_encoder.inverse_transform(prediction)[0]
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
        prob_dict = {
            crop: round(prob, 4) 
            for crop, prob in zip(label_encoder.classes_, probabilities)
        }
        # Sort by probability
        top_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    else:
        top_predictions = [(predicted_crop, 1.0)]
        prob_dict = {}
    
    return {
        'recommended_crop': predicted_crop,
        'confidence': float(max(probabilities)) if hasattr(model, 'predict_proba') else 1.0,
        'all_probabilities': prob_dict,
        'top_5_predictions': top_predictions
    }


def make_sample_predictions(trained_models, best_model_name, scaler, label_encoder):
    """
    Make sample predictions to demonstrate the system.
    """
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    best_model = trained_models[best_model_name]
    
    # Sample test cases
    test_cases = [
        {
            'name': "Tropical Climate - High Rainfall",
            'features': [90, 40, 80, 28, 85, 6.5, 200]
        },
        {
            'name': "Temperate Climate - Moderate Conditions",
            'features': [60, 35, 50, 20, 65, 6.5, 100]
        },
        {
            'name': "Hot & Dry Climate",
            'features': [70, 50, 60, 32, 45, 7.0, 60]
        },
        {
            'name': "Cool Climate - Low Rainfall",
            'features': [50, 30, 45, 15, 55, 6.8, 50]
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print(f"Input: N={test_case['features'][0]}, P={test_case['features'][1]}, "
              f"K={test_case['features'][2]}, Temp={test_case['features'][3]}, "
              f"Humidity={test_case['features'][4]}, pH={test_case['features'][5]}, "
              f"Rainfall={test_case['features'][6]}")
        
        result = predict_crop(best_model, scaler, label_encoder, test_case['features'])
        
        print(f"Recommended Crop: {result['recommended_crop']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print("Top 5 Predictions:")
        for crop, prob in result['top_5_predictions']:
            print(f"  - {crop}: {prob*100:.2f}%")


# ============================================================
# SECTION 5: MAIN EXECUTION
# ============================================================

def main():
    """
    Main function to run the crop recommendation system.
    """
    print("\n" + "=" * 60)
    print("   AI-BASED CROP RECOMMENDATION SYSTEM")
    print("=" * 60)
    print("\nThis system recommends optimal crops based on:")
    print("  - Soil nutrients (N, P, K)")
    print("  - Environmental conditions (Temperature, Humidity)")
    print("  - Soil pH and Rainfall")
    print("\nUsing Multiple ML Algorithms for best accuracy!")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data(df)
    
    # Step 3: Train and evaluate models
    trained_models, results, best_model_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, label_encoder
    )
    
    # Step 4: Detailed evaluation of best model
    best_model = evaluate_best_model(
        trained_models, best_model_name, X_test, y_test, label_encoder
    )
    
    # Step 5: Make sample predictions
    make_sample_predictions(trained_models, best_model_name, scaler, label_encoder)
    
    # Step 6: Interactive prediction
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION")
    print("=" * 60)
    
    while True:
        print("\nEnter soil and environmental conditions:")
        try:
            n = float(input("Nitrogen (N) [0-150]: "))
            p = float(input("Phosphorus (P) [0-100]: "))
            k = float(input("Potassium (K) [0-150]: "))
            temp = float(input("Temperature [0-50°C]: "))
            humidity = float(input("Humidity [0-100%]: "))
            ph = float(input("pH [0-14]: "))
            rainfall = float(input("Rainfall [0-300mm]: "))
            
            features = [n, p, k, temp, humidity, ph, rainfall]
            
            result = predict_crop(best_model, scaler, label_encoder, features)
            
            print("\n" + "-" * 40)
            print(f"RECOMMENDED CROP: {result['recommended_crop']}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
            print("-" * 40)
            
            another = input("\nMake another prediction? (y/n): ").lower()
            if another != 'y':
                break
                
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
    
    print("\n" + "=" * 60)
    print("THANK YOU FOR USING CROP RECOMMENDATION SYSTEM!")
    print("=" * 60)
    
    return trained_models, scaler, label_encoder, best_model


if __name__ == "__main__":
    trained_models, scaler, label_encoder, best_model = main()
