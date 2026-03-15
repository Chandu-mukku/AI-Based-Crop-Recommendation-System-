# AI-Based Crop Recommendation System

A machine learning system that recommends optimal crops based on soil and environmental conditions.

## Features

- **Multiple ML Algorithms**: Compares Random Forest, Decision Tree, SVM, KNN, and Naive Bayes
- **Automatic Model Selection**: Identifies the best performing model
- **Interactive Predictions**: Make predictions for your specific conditions
- **Sample Data Generation**: No external dataset required - generates realistic training data

## Input Features

The system considers the following parameters:
- **N (Nitrogen)**: Soil nitrogen content (0-150)
- **P (Phosphorus)**: Soil phosphorus content (0-100)
- **K (Potassium)**: Soil potassium content (0-150)
- **Temperature**: Ambient temperature in Celsius (0-50)
- **Humidity**: Relative humidity percentage (0-100)
- **pH**: Soil pH level (0-14)
- **Rainfall**: Rainfall in mm (0-300)

## Supported Crops

The model can recommend from 15 different crops:
- Rice, Wheat, Corn, Cotton, Sugarcane
- Coffee, Tea, Banana, Apple, Grapes
- Tomato, Potato, Onion, Carrot, Lettuce

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```
bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```
bash
python crop_recommendation.py
```

The system will:
1. Generate/load training data
2. Preprocess the data
3. Train multiple ML models
4. Compare their accuracy
5. Select the best model
6. Allow you to make interactive predictions

## Example Output

```
============================================================
   AI-BASED CROP RECOMMENDATION SYSTEM
============================================================

Training Random Forest...
Random Forest Accuracy: 0.9850 (98.50%)

Training Decision Tree...
Decision Tree Accuracy: 0.9700 (97.00%)

Training SVM (RBF Kernel)...
SVM (RBF Kernel) Accuracy: 0.9650 (96.50%)

...

BEST MODEL: Random Forest with 98.50% accuracy
```

## Project Structure

```
crop-recommendation-system/
├── crop_recommendation.py   # Main Python script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## How It Works

1. **Data Generation**: Creates realistic sample data based on crop requirements
2. **Preprocessing**: Scales features and encodes labels
3. **Training**: Trains multiple ML classifiers
4. **Evaluation**: Compares accuracy and selects best model
5. **Prediction**: Uses the best model to recommend crops

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn

## License

This project is for educational purposes.
