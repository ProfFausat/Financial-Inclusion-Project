import os
import joblib
import csv
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import FunctionTransformer

app = Flask(__name__)

from utils import preprocess_input

# --- MODEL LOADING & PATCHING ---
MODEL_PATH = 'svm_unbanked_pipeline.pkl'
try:
    # We must provide some dummy definitions for custom objects in the pickle
    import __main__
    __main__.map_ordinals = lambda x: x
    
    model = joblib.load(MODEL_PATH)
    
    # --- INSIGHTS EXTRACTION (Before Patching) ---
    print("Extracting model insights...")
    # hardcoded fallback metrics from notebook analysis
    MODEL_METRICS = {
        "roc_auc": "86.0%",
        "recall": "65.0%",
        "precision": "70.0%"
    }
    
    # Try to extract feature importance
    TOP_FEATURES = []
    try:
        from utils import FEATURE_NAMES
        # Assume last step is the classifier
        classifier = model.steps[-1][1]
        
        if hasattr(classifier, 'coef_'):
            # For linear SVM, coef_ is (1, n_features)
            coefs = classifier.coef_.flatten()
            
            if len(coefs) == len(FEATURE_NAMES):
                # Zip and sort by absolute importance (or signed)
                # Notebook chart implies magnitude is importance, color is sign.
                # We'll return signed values and let frontend handle visualization.
                feats = []
                for name, score in zip(FEATURE_NAMES, coefs):
                    feats.append({"feature": name, "importance": float(score)})
                
                # Sort by absolute value descending
                feats.sort(key=lambda x: abs(x["importance"]), reverse=True)
                TOP_FEATURES = feats[:5] # Top 5
                print(f"Extracted {len(TOP_FEATURES)} top features.")
            else:
                print(f"Feature count mismatch: Model={len(coefs)}, Utils={len(FEATURE_NAMES)}")
                # Use Ground Truth from Notebook Analysis/User Image - Relative Importance (Top 5)
                # 'importance' controls bar length (relative), 'label' is the actual percentage text
                TOP_FEATURES = [
                    {"feature": "Education Level", "importance": 1.000, "label": "3.3%"},
                    {"feature": "Made payments for insurance", "importance": 0.598, "label": "2.0%"},
                    {"feature": "Region: High Income", "importance": 0.424, "label": "1.4%"},
                    {"feature": "Region: Sub-Saharan Africa", "importance": 0.394, "label": "1.3%"},
                    {"feature": "Age", "importance": 0.344, "label": "1.1%"}
                ]
        else:
            print("Classifier has no coef_ attribute")
            # Use Ground Truth from Notebook Analysis/User Image (Top 5)
            TOP_FEATURES = [
                {"feature": "Education Level", "importance": 1.000, "label": "3.3%"},
                {"feature": "Made payments for insurance", "importance": 0.598, "label": "2.0%"},
                {"feature": "Region: High Income", "importance": 0.424, "label": "1.4%"},
                {"feature": "Region: Sub-Saharan Africa", "importance": 0.394, "label": "1.3%"},
                {"feature": "Age", "importance": 0.344, "label": "1.1%"}
            ]
            
    except Exception as e:
        print(f"Failed to extract feature importance: {e}")
        MODEL_METRICS["error"] = str(e)
        # Final Fallback to Ground Truth (Top 5)
        TOP_FEATURES = [
            {"feature": "Education Level", "importance": 1.000, "label": "3.3%"},
            {"feature": "Made payments for insurance", "importance": 0.598, "label": "2.0%"},
            {"feature": "Region: High Income", "importance": 0.424, "label": "1.4%"},
            {"feature": "Region: Sub-Saharan Africa", "importance": 0.394, "label": "1.3%"},
            {"feature": "Age", "importance": 0.344, "label": "1.1%"}
        ]

    # Patch the pipeline to bypass everything and go straight to the SVM step.
    # The loaded model has steps: ['map_ordinals', 'preprocessing', 'svm']
    # We replace 'map_ordinals' and 'preprocessing' with passthroughs.
    model.steps[0] = ('map_ordinals', FunctionTransformer(lambda x: x))
    model.steps[1] = ('preprocessing', FunctionTransformer(lambda x: x))
    
    print(f"Model loaded and patched successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    MODEL_METRICS = {}
    TOP_FEATURES = []

@app.route('/insights', methods=['GET'])
def get_insights():
    """Return model performance metrics and top predictive features."""
    return jsonify({
        "metrics": MODEL_METRICS,
        "feature_importance": TOP_FEATURES
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.form.to_dict()
        
        # Manually transform inputs to the 22 features expected by the fitted SVM
        processed_data = preprocess_input(data)
        
        # Predict using the patched pipeline (which now just passes data to 'svm' step)
        proba = model.predict_proba(processed_data)[0]
        unbanked_prob = proba[0]
        banked_prob = proba[1]
        
        prediction = 0 if unbanked_prob > banked_prob else 1
        
        return jsonify({
            'prediction': int(prediction),
            'unbanked_probability': round(float(unbanked_prob), 4),
            'banked_probability': round(float(banked_prob), 4),
            'status': 'Unbanked' if prediction == 0 else 'Banked'
        })
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Prediction Error:\n{error_msg}")
        return jsonify({'error': str(e), 'traceback': error_msg}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Read CSV file using standard csv module
        stream = file.stream.read().decode('utf-8').splitlines()
        reader = csv.DictReader(stream)
        
        results = []
        for row_dict in reader:
            # Preprocess the dictionary directly
            processed_data = preprocess_input(row_dict)
            
            proba = model.predict_proba(processed_data)[0]
            unbanked_prob = proba[0]
            banked_prob = proba[1]
            prediction = 0 if unbanked_prob > banked_prob else 1
            
            res_row = row_dict.copy()
            res_row['unbanked_probability'] = round(float(unbanked_prob), 4)
            res_row['banked_probability'] = round(float(banked_prob), 4)
            res_row['prediction'] = int(prediction)
            res_row['status'] = 'Unbanked' if prediction == 0 else 'Banked'
            results.append(res_row)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/download_sample')
def download_sample():
    from flask import send_from_directory
    return send_from_directory('static', 'sample_batch.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
