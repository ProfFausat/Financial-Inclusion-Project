import numpy as np

# --- TRAINING STATISTICS (From Notebook) ---
# Used for manual scaling of numeric features
NUMERIC_STATS = {
    'Age': {'mean': 43.07, 'std': 18.08},
    'Income_quintile': {'mean': 3.19, 'std': 1.41},
    'Education': {'mean': 1.95, 'std': 0.69}
}

# Regional categories (minus 'East Asia & Pacific (excluding high income)' which is 'first' dropped)
REGIONS_DROPPED = 'East Asia & Pacific (excluding high income)'
REGION_CATEGORIES = [
    'Europe & Central Asia (excluding high income)',
    'High income',
    'Latin America & Caribbean (excluding high income)',
    'Middle East & North Africa (excluding high income)',
    'South Asia',
    'Sub-Saharan Africa (excluding high income)'
]

# Ordered list of feature names as expected by the model (22 features)
FEATURE_NAMES = [
    # Scaled Numeric
    'Age_scaled', 'Income_scaled', 'Education_scaled',
    # Binary
    'Is_employed', 'Is_rural', 'Is_female', 'Has_mobile_phone', 'Has_an_ID', 'Used_internet_in_past_7_days',
    'Made_payments_for_insurance', 'Applied_for_loan_using_mobile_phone',
    # Regions
    'Region_Europe & Central Asia (excluding high income)',
    'Region_High income',
    'Region_Latin America & Caribbean (excluding high income)',
    'Region_Middle East & North Africa (excluding high income)',
    'Region_South Asia',
    'Region_Sub-Saharan Africa (excluding high income)',
    # Missing Indicators
    'Age_missing', 'Income_missing', 'Education_missing', 'Employed_missing', 'Mobile_missing'
]

def preprocess_input(data_dict):
    """
    Manually transforms inputs into the 22 features expected by the SVM.
    3 Scaled Numeric + 8 Binary + 6 One-Hot Region + 5 Missing Indicators = 22
    """
    # Mapping from HTML form names to backend feature names
    mapping = {
        'age_years': 'Age',
        'income_quintile': 'Income_quintile',
        'education_level': 'Education',
        'is_employed': 'Is_employed',
        'is_rural': 'Is_rural',
        'is_female': 'Is_female',
        'has_a_mobile_phone': 'Has_mobile_phone',
        'has_national_id': 'Has_an_ID',
        'used_internet_recently': 'Used_internet_in_past_7_days',
        'world_bank_region': 'Region'
    }
    
    # Create mapped_data with standardized keys
    mapped_data = {}
    for form_key, backend_key in mapping.items():
        mapped_data[backend_key] = data_dict.get(form_key)
    
    # Features that are not in the HTML form current version
    mapped_data['Made_payments_for_insurance'] = data_dict.get('Made_payments_for_insurance', 0)
    mapped_data['Applied_for_loan_using_mobile_phone'] = data_dict.get('Applied_for_loan_using_mobile_phone', 0)

    features = []
    
    # 1. Numeric (Scaled)
    for col in ['Age', 'Income_quintile', 'Education']:
        val = float(mapped_data.get(col, 0) or 0)
        scaled_val = (val - NUMERIC_STATS[col]['mean']) / NUMERIC_STATS[col]['std']
        features.append(scaled_val)
        
    # 2. Binary (0/1)
    binary_cols = [
        'Is_employed', 'Is_rural', 'Made_payments_for_insurance', 
        'Used_internet_in_past_7_days', 'Has_mobile_phone', 'Has_an_ID', 
        'Is_female', 'Applied_for_loan_using_mobile_phone'
    ]
    for col in binary_cols:
        val = mapped_data.get(col, 0)
        # Handle strings '0'/'1' or None
        if val is None or str(val).strip() == '':
            features.append(0.0)
        else:
            features.append(float(val))
            
    # 3. Categorical (One-Hot Region)
    input_region = mapped_data.get('Region', '')
    # Handle truncated labels from HTML dropdown
    region_map = {
        'Sub-Saharan Africa': 'Sub-Saharan Africa (excluding high income)',
        'South Asia': 'South Asia',
        'East Asia & Pacific': 'East Asia & Pacific (excluding high income)',
        'Latin America & Caribbean': 'Latin America & Caribbean (excluding high income)',
        'Middle East & North Africa': 'Middle East & North Africa (excluding high income)',
        'Europe & Central Asia': 'Europe & Central Asia (excluding high income)',
        'High income': 'High income'
    }
    canonical_region = region_map.get(input_region, input_region)

    for cat in REGION_CATEGORIES:
        features.append(1.0 if canonical_region == cat else 0.0)
        
    # 4. Missing Indicators (0/1)
    # For a single prediction, we assume the user provided values (not missing) 
    missing_cols = [
        'Is_employed', 'Is_rural', 'Made_payments_for_insurance', 
        'Has_an_ID', 'Applied_for_loan_using_mobile_phone'
    ]
    for col in missing_cols:
        val = mapped_data.get(col)
        is_missing = 1.0 if (val is None or str(val).strip() == '') else 0.0
        features.append(is_missing)
    
    feature_vector = np.array(features).reshape(1, -1)
    return feature_vector
