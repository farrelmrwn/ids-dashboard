import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    classification_report,
)

# === Metrik Lengkap: Precision, Recall, F1 Score per Kelas dan Macro Avg ===
def compute_detailed_metrics(true_labels, preds):
    report = classification_report(true_labels, preds, output_dict=True, zero_division=0)

    return {
        'accuracy': report['accuracy'] * 100,
        'precision_0': report.get('0', {}).get('precision', 0) * 100,
        'recall_0': report.get('0', {}).get('recall', 0) * 100,
        'f1_score_0': report.get('0', {}).get('f1-score', 0) * 100,
        'precision_1': report.get('1', {}).get('precision', 0) * 100,
        'recall_1': report.get('1', {}).get('recall', 0) * 100,
        'f1_score_1': report.get('1', {}).get('f1-score', 0) * 100,
        'macro_avg_precision': report['macro avg']['precision'] * 100,
        'macro_avg_recall': report['macro avg']['recall'] * 100,
        'macro_avg_f1': report['macro avg']['f1-score'] * 100,
        'mse': mean_squared_error(true_labels, preds),
        'rmse': np.sqrt(mean_squared_error(true_labels, preds)),
        'r2_score': r2_score(true_labels, preds)
    }

def run_pipeline_unsw(input_file_path: str):
    test_data = pd.read_csv(input_file_path)
    raw_input = test_data.copy()
    test_data['service'] = test_data['service'].replace('-', np.nan)
    test_data = test_data.dropna(subset=['service'])
    raw_input = raw_input.loc[test_data.index]

    cat_cols = ['proto', 'service', 'state']
    data_cat = pd.get_dummies(test_data[cat_cols], columns=cat_cols).astype(int)

    processed_data_path = 'model/'
    ohe_columns = np.load(os.path.join(processed_data_path, 'ohe_columns.npy'), allow_pickle=True)
    for col in ohe_columns:
        if col not in data_cat.columns:
            data_cat[col] = 0
    data_cat = data_cat[ohe_columns]

    test_data = pd.concat([test_data.reset_index(drop=True), data_cat.reset_index(drop=True)], axis=1)
    test_data.drop(columns=cat_cols, inplace=True)

    selected_features = list(np.load(os.path.join(processed_data_path, 'selected_features.npy'), allow_pickle=True))
    selected_features = [col for col in selected_features if col != 'label']
    scaler = joblib.load(os.path.join(processed_data_path, 'minmax_scaler.pkl'))
    training_features = scaler.feature_names_in_

    features_for_scaling = test_data[selected_features].copy()
    for feature in set(training_features) - set(selected_features):
        features_for_scaling[feature] = 0
    features_for_scaling = features_for_scaling[training_features]

    features_scaled = pd.DataFrame(scaler.transform(features_for_scaling), columns=training_features, index=features_for_scaling.index)
    test_data[training_features] = features_scaled
    features = test_data[selected_features]

    models = {
        'KNN': joblib.load(os.path.join(processed_data_path, 'modelsknn_model.pkl')),
        'Random Forest': joblib.load(os.path.join(processed_data_path, 'modelsrf_model.pkl')),
        'Decision Tree': joblib.load(os.path.join(processed_data_path, 'modelsdt_model.pkl')),
        'Naive Bayes': joblib.load(os.path.join(processed_data_path, 'modelsnb_model.pkl')),
        'SVM': joblib.load(os.path.join(processed_data_path, 'modelssvm_model.pkl')),
        'Logistic Regression': joblib.load(os.path.join(processed_data_path, 'modelslr_model.pkl')),
    }

    predictions = {}
    model_metrics = {}

    for name, model in models.items():
        try:
            start = time.time()
            preds = model.predict(features)
            end = time.time()
            latency = (end - start) * 1000

            predictions[name] = preds

            if 'attack_cat' in raw_input.columns:
                true_labels = raw_input['attack_cat'].apply(
                    lambda x: 0 if str(x).strip().lower() == 'normal' else 1
                )
                metrics = compute_detailed_metrics(true_labels, preds)
                metrics['latency'] = latency
                model_metrics[name] = metrics
            else:
                model_metrics[name] = {'latency': latency}
        except Exception:
            predictions[name] = [None] * len(features)
            model_metrics[name] = {'accuracy': 0, 'latency': 0}

    predicted_df = pd.DataFrame(predictions, index=features.index)

    if 'attack_cat' in raw_input.columns:
        raw_input = raw_input.reset_index(drop=True)
        features = features.reset_index(drop=True)
        true_labels = raw_input['attack_cat'].apply(
            lambda x: 0 if str(x).strip().lower() == 'normal' else 1
        )
        predicted_df['True Labels'] = true_labels

    output_df = pd.concat([raw_input.reset_index(drop=True), predicted_df.reset_index(drop=True)], axis=1)

    return output_df, model_metrics

def run_pipeline_nslkdd(input_file_path: str):
    import pandas as pd
    import joblib
    import os
    import time
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        mean_squared_error,
        r2_score
    )

    data_path = 'model/'

    # Load CSV
    test_df = pd.read_csv(input_file_path)

    # Pisahkan fitur dan label
    feature_cols = test_df.columns.drop('label_binary')
    features = test_df[feature_cols]
    true_labels = test_df['label_binary'].astype(int)

    # Load semua model
    models = {
        'KNN': joblib.load(os.path.join(data_path, 'knn_model.pkl')),
        'Random Forest': joblib.load(os.path.join(data_path, 'rf_model.pkl')),
        'Decision Tree': joblib.load(os.path.join(data_path, 'dt_model.pkl')),
        'Naive Bayes': joblib.load(os.path.join(data_path, 'nb_model.pkl')),
        'SVM': joblib.load(os.path.join(data_path, 'svm_rbf_optimized.pkl')),
        'Logistic Regression': joblib.load(os.path.join(data_path, 'lr_model.pkl')),
    }

    predictions = {}
    model_metrics = {}

    for name, model in models.items():
        try:
            start = time.time()
            preds = model.predict(features)
            end = time.time()

            predictions[name] = preds
            report = classification_report(true_labels, preds, output_dict=True, zero_division=0)

            model_metrics[name] = {
                'accuracy': report['accuracy'] * 100,
                'precision_0': report.get('0', {}).get('precision', 0) * 100,
                'recall_0': report.get('0', {}).get('recall', 0) * 100,
                'f1_score_0': report.get('0', {}).get('f1-score', 0) * 100,
                'precision_1': report.get('1', {}).get('precision', 0) * 100,
                'recall_1': report.get('1', {}).get('recall', 0) * 100,
                'f1_score_1': report.get('1', {}).get('f1-score', 0) * 100,
                'macro_avg_precision': report['macro avg']['precision'] * 100,
                'macro_avg_recall': report['macro avg']['recall'] * 100,
                'macro_avg_f1': report['macro avg']['f1-score'] * 100,
                'mse': mean_squared_error(true_labels, preds),
                'rmse': np.sqrt(mean_squared_error(true_labels, preds)),
                'r2_score': r2_score(true_labels, preds),
                'latency': (end - start) * 1000
            }

        except Exception as e:
            predictions[name] = [None] * len(features)
            model_metrics[name] = {
                'accuracy': 0,
                'latency': 0,
                'error': str(e)
            }

    # Tambahkan kolom prediksi dan true label ke hasil akhir
    predicted_df = pd.DataFrame(predictions, index=features.index)
    predicted_df['True Labels'] = true_labels.reset_index(drop=True)

    # Gabungkan dengan input untuk ditampilkan ke user
    output_df = pd.concat([test_df.reset_index(drop=True), predicted_df.reset_index(drop=True)], axis=1)

    return output_df, model_metrics

def run_pipeline(input_file_path: str, dataset='unsw'):
    if dataset == 'unsw':
        return run_pipeline_unsw(input_file_path)
    elif dataset == 'nslkdd':
        return run_pipeline_nslkdd(input_file_path)
    else:
        raise ValueError("Unknown dataset type, choose 'unsw' or 'nslkdd'")