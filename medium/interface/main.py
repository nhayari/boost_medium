import numpy as np
import pandas as pd
from pathlib import Path
from medium.params import *
from medium.ml_logic.data import clean_data, load_json_from_files, create_dataframe_to_predict
from medium.ml_logic.registry import load_model, save_model, save_results, save_preprocessor, load_preprocessor
from medium.ml_logic.model import (
    initialize_model, compile_model, train_model, evaluate_model, implemented_model, train_pipeline, predict_pipeline, evaluate_pipeline
)
from medium.ml_logic.preprocessor import (
    preprocess_pred, MediumPreprocessingPipeline
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import time
import pickle


def save_model_with_custom_name(model, custom_name: str) -> bool:
    try:
        print(f"🎬 save_model_with_custom_name starting ({custom_name})................\n")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(PATH_MODELS, f"{custom_name}_{DATA_SIZE}_{timestamp}.pickle")
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        print(f" ✅ save_model_with_custom_name() done \n")
        return True
    except Exception as e:
        print(f"Error saving model with custom name: {e}")
        return False

def preprocess_and_split(split_ratio: float = 0.2, chunk_size: int = 1000,
                         remove_punct: bool = False, remove_stopwords: bool = False,
                         content_only: bool = False, metadata_only: bool = False,
                         model_is_tree: bool = False) -> tuple:
    """
    Load data, split FIRST, then preprocess to avoid data leakage.
    Returns preprocessed train and validation data along with fitted preprocessor.
    """
    print("🎬 preprocess_and_split starting (leak-free approach)................\n")
    print(locals())

    # Load raw data
    data = load_json_from_files(X_filepath=DATA_TRAIN, y_filepath=DATA_LOG_RECOMMEND, num_lines=DATA_SIZE)

    # CRITICAL: Split BEFORE any preprocessing
    train_length = int(len(data) * (1 - split_ratio))
    data_train_raw, data_val_raw = data[:train_length].copy(), data[train_length:].copy()

    print(f"📊 Data split: {len(data_train_raw)} train, {len(data_val_raw)} validation")

    # Create preprocessing pipeline
    preprocessor = MediumPreprocessingPipeline(
        chunk_size=chunk_size,
        remove_punct=remove_punct,
        remove_stopwords=remove_stopwords,
        content_only=content_only,
        metadata_only=metadata_only,
        model_is_tree=model_is_tree,
        show_progress=True
    )

    # FIT preprocessor ONLY on training data
    print("🔧 Fitting preprocessor on training data only...")
    df_train_processed = preprocessor.fit_transform(data_train_raw)

    # TRANSFORM validation data with fitted preprocessor
    print("🔄 Transforming validation data with fitted preprocessor...")
    df_val_processed = preprocessor.transform(data_val_raw)

    # Create file path suffix
    file_path_suffix = (
        f"{'_content_only' if content_only else ''}"
        f"{'_metadata_only' if metadata_only else ''}"
        f"{'_punct_removed' if (not metadata_only) and remove_punct else ''}"
        f"{'_stopwords_removed' if (not metadata_only) and remove_stopwords else ''}"
        f"{'_data_scaled' if (not model_is_tree) and (not content_only) else ''}"
    )

    # Save preprocessed data
    if hasattr(df_train_processed, 'to_csv'):
        train_path = os.path.join(PATH_DATA, f"df_train_processed_{DATA_SIZE}{file_path_suffix}.csv")
        val_path = os.path.join(PATH_DATA, f"df_val_processed_{DATA_SIZE}{file_path_suffix}.csv")
        df_train_processed.to_csv(train_path, index=False) # type: ignore
        df_val_processed.to_csv(val_path, index=False)

    # Save the preprocessor (fitted on train only)
    save_preprocessor(preprocessor, f"pipeline_preprocessor_train_only{file_path_suffix}")

    # Also save components for compatibility
    if hasattr(preprocessor, 'tfidf_vectorizer'):
        save_preprocessor(preprocessor.tfidf_vectorizer, "tfidf_vectorizer_train_only")
    if hasattr(preprocessor, 'std_scaler'):
        save_preprocessor(preprocessor.std_scaler, "standard_scaler_train_only")

    print("✅ preprocess_and_split done (leak-free approach)\n")

    return df_train_processed, df_val_processed, preprocessor


def train(model_name: str, split_ratio: float = 0.2, chunk_size: int = 200,
          remove_punct: bool = True, remove_stopwords: bool = True,
          content_only: bool = False, metadata_only: bool = False,
          tf_idf_min_ratio: float = 0.02, model_is_tree: bool = False):
    """
    Train model with proper data leakage prevention.
    """
    print("🎬 main train starting (leak-free approach)................\n")

    if model_name in ['RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor', 'XGBRegressor']:
        model_is_tree = True

    # Create file path suffix for loading preprocessed data
    file_path_suffix = (
        f"{'_content_only' if content_only else ''}"
        f"{'_metadata_only' if metadata_only else ''}"
        f"{'_punct_removed' if (not metadata_only) and remove_punct else ''}"
        f"{'_stopwords_removed' if (not metadata_only) and remove_stopwords else ''}"
        f"{'_data_scaled' if (not model_is_tree) and (not content_only) else ''}"
    )

    train_file = os.path.join(PATH_DATA, f"df_train_processed_{DATA_SIZE}{file_path_suffix}.csv")
    val_file = os.path.join(PATH_DATA, f"df_val_processed_{DATA_SIZE}{file_path_suffix}.csv")

    # Check if preprocessed data exists
    if os.path.isfile(train_file) and os.path.isfile(val_file):
        print(f"📂 Loading preprocessed data from files...")
        df_train_processed = pd.read_csv(train_file)
        df_val_processed = pd.read_csv(val_file)

        # Separate features and target
        if 'log1p_recommends' in df_train_processed.columns:
            y_train = df_train_processed['log1p_recommends']
            X_train = df_train_processed.drop(columns=['log1p_recommends'])
            y_val = df_val_processed['log1p_recommends']
            X_val = df_val_processed.drop(columns=['log1p_recommends'])
        else:
            raise ValueError("Target variable 'log1p_recommends' not found in preprocessed data")

        # Initialize and train model
        model = initialize_model(model_name)
        model = train_model(model=model, X=X_train, y=y_train)

        # Evaluate on validation set
        val_predictions = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_predictions) # type: ignore

        # Save model
        save_model(model=model)

    else:
        print(f"⚠️ Preprocessed files not found. Running preprocessing with split...")

        # Run preprocessing with proper split
        df_train_processed, df_val_processed, preprocessor = preprocess_and_split(
            split_ratio=split_ratio,
            chunk_size=chunk_size,
            remove_punct=remove_punct,
            remove_stopwords=remove_stopwords,
            content_only=content_only,
            metadata_only=metadata_only,
            model_is_tree=model_is_tree
        )

        # Create pipeline with already fitted preprocessor
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', initialize_model(model_name))
        ])

        # Train only the model part (preprocessor already fitted)
        if 'log1p_recommends' in df_train_processed.columns:
            y_train = df_train_processed['log1p_recommends']
            X_train = df_train_processed.drop(columns=['log1p_recommends'])

            # Train model
            pipeline.named_steps['model'].fit(X_train, y_train)

            # Evaluate
            y_val = df_val_processed['log1p_recommends']
            X_val = df_val_processed.drop(columns=['log1p_recommends'])
            val_predictions = pipeline.named_steps['model'].predict(X_val)
            val_mae = mean_absolute_error(y_val, val_predictions)
        else:
            raise ValueError("Target variable not found in preprocessed data")

        # Save complete pipeline
        custom_name = f"{model_name}{file_path_suffix}"
        save_model_with_custom_name(pipeline, custom_name=custom_name)

    # Save results
    params = {
        "split_ratio": split_ratio,
        "approach": "leak_free_pipeline",
        "model_name": model_name
    }

    save_results(model_name, params=params, metrics=dict(mae=val_mae))
    print(f"✅ Train finished with Val MAE: {val_mae:.4f}")
    print("✅ main train() done (leak-free approach)\n")

    return val_mae


def evaluate(model_name: str, df_test: pd.DataFrame | None = None,
            remove_punct: bool = False, remove_stopwords: bool = False,
            content_only: bool = False, metadata_only: bool = False,
            model_is_tree: bool = False):
    """
    Evaluate model on test set using preprocessor fitted ONLY on training data.
    """
    print("🎬 main evaluate starting (leak-free approach)................\n")

    if model_name in ['RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor', 'XGBRegressor']:
        model_is_tree = True

    if df_test is None:
        # Load test data
        df_test = load_json_from_files(
            X_filepath=DATA_TEST,
            y_filepath=DATA_TEST_LOG_RECOMMEND,
            num_lines=DATA_TEST_SIZE
        )

    # Create file path suffix
    file_path_suffix = (
        f"{'_content_only' if content_only else ''}"
        f"{'_metadata_only' if metadata_only else ''}"
        f"{'_punct_removed' if (not metadata_only) and remove_punct else ''}"
        f"{'_stopwords_removed' if (not metadata_only) and remove_stopwords else ''}"
        f"{'_data_scaled' if (not model_is_tree) and (not content_only) else ''}"
    )

    try:
        # Load the preprocessor that was fitted ONLY on training data
        preprocessor_name = f"pipeline_preprocessor_train_only{file_path_suffix}"
        preprocessor = load_preprocessor(preprocessor_name)

        if preprocessor is None:
            raise ValueError(
                f"❌ No training-only preprocessor found ({preprocessor_name}). "
                "This could indicate data leakage in previous training!"
            )

        # Transform test data using training preprocessor
        print("🔄 Transforming test data with training preprocessor...")
        df_test_processed = preprocessor.transform(df_test)

        # Load model
        model = load_model(model_name)

        if model is None:
            # Try loading pipeline
            custom_name = f"{model_name}{file_path_suffix}"
            model = load_model(custom_name)

            if model is None:
                raise ValueError(f"Model {model_name} not found")

        # Separate features and target
        if 'log1p_recommends' in df_test_processed.columns:
            y_test = df_test_processed['log1p_recommends']
            X_test = df_test_processed.drop(columns=['log1p_recommends'])
        else:
            raise ValueError("Target variable not found in test data")

        # Make predictions
        # if hasattr(model, 'predict'):
        #     y_pred = model.predict(X_test)
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            y_pred = model.named_steps['model'].predict(X_test)
        elif hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            raise ValueError("Model does not have predict method")

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)

        # Transform back to original scale
        y_pred_original = np.expm1(y_pred)

        # Create results DataFrame
        results_df = pd.DataFrame({
            'actual_log1p': y_test[:len(y_pred)],
            'predicted_log1p': y_pred,
            'predicted_recommends': y_pred_original
        })

        # Save predictions
        date_run = time.strftime("%Y%m%d-%H%M%S")
        model_type = model.__class__.__name__ if hasattr(model, '__class__') else "Unknown"
        target_path = os.path.join(PATH_METRICS, f"test_metrics_{DATA_SIZE}_{model_type}_{date_run}.csv")
        results_df.to_csv(target_path, index=False)

        print(f"✅ Test MAE: {mae:.4f}")
        print(f"✅ evaluate done (leak-free approach)")

        return mae

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


def pred(model_name: str, text: str = "", title: str = ""):
    """
    Make prediction using model and preprocessor fitted ONLY on training data.
    """
    print("🎬 pred starting (leak-free approach)................\n")

    # Create prediction DataFrame
    X_pred = create_dataframe_to_predict(text, title)
    print(f"ℹ️ The text: {text[:100]}...")
    print(f"ℹ️ The title: {title}")

    try:
        # Load preprocessor fitted on training data only
        preprocessor = load_preprocessor("pipeline_preprocessor_train_only")

        if preprocessor is None:
            # Try with full name
            preprocessor = load_preprocessor("tfidf_vectorizer_train_only")

            if preprocessor is None:
                raise ValueError(
                    "❌ No training-only preprocessor found. "
                    "Please retrain with leak-free approach."
                )

        # Load model
        model = load_model(model_name)

        if model is None:
            raise ValueError(f"Model {model_name} not found")

        # Preprocess input
        if hasattr(preprocessor, 'transform'):
            # Using full pipeline preprocessor
            X_processed = preprocessor.transform(X_pred)

            # Remove target column if present
            if 'log1p_recommends' in X_processed.columns:
                X_processed = X_processed.drop(columns=['log1p_recommends'])
        else:
            # Using individual preprocessor (backward compatibility)
            data_cleaned = clean_data(X_pred)
            X_processed = preprocess_pred(data_cleaned, preprocessor)

        # Make prediction
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_processed)
        else:
            raise ValueError("Model does not have predict method")

        print(f"ℹ️ Prediction (log1p): {y_pred}")

        # Transform back to original scale
        nb_recommandation = np.expm1(y_pred)
        print(f"ℹ️ Predicted number of claps: {nb_recommandation}")

        print(f"✅ pred() done (leak-free approach)")
        return nb_recommandation

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        raise


def run_all(model_name: str, remove_punct: bool = False, remove_stopwords: bool = False,
            content_only: bool = False, metadata_only: bool = False,
            model_is_tree: bool = False):
    """
    Run complete pipeline with proper data leakage prevention.
    """
    print("🎬 run_all starting (leak-free approach)................\n")

    # This will handle splitting and preprocessing properly
    train(
        model_name=model_name,
        remove_punct=remove_punct,
        remove_stopwords=remove_stopwords,
        content_only=content_only,
        metadata_only=metadata_only,
        model_is_tree=model_is_tree
    )

    # Evaluate using preprocessor fitted only on training data
    evaluate(
        model_name=model_name,
        df_test=None,
        remove_punct=remove_punct,
        remove_stopwords=remove_stopwords,
        content_only=content_only,
        metadata_only=metadata_only,
        model_is_tree=model_is_tree
    )

    print(f"✅ run_all done (leak-free approach)\n")


if __name__ == '__main__':
    pass
