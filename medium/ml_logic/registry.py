import glob
import os
import time
import pickle
from typing import Optional, Union, Any
from sklearn.pipeline import Pipeline
from medium.params import *


def save_results(model_name: str, params: dict, metrics: dict) -> bool:
    """
    Persist params & metrics locally on the hard drive at
    "{PATH_PARAMS}/params/{current_timestamp}.pickle"
    "{PATH_METRICS}/metrics/{current_timestamp}.pickle"
    """
    try:
        print("🎬 save_results starting ................\n")
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Save params locally
        if params is not None:
            params_path = os.path.join(PATH_PARAMAS, f"{model_name}_{DATA_SIZE}_{timestamp}.pickle")
            with open(params_path, "wb") as file:
                pickle.dump(params, file)

        # Save metrics locally
        if metrics is not None:
            metrics_path = os.path.join(PATH_METRICS, f"{model_name}_{DATA_SIZE}_{timestamp}.pickle")
            with open(metrics_path, "wb") as file:
                pickle.dump(metrics, file)

        print(f" ✅ save_results() done \n")
        return True
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False


def save_model(model: Any, custom_name: Optional[str] = None) -> bool:
    """
    Persist trained model locally on the hard drive.

    Parameters:
    -----------
    model : Any
        The model or pipeline to save
    custom_name : str, optional
        Custom name for the model. If None, uses the model class name.

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        print("🎬 save_model starting ................\n")
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Determine the model name
        if custom_name:
            model_name = custom_name
        elif isinstance(model, Pipeline):
            # If it's a pipeline, use the model step's class name
            if 'model' in model.named_steps:
                model_name = f"Pipeline_{model.named_steps['model'].__class__.__name__}"
            else:
                model_name = "Pipeline"
        else:
            model_name = model.__class__.__name__

        model_path = os.path.join(PATH_MODELS, f"{model_name}_{DATA_SIZE}_{timestamp}.pickle")

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        print(f" ✅ Model saved as: {model_name}_{DATA_SIZE}_{timestamp}.pickle")
        return True
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False


def load_model(model_identifier: str, latest: bool = True) -> Optional[Any]:
    """
    Load a saved model by name or pattern.

    Parameters:
    -----------
    model_identifier : str
        The model name or pattern to search for
    latest : bool
        If True, returns the most recent model matching the pattern

    Returns:
    --------
    Any
        The loaded model/pipeline, or None if not found
    """
    try:
        print(f"🎬 load_model starting for '{model_identifier}'................\n")

        # Get the model directory
        local_model_directory = os.path.join(PATH_MODELS, "")

        # Search for models matching the identifier
        # Try exact match first
        search_patterns = [
            f"{local_model_directory}{model_identifier}_{DATA_SIZE}_*.pickle",
            f"{local_model_directory}{model_identifier}_*.pickle",
            f"{local_model_directory}*{model_identifier}*.pickle"
        ]

        local_model_paths = []
        for pattern in search_patterns:
            found_paths = glob.glob(pattern)
            if found_paths:
                local_model_paths = found_paths
                print(f"ℹ️ Found {len(found_paths)} models matching pattern: {pattern}")
                break

        if not local_model_paths:
            print(f"⚠️ No model found matching '{model_identifier}'")
            return None

        # Get the most recent model if requested
        if latest:
            most_recent_model_path = sorted(local_model_paths)[-1]
            model_path = most_recent_model_path
        else:
            model_path = local_model_paths[0]

        print(f"📂 Loading model from: {os.path.basename(model_path)}")

        with open(model_path, "rb") as file:
            model = pickle.load(file)

        # Validate the loaded object
        if isinstance(model, Pipeline):
            print(f"✅ Loaded Pipeline with steps: {list(model.named_steps.keys())}")
            if 'model' in model.named_steps:
                print(f"   Model type: {model.named_steps['model'].__class__.__name__}")
        else:
            print(f"✅ Loaded model type: {model.__class__.__name__}")

        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def save_preprocessor(preprocessor: Any, name: str = 'preprocessor',
                     suffix: Optional[str] = None) -> bool:
    """
    Save the preprocessor object locally on the hard drive.

    Parameters:
    -----------
    preprocessor : Any
        The preprocessor object to save
    name : str
        Base name for the preprocessor
    suffix : str, optional
        Additional suffix to add to the name

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        print(f"🎬 save_preprocessor starting (name: {name})................\n")
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Build the full name with optional suffix
        full_name = name
        if suffix:
            full_name = f"{name}_{suffix}"

        preprocessor_path = os.path.join(
            PATH_PREPROCESSOR,
            f"{full_name}_{DATA_SIZE}_{timestamp}.pickle"
        )

        with open(preprocessor_path, "wb") as file:
            pickle.dump(preprocessor, file)

        print(f" ✅ Preprocessor saved as: {full_name}_{DATA_SIZE}_{timestamp}.pickle")
        return True

    except Exception as e:
        print(f"❌ Error saving preprocessor: {e}")
        return False


def load_preprocessor(name: str = 'preprocessor', latest: bool = True) -> Optional[Any]:
    """
    Load the preprocessor object from disk.

    Parameters:
    -----------
    name : str
        Name of the preprocessor to load (can include wildcards)
    latest : bool
        If True, loads the most recent version

    Returns:
    --------
    Any
        The loaded preprocessor, or None if not found
    """
    try:
        print(f"🎬 load_preprocessor starting (name: {name})................\n")

        # Get the preprocessor directory
        local_preprocessor_directory = os.path.join(PATH_PREPROCESSOR, "")

        # Search for preprocessors matching the name
        search_patterns = [
            f"{local_preprocessor_directory}{name}_{DATA_SIZE}_*.pickle",
            f"{local_preprocessor_directory}{name}_*.pickle",
            f"{local_preprocessor_directory}*{name}*.pickle"
        ]

        local_preprocessor_paths = []
        for pattern in search_patterns:
            found_paths = glob.glob(pattern)
            if found_paths:
                local_preprocessor_paths = found_paths
                print(f"ℹ️ Found {len(found_paths)} preprocessors matching: {pattern}")
                break

        if not local_preprocessor_paths:
            print(f"⚠️ No preprocessor found matching '{name}'")
            return None

        # Get the most recent preprocessor if requested
        if latest:
            most_recent_path = sorted(local_preprocessor_paths)[-1]
            preprocessor_path = most_recent_path
        else:
            preprocessor_path = local_preprocessor_paths[0]

        print(f"📂 Loading preprocessor from: {os.path.basename(preprocessor_path)}")

        with open(preprocessor_path, "rb") as file:
            preprocessor = pickle.load(file)

        # Validate and provide info about the loaded preprocessor
        if hasattr(preprocessor, '__class__'):
            print(f"✅ Loaded preprocessor type: {preprocessor.__class__.__name__}")

            # Check for specific preprocessor types
            if hasattr(preprocessor, 'tfidf_vectorizer'):
                print("   Contains TF-IDF vectorizer")
            if hasattr(preprocessor, 'std_scaler'):
                print("   Contains StandardScaler")
            if hasattr(preprocessor, 'feature_columns_'):
                print(f"   Number of features: {len(preprocessor.feature_columns_)}")

        return preprocessor

    except Exception as e:
        print(f"❌ Error loading preprocessor: {e}")
        return None


def list_saved_models(pattern: Optional[str] = None) -> list:
    """
    List all saved models matching a pattern.

    Parameters:
    -----------
    pattern : str, optional
        Pattern to filter models. If None, lists all models.

    Returns:
    --------
    list
        List of model file paths
    """
    try:
        model_directory = os.path.join(PATH_MODELS, "")

        if pattern:
            search_pattern = f"{model_directory}*{pattern}*.pickle"
        else:
            search_pattern = f"{model_directory}*.pickle"

        model_files = glob.glob(search_pattern)

        if model_files:
            print(f"📋 Found {len(model_files)} models:")
            for file in sorted(model_files)[-10:]:  # Show last 10
                file_name = os.path.basename(file)
                file_size = os.path.getsize(file) / (1024 * 1024)  # Convert to MB
                print(f"   - {file_name} ({file_size:.2f} MB)")
        else:
            print("📋 No models found")

        return model_files

    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []


def list_saved_preprocessors(pattern: Optional[str] = None) -> list:
    """
    List all saved preprocessors matching a pattern.

    Parameters:
    -----------
    pattern : str, optional
        Pattern to filter preprocessors. If None, lists all.

    Returns:
    --------
    list
        List of preprocessor file paths
    """
    try:
        preprocessor_directory = os.path.join(PATH_PREPROCESSOR, "")

        if pattern:
            search_pattern = f"{preprocessor_directory}*{pattern}*.pickle"
        else:
            search_pattern = f"{preprocessor_directory}*.pickle"

        preprocessor_files = glob.glob(search_pattern)

        if preprocessor_files:
            print(f"📋 Found {len(preprocessor_files)} preprocessors:")
            for file in sorted(preprocessor_files)[-10:]:  # Show last 10
                file_name = os.path.basename(file)
                file_size = os.path.getsize(file) / (1024 * 1024)  # Convert to MB
                print(f"   - {file_name} ({file_size:.2f} MB)")
        else:
            print("📋 No preprocessors found")

        return preprocessor_files

    except Exception as e:
        print(f"❌ Error listing preprocessors: {e}")
        return []


def cleanup_old_files(days_old: int = 7, dry_run: bool = True) -> dict:
    """
    Clean up old model and preprocessor files.

    Parameters:
    -----------
    days_old : int
        Remove files older than this many days
    dry_run : bool
        If True, only shows what would be deleted without actually deleting

    Returns:
    --------
    dict
        Statistics about cleaned files
    """
    import datetime

    stats = {
        'models_deleted': 0,
        'preprocessors_deleted': 0,
        'space_freed_mb': 0.0
    }

    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)

    # Clean models
    model_files = glob.glob(os.path.join(PATH_MODELS, "*.pickle"))
    for file_path in model_files:
        file_stat = os.stat(file_path)
        if file_stat.st_mtime < cutoff_time:
            file_size_mb = file_stat.st_size / (1024 * 1024)
            if dry_run:
                print(f"Would delete: {os.path.basename(file_path)} ({file_size_mb:.2f} MB)")
            else:
                os.remove(file_path)
                print(f"Deleted: {os.path.basename(file_path)}")
            stats['models_deleted'] += 1
            stats['space_freed_mb'] += file_size_mb

    # Clean preprocessors
    preprocessor_files = glob.glob(os.path.join(PATH_PREPROCESSOR, "*.pickle"))
    for file_path in preprocessor_files:
        file_stat = os.stat(file_path)
        if file_stat.st_mtime < cutoff_time:
            file_size_mb = file_stat.st_size / (1024 * 1024)
            if dry_run:
                print(f"Would delete: {os.path.basename(file_path)} ({file_size_mb:.2f} MB)")
            else:
                os.remove(file_path)
                print(f"Deleted: {os.path.basename(file_path)}")
            stats['preprocessors_deleted'] += 1
            stats['space_freed_mb'] += file_size_mb

    print(f"\n📊 Cleanup {'(DRY RUN)' if dry_run else ''} Statistics:")
    print(f"   Models deleted: {stats['models_deleted']}")
    print(f"   Preprocessors deleted: {stats['preprocessors_deleted']}")
    print(f"   Space freed: {stats['space_freed_mb']:.2f} MB")

    return stats


def validate_model_preprocessor_compatibility(model_name: str) -> bool:
    """
    Validate that a model has a compatible preprocessor.

    Parameters:
    -----------
    model_name : str
        Name of the model to validate

    Returns:
    --------
    bool
        True if compatible preprocessor exists, False otherwise
    """
    print(f"🔍 Validating model-preprocessor compatibility for '{model_name}'...")

    # Load model
    model = load_model(model_name)
    if model is None:
        print("❌ Model not found")
        return False

    # Check if it's a pipeline (self-contained)
    if isinstance(model, Pipeline):
        print("✅ Model is a self-contained pipeline")
        return True

    # For standalone models, check for preprocessor
    preprocessor = load_preprocessor("pipeline_preprocessor_train_only")
    if preprocessor is None:
        print("❌ No compatible preprocessor found")
        return False

    print("✅ Compatible preprocessor found")
    return True
