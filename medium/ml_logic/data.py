import pandas as pd
import jsonlines
import re
from html import unescape
from medium.params import *

HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\s+')

def load_json_from_files(X_filepath, y_filepath, num_lines: int | None = None) -> pd.DataFrame:
    """
    Create DataFrame from the .json files with possibility to limit the number of lines.

    Args:
        X_filepath (str): Path to the JSON file
        y_filepath (str): Path to the CSV file with target values
        num_lines (int, optional): Maximum number of lines to load. If None, loads all.

    Returns:
        pd.DataFrame: DataFrame containing the data
    """
    records = []
    line_count = 0

    try:
        with jsonlines.open(X_filepath, mode='r') as reader:
            for obj in reader:
                records.append(obj)
                line_count += 1
                # Stop if we've reached the requested number of lines
                if num_lines is not None and line_count >= int(num_lines):
                    break
    except Exception as generalError:
        print(f"Error reading file {X_filepath}: {generalError}")
        return pd.DataFrame()

    print(f"Loaded {len(records)} lines from {X_filepath}")

    df = pd.DataFrame(records)
    df_y = load_csv(y_filepath, num_lines=num_lines)

    df_final = pd.concat([df, df_y['log1p_recommends']], axis=1)

    print(f"Final DataFrame shape after concatenation: {df_final.shape}")

    return df_final


def load_csv(filepath, num_lines: int | None = None) -> pd.DataFrame:
    """
    Read CSV file and create DataFrame.

    Args:
        filepath (str): CSV file path
        num_lines (int, optional): Number of lines to read

    Returns:
        pd.DataFrame: DataFrame with columns '_id' and 'log1p_recommends'
    """
    if num_lines:
        df = pd.read_csv(filepath, nrows=num_lines)
    else:
        df = pd.read_csv(filepath)

    # Rename columns for consistency
    df = df.rename(columns={'id': '_id', 'log_recommends': 'log1p_recommends'})

    return df


def convert_dict_columns(df: pd.DataFrame, drop_cols: bool = True) -> pd.DataFrame:
    """
    Extract and flatten columns containing dictionaries in a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing columns to process
    drop_cols : bool, default True
        If True, removes original columns after extraction

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame with new extracted columns
    """
    # Create a copy to avoid modifying the original
    df_result = df.copy()

    # Columns to drop at the end
    cols_to_drop = []

    for col in df.columns:
        # Check if all values are dictionaries
        if df[col].apply(lambda c: isinstance(c, dict)).all():

            # Use json_normalize to flatten the column
            dict_list = df[col].tolist()
            normalized = pd.json_normalize(dict_list)

            # Rename columns with the original column prefix
            normalized.columns = [f'{col}_{col_name}' for col_name in normalized.columns]

            # Keep the original index for alignment
            normalized.index = df.index

            # Add the new columns to the DataFrame
            df_result = pd.concat([df_result, normalized], axis=1)

            # Mark column for deletion if requested
            if drop_cols:
                cols_to_drop.append(col)

    # Remove original columns if requested
    if cols_to_drop:
        df_result.drop(columns=cols_to_drop, inplace=True)

    return df_result


def title_contains_high_non_ascii_ratio(title: str, threshold: float = 0.2) -> bool:
    """
    Check if the title contains a high ratio of non-ASCII characters.

    Parameters:
    -----------
    title : str
        The title string to check.
    threshold : float
        The ratio threshold above which the title is considered to have high non-ASCII content.

    Returns:
    --------
    bool
        True if the ratio of non-ASCII characters exceeds the threshold, False otherwise.
    """
    if not isinstance(title, str) or len(title) == 0:
        return False

    non_ascii_count = sum(1 for char in title if ord(char) > 127)
    ratio = non_ascii_count / len(title)

    return ratio > threshold

def flag_problematic_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag articles that might be problematic instead of removing them.
    This allows the model to learn from these patterns during training.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe

    Returns:
    --------
    pandas.DataFrame
        Dataframe with flags added
    """
    df_flagged = df.copy()

    # Flag articles with high non-ASCII ratio in title
    if 'title' in df_flagged.columns:
        df_flagged['high_non_ascii_title'] = df_flagged['title'].apply(
            title_contains_high_non_ascii_ratio
        ).astype(int)

    # Flag non-Medium articles
    if 'domain' in df_flagged.columns:
        df_flagged['is_medium'] = (df_flagged['domain'] == 'medium.com').astype(int)

    return df_flagged


def remove_constant_columns(df: pd.DataFrame, exclude_cols: list | str | None = None) -> pd.DataFrame:
    """
    Remove columns with only one unique value.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    exclude_cols : list, str, or None
        Column name(s) to exclude from constant checking.

    Returns:
    --------
    pandas.DataFrame
        Dataframe with constant columns removed (except excluded ones)
    """
    # Handle exclude_cols parameter
    if exclude_cols is None:
        exclude_cols = []
    elif isinstance(exclude_cols, str):
        exclude_cols = [exclude_cols]
    elif not isinstance(exclude_cols, (list, tuple, set)):
        raise ValueError("exclude_cols must be None, string, or list-like")

    # Convert to set for faster lookup
    exclude_set = set(exclude_cols)

    # Validate that excluded columns exist in dataframe
    missing_cols = exclude_set - set(df.columns)
    if missing_cols:
        raise ValueError(f"Excluded columns not found in dataframe: {missing_cols}")

    cols_to_keep = []
    for col in df.columns:
        # Keep column if it's excluded OR has more than 1 unique value
        if col in exclude_set:
            cols_to_keep.append(col)
        elif df[col].nunique() > 1:
            cols_to_keep.append(col)

    return df[cols_to_keep]


def clean_data(df: pd.DataFrame, remove_problematic: bool = True) -> pd.DataFrame:
    """
    Clean raw data by flattening dict columns, removing constant columns,
    and optionally filtering problematic articles.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to clean
    remove_problematic : bool
        If True, removes problematic articles (high non-ASCII, non-Medium).
        Should be False during initial preprocessing to preserve data distribution.

    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe
    """
    print("🎬 Data cleaning started...\n")

    if not df.empty:
        print(f"Initial data shape: {df.shape}")

        # 1. Flatten dictionary columns
        df = convert_dict_columns(df, drop_cols=True)
        print(f" - Flattened dictionary columns, total columns now: {df.shape[1]}")

        # 2. Remove constant columns
        if len(df) > 1:
            df = remove_constant_columns(df, exclude_cols=['content', 'tags', 'log1p_recommends'])
            print(f" - Removed constant columns, remaining columns: {df.shape[1]}")

        # 3. Handle problematic articles
        if remove_problematic:
            # Remove articles where title contains >20% non-ASCII characters
            if 'title' in df.columns:
                initial_count = len(df)
                df = df[~df['title'].apply(title_contains_high_non_ascii_ratio)]
                removed_count = initial_count - len(df)
                print(f" - Removed {removed_count} articles with high non-ASCII ratio in title")

            # Remove articles not from domain 'medium.com'
            if 'domain' in df.columns:
                initial_count = len(df)
                df = df[df['domain'] == 'medium.com']
                removed_count = initial_count - len(df)
                print(f" - Removed {removed_count} non-Medium articles")
        else:
            # Just flag them for the model to learn from
            df = flag_problematic_articles(df)
            print(" - Flagged problematic articles (not removed)")

    print("✅ Data cleaned")
    return df


def create_dataframe_to_predict(text: str = "", title: str = "") -> pd.DataFrame:
    """
    Create a DataFrame for prediction with all required columns for the preprocessing pipeline.
    """
    import pandas as pd
    from datetime import datetime
    import time

    # Create a comprehensive DataFrame with all expected columns
    current_time = datetime.now()
    timestamp = int(time.time())

    df = pd.DataFrame({
        'content': [text],
        'title': [title],
        'tags': [""],
        'domain': ['medium.com'],  # Required for cleaning
        'published': [current_time.isoformat()],  # Will become 'published_$date' after flattening
        '_timestamp': [timestamp],
        'author': ['Unknown'],
        'image_url': [None],
        'link_tags': [{}],
        'meta_tags': [{
            'twitter:data1': '5 min read',
            'referrer': 'origin'
        }],
        '_id': ['pred_001'],
        '_spider': ['prediction'],
        'url': ['https://medium.com/prediction']
    })

    return df


def validate_data_integrity(df: pd.DataFrame) -> dict:
    """
    Validate data integrity and return statistics.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to validate

    Returns:
    --------
    dict
        Dictionary with validation statistics
    """
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
    }

    if 'title' in df.columns:
        stats['high_non_ascii_titles'] = df['title'].apply(title_contains_high_non_ascii_ratio).sum()

    if 'domain' in df.columns:
        stats['non_medium_articles'] = (df['domain'] != 'medium.com').sum()

    if 'log1p_recommends' in df.columns:
        stats['target_mean'] = df['log1p_recommends'].mean()
        stats['target_std'] = df['log1p_recommends'].std()
        stats['target_min'] = df['log1p_recommends'].min()
        stats['target_max'] = df['log1p_recommends'].max()

    return stats


def split_data_stratified(df: pd.DataFrame, split_ratio: float = 0.2,
                         stratify_col: str | None = None) -> tuple:
    """
    Split data with optional stratification.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to split
    split_ratio : float
        Proportion of data for validation
    stratify_col : str
        Column name to use for stratification

    Returns:
    --------
    tuple
        (train_df, val_df)
    """
    if stratify_col and stratify_col in df.columns:
        # Use stratified split
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df,
            test_size=split_ratio,
            stratify=df[stratify_col],
            random_state=42
        )
    else:
        # Simple split
        train_length = int(len(df) * (1 - split_ratio))
        train_df = df[:train_length].copy()
        val_df = df[train_length:].copy()

    return train_df, val_df
