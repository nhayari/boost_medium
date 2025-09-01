import pandas as pd
import jsonlines

from medium.params import *
from medium.deep_learning.preprocessor import MediumPreprocessingPipeline
from sklearn.model_selection import train_test_split

class DeepLearningData:

    def __init__(self):
        self.path_data = PATH_DATA
        self.X_filepath = DATA_TRAIN
        self.y_filepath = DATA_LOG_RECOMMEND
        self.X_test_filepath = DATA_TEST
        self.y_test_filepath = DATA_TEST_LOG_RECOMMEND
        self.num_lines = DATA_SIZE if type(DATA_SIZE) == int else None
        self.test_size = DATA_TEST_SIZE

    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load training data for deep learning.

        Args:
            force_reload (bool, optional): If True, forces reloading of the data even if a cached CSV exists. Defaults to False.

        Returns:
            pd.DataFrame: The loaded and processed data as a pandas DataFrame.
        """
        # Check if a cached CSV file exists
        df = None
        file_path = self.path_data + f'/deep_learning/data_{self.num_lines}.csv' if self.num_lines else self.path_data + '/deep_learning/data_full.csv'

        if not force_reload:
            df = self.load_csv(file_path)

        if df is None:
            try:
                df_x = self.load_json(self.X_filepath)
                df_y = self.load_csv(self.y_filepath)

                df = pd.concat([df_x, df_y['log1p_recommends']], axis=1)

                self.save_to_csv(df, file_path)
            except Exception as generalError:
                print(f"Error reading file {self.X_filepath}: {generalError}")
                return pd.DataFrame()

        print(f"Final DataFrame shape after concatenation: {df.shape}")

        return df

    def load_preprocess_data(self, data: pd.DataFrame = None, force_reload: bool = False) -> pd.DataFrame:
        """
        Load and preprocess the data for deep learning.

        Args:
            data (pd.DataFrame, optional): The data to preprocess. If None, loads the data using load_data().
            force_reload (bool, optional): If True, forces reloading of the data even if a cached CSV exists. Defaults to False.

        Returns:
            pd.DataFrame: The loaded and preprocessed data as a pandas DataFrame.
        """
        preprocessed_df = None
        file_path = self.path_data + f'/deep_learning/preprocessed_data_{self.num_lines}.csv' if self.num_lines else self.path_data + '/deep_learning/preprocessed_data_full.csv'

        if not force_reload:
            preprocessed_df = self.load_csv(file_path)

        if preprocessed_df is None:
            try:
                if data is None:
                    data = self.load_data(force_reload=force_reload)
                # Implement your preprocessing steps here
                preprocessed_df = self.preprocess_data(data)
                self.save_to_csv(preprocessed_df, file_path)
            except Exception as generalError:
                print(f"Error reading file {self.X_filepath}: {generalError}")
                return pd.DataFrame()

        return preprocessed_df

    def load_train_val_split_data(self, preprocessed_data: pd.DataFrame = None, test_size=0.2, random_state=42, force_reload: bool = False) -> pd.DataFrame:
        """
        Load and preprocess the training data.

        Args:
            preprocessed_data (pd.DataFrame, optional): The preprocessed data to split. If None, loads the data using load_preprocess_data().
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 42.
            force_reload (bool, optional): If True, forces reloading of the data even if a cached CSV exists. Defaults to False.

        Returns:
            pd.DataFrame: The loaded and preprocessed training data as a pandas DataFrame.
        """
        X_train, X_val = None, None
        y_train, y_val = None, None

        X_train_path = self.path_data + f'/deep_learning/X_train_{self.num_lines}_{random_state}_{test_size.replace(".", "")}.csv' if self.num_lines else self.path_data + f'/deep_learning/X_train_full_{random_state}_{test_size.replace(".", "")}.csv'
        X_val_path = self.path_data + f'/deep_learning/X_val_{self.num_lines}_{random_state}_{test_size.replace(".", "")}.csv' if self.num_lines else self.path_data + f'/deep_learning/X_val_full_{random_state}_{test_size.replace(".", "")}.csv'

        y_train_path = self.path_data + f'/deep_learning/y_train_{self.num_lines}_{random_state}_{test_size.replace(".", "")}.csv' if self.num_lines else self.path_data + f'/deep_learning/y_train_full_{random_state}_{test_size.replace(".", "")}.csv'
        y_val_path = self.path_data + f'/deep_learning/y_val_{self.num_lines}_{random_state}_{test_size.replace(".", "")}.csv' if self.num_lines else self.path_data + f'/deep_learning/y_val_full_{random_state}_{test_size.replace(".", "")}.csv'

        if not force_reload:
            X_train, y_train = self.load_csv(X_train_path), self.load_csv(y_train_path)
            X_val, y_val = self.load_csv(X_val_path), self.load_csv(y_val_path)

        if X_train is None or X_val is None or y_train is None or y_val is None:
            # Train Test split
            if preprocessed_data is None:
                preprocessed_data = self.load_preprocess_data(force_reload=force_reload)
            X = preprocessed_data[['full_content']]
            y = preprocessed_data['log1p_recommends']
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
            self.save_to_csv(X_train, X_train_path)
            self.save_to_csv(X_val, X_val_path)
            self.save_to_csv(y_train, y_train_path)
            self.save_to_csv(y_val, y_val_path)

        return X_train, X_val, y_train, y_val

    def load_test_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load and preprocess the test data.
        """
        df = None
        file_path = self.path_data + f'/deep_learning/test_data_{self.test_size}.csv' if self.num_lines else self.path_data + '/deep_learning/test_data_full.csv'
        if not force_reload:
            df = self.load_csv(file_path)
            if df is not None:
                try:
                    df_x = self.load_json(self.X_test_filepath)
                    df_y = self.load_csv(self.y_test_filepath)
                    df = pd.concat([df_x, df_y['log1p_recommends']], axis=1)

                    df = self.preprocess_data(df)

                    self.save_to_csv(df, file_path)
                except Exception as e:
                    print(f"Error loading test data: {e}")

        return df

    def load_csv(self, filepath) -> pd.DataFrame:
        """
        Load a cached CSV file into a DataFrame.
        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
            None: If the file could not be read.
        """
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Error reading CSV file {filepath}: {e}")
            return None

    def save_to_csv(self, df: pd.DataFrame, filepath: str):
        """
        Save a DataFrame to a CSV file.
        """
        try:
            df.to_csv(filepath, index=False)
            print(f"DataFrame saved to {filepath}")
        except Exception as e:
            print(f"Error saving DataFrame to CSV {filepath}: {e}")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"🎬 Data preprocessing started...\n")

        # 1. Remove article not from domain name 'medium.com'
        print(f" - Remove articles not on Medium.")
        df = df[df['domain'] == 'medium.com']

        # 2. Only keep title and content columns + target
        df = df[['title', 'content', 'log1p_recommends']].copy()

        # 3. Preprocess data
        preprocessed_df = MediumPreprocessingPipeline().fit_transform(df)

        # 4. Drop rows with missing values in 'title' or 'content'
        preprocessed_df = preprocessed_df.dropna(subset=['title', 'content'])

        # 5. Concat title and content
        preprocessed_df['full_content'] = preprocessed_df['title'] + ' ' + preprocessed_df['content']

        # 6. Only keep full_content and target
        preprocessed_df = preprocessed_df[['full_content', 'log1p_recommends']]

        print("✅ Data preprocessed")

        return preprocessed_df

    def load_json(self, file_path: str):
        records = []
        line_count = 0
        with jsonlines.open(file_path, mode='r') as reader:
            for obj in reader:
                records.append(obj)
                line_count += 1
                # Early stopping
                if self.num_lines is not None and line_count >= int(self.num_lines):
                    break

        print(f"Loaded {len(records)} lines from {file_path}")
        df = pd.DataFrame(records)

        return df
