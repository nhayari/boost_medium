import pandas as pd
import jsonlines

from medium.params import *
from medium.deep_learning.preprocessor import MediumPreprocessingPipeline

class DeepLearningData:

    def __init__(self):
        self.path_data = PATH_DATA
        self.X_filepath = DATA_TRAIN
        self.y_filepath = DATA_LOG_RECOMMEND
        self.num_lines = DATA_SIZE

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
                records = []
                line_count = 0
                with jsonlines.open(self.X_filepath, mode='r') as reader:
                    for obj in reader:
                        records.append(obj)
                        line_count += 1
                        # Early stopping
                        if self.num_lines is not None and line_count >= int(self.num_lines):
                            break

                print(f"Loaded {len(records)} lines from {self.X_filepath}")
                df = pd.DataFrame(records)
                df_y = self.load_csv(self.y_filepath)

                df = pd.concat([df, df_y['log1p_recommends']], axis=1)

                self.save_to_csv(df, file_path)
            except Exception as generalError:
                print(f"Error reading file {self.X_filepath}: {generalError}")
                return pd.DataFrame(records)

        print(f"Final DataFrame shape after concatenation: {df.shape}")

        return df

    def load_preprocess_data(self, data, force_reload: bool = False) -> pd.DataFrame:
        """
        Load and preprocess the data for deep learning.

        Args:
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
                # Implement your preprocessing steps here
                preprocessed_df = self.preprocess_data(data)
                self.save_to_csv(preprocessed_df, file_path)
            except Exception as generalError:
                print(f"Error reading file {self.X_filepath}: {generalError}")
                return pd.DataFrame()

        return preprocessed_df

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

        # 5. Only keep title, content and target
        preprocessed_df = preprocessed_df[['title', 'content', 'log1p_recommends']]

        print("✅ Data preprocessed")

        return preprocessed_df
