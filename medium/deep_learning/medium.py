from medium.deep_learning.data import DeepLearningData
from medium.params import *

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class Medium:

    def __init__(self, model: Sequential = None):
        """
        Initialize the Medium class.
        Args:
            model (Sequential): A pre-trained Keras Sequential model.
        """
        self.model = model
        self.is_fitted = model is not None

    def load_data(self, test_size=0.2, random_state=42, force_reload: bool = False):
        """
        Loads and preprocesses the dataset for deep learning tasks.
        This method performs the following steps:
        1. Initializes the data loader.
        2. Loads the raw data, optionally forcing a reload.
        3. Applies preprocessing to the loaded data.
        4. Splits the preprocessed data into training and validation sets.
        Attributes set:
            - self.data_loader: Instance of DeepLearningData used for data operations.
            - self.data: Raw loaded data.
            - self.preprocessed_data: Data after preprocessing.
            - self.X_train, self.X_val: Features for training and validation.
            - self.y_train, self.y_val: Targets for training and validation.
        Raises:
            Any exceptions raised by the DeepLearningData methods.
        """
        self.data_loader = DeepLearningData()
        self.data = self.data_loader.load_data(force_reload=force_reload)
        self.preprocessed_data = self.data_loader.load_preprocess_data(
            self.data, force_reload=force_reload
        )
        self.X_train, self.X_val, self.y_train, self.y_val = self.data_loader.load_train_val_split_data(
            self.preprocessed_data,
            test_size=test_size,
            random_state=random_state,
            force_reload=force_reload
        )
        return self

    def get_data_loader(self):
        if hasattr(self, 'data_loader'):
            return self.data_loader
        else:
            self.data_loader = DeepLearningData()
            return self.data_loader

    def get_data(self):
        if hasattr(self, 'data'):
            return self.data
        else:
            self.data = self.get_data_loader().load_data()
            return self.data

    def get_preprocessed_data(self):
        if hasattr(self, 'preprocessed_data'):
            return self.preprocessed_data
        else:
            self.preprocessed_data = self.get_data_loader().load_preprocess_data(self.get_data())
            return self.preprocessed_data

    def get_train_val_data(self):
        if hasattr(self, 'X_train') and hasattr(self, 'X_val') and hasattr(self, 'y_train') and hasattr(self, 'y_val'):
            return self.X_train, self.X_val, self.y_train, self.y_val
        else:
            self.X_train, self.X_val, self.y_train, self.y_val = self.get_data_loader().load_train_val_split_data(self.get_preprocessed_data())
            return self.X_train, self.X_val, self.y_train, self.y_val

    def tokenize(self, X):
        """
        Tokenize the text data.
        """
        print("🎬 Tokenization started...\n")
        # Initialize tokenizer
        if hasattr(self, 'tokenizer'):
            tokenizer = self.tokenizer
        else:
            tokenizer = Tokenizer(oov_token='<OOV>')
            # Fit tokenizer on training data
            tokenizer.fit_on_texts(X)

        # Convert texts to sequences
        X_seq = tokenizer.texts_to_sequences(X)

        print("✅ Tokenization completed")

        # Save tokenizer for future use
        self.tokenizer = tokenizer

        return X_seq

    def pad_sequences(self, X, max_length=5000, padding_type='post', truncating_type='post'):
        """
        Pad the sequences to the same length.
        """
        print("🎬 Padding sequences started...\n")
        X_pad = pad_sequences(X, maxlen=max_length, padding=padding_type, truncating=truncating_type)

        print("✅ Padding sequences completed")

        return X_pad

    def get_vocab_size(self):
        """
        Get the vocabulary size of the tokenizer.
        """
        if not hasattr(self, 'tokenizer'):
            raise ValueError("Tokenizer not found. Please tokenize the data first.")

        return len(self.tokenizer.word_index) + 1  # +1 for padding token

    def get_max_sequence_length(self):
        """
        Get the maximum sequence length from the padded sequences.
        """
        if not hasattr(self, 'X_train_pad'):
            raise ValueError("Training data not found. Please pad the sequences first.")

        return self.X_train_pad.shape[1]

    def build_LSTM_model(self, embedding_dim=64):
        """
        Build a simple LSTM(Long Short Term Memory) model for text regression.
        """
        if not hasattr(self, 'X_train_pad') or not hasattr(self, 'X_val_pad'):
            raise ValueError("Training and validation data not found. Please pad the sequences first.")

        vocab_size = self.get_vocab_size()
        max_length = self.get_max_sequence_length()

        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1, activation='linear')  # Linear activation for regression
        ])

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        model.name = "LSTM"
        self.model = model
        self.is_fitted = False

        return self

    def build_CNN_model(self, embedding_dim=128):
        """
        Build a simple CNN(Convolutional Neural Network) model for text regression.
        """
        if not hasattr(self, 'X_train_pad') or not hasattr(self, 'X_val_pad'):
            raise ValueError("Training and validation data not found. Please pad the sequences first.")

        vocab_size = self.get_vocab_size()
        max_length = self.get_max_sequence_length()

        model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        Conv1D(128, 5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(64, 3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')  # Linear activation for regression
    ])

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        model.name = "CNN"
        self.model = model
        self.is_fitted = False

        return self

    def fit(self, epochs=20, batch_size=64):
        """
        Fit the model to the training data.
        """
        if self.model is None:
            raise ValueError("Model is not built yet.")

        if self.is_fitted:
            print("⚠️ Model is already fitted.")
            return self

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
        ]

        print("🎬 Model training started...\n")
        self.model.fit(
            self.X_train_pad,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_pad, self.y_val),
            callbacks=callbacks,
            verbose=2
        )
        self.is_fitted = True
        print("✅ Model training completed")

        return self

    def evaluate(self, X_test_pad, y_test, save_best_only=False):
        """
        Evaluate the model on the test data.
        """
        if self.model and self.is_fitted:
            print("🎬 Model evaluation started...\n")
            results = self.model.evaluate(X_test_pad, y_test)
            print("✅ Model evaluation completed")

            return results
        else:
            if self.model is None and self.is_fitted == False:
                print("⚠️ Model is not built and fitted yet.")
                raise ValueError("Model is not built and fitted yet.")
            else:
                raise ValueError("Model is not built yet.") if self.model is None else ValueError("Model is not fitted yet.")

    def predict(self, X_new):
        """
        Predict the target values for new data.
        """
        if self.model and self.is_fitted:
            print("🎬 Model prediction started...\n")
            predictions = self.model.predict(X_new)
            print("✅ Model prediction completed")
            return predictions
        else:
            if self.model is None and self.is_fitted == False:
                print("⚠️ Model is not built and fitted yet.")
                raise ValueError("Model is not built and fitted yet.")
            else:
                raise ValueError("Model is not built yet.") if self.model is None else ValueError("Model is not fitted yet.")

    def summary(self):
        """
        Print the model summary.
        """
        if self.model:
            return self.model.summary()
        else:
            raise ValueError("Model is not built yet.")

    def get_model_name(self):
        """
        Get the model name.
        """
        if self.model:
            return self.model.name
        else:
            raise ValueError("Model is not built yet.")

    def load_test_data(self, force_reload: bool = False):
        """
        Get the test data.
        """
        if self.data_loader:
            print("Loading test data...")
            df_test = self.data_loader.load_test_data(force_reload=force_reload)
            print(f"Test data loaded with {len(df_test)} records.")
            print("Preprocessing test data...")
            df_test = self.data_loader.preprocess_data(df_test)
            print("Setting test features and targets...")
            self.X_test = df_test[['full_content']]
            self.y_test = df_test['log_recommends']
            print("Test data is ready.")
        else:
            raise ValueError("Data loader is not initialized.")

    def set_attr(self, name, value):
        super().__setattr__(name, value)

    def tokenize_and_pad(self, data, attr_prefix: str, pad_args: dict = {}):
        """
        Tokenize and pad the sequences for the given data.

        """
        if attr_prefix not in ['X_train', 'X_val', 'X_test', 'X_pred']:
            raise ValueError("attr_prefix must be one of 'X_train', 'X_val', 'X_test', or 'X_pred'.")
        seq = self.tokenize(data['full_content'])
        self.set_attr(f'{attr_prefix}_seq', seq)
        pad = self.pad_sequences(seq, **pad_args)
        self.set_attr(f'{attr_prefix}_pad', pad)
