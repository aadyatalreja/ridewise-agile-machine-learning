import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, ClassifierMixin

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, Bidirectional
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2

# CNN: Convolutional Neural Network implementation for tabular data
class CNNClassifier(BaseEstimator, ClassifierMixin):
    """CNN Classifier for tabular data"""
    
    def __init__(self, input_shape=(3, 1), n_filters=32, kernel_size=2, 
                 epochs=20, batch_size=32, learning_rate=0.001, random_state=0):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        
        # Set random seed
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
    def _build_model(self, input_shape):
        model = Sequential([
            Conv1D(filters=self.n_filters, kernel_size=self.kernel_size, activation='relu', 
                   input_shape=input_shape),
            MaxPooling1D(pool_size=1),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def fit(self, X, y):
        # Preprocess data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape input for Conv1D (samples, timesteps, features)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Create or reset model
        self.model = self._build_model((X_scaled.shape[1], 1))
        
        # Train model with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_reshaped, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        y_pred_prob = self.model.predict(X_reshaped)
        return (y_pred_prob > 0.5).astype(int)


# LSTM: Long Short-Term Memory implementation for tabular data
class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """LSTM Classifier for tabular data"""
    
    def __init__(self, input_shape=(3, 1), lstm_units=64, 
                 epochs=20, batch_size=32, learning_rate=0.001, random_state=0):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        
        # Set random seed
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
    def _build_model(self, input_shape):
        model = Sequential([
            LSTM(self.lstm_units, input_shape=input_shape, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def fit(self, X, y):
        # Preprocess data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape input for LSTM (samples, timesteps, features)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Create or reset model
        self.model = self._build_model((X_scaled.shape[1], 1))
        
        # Train model with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_reshaped, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        y_pred_prob = self.model.predict(X_reshaped)
        return (y_pred_prob > 0.5).astype(int)


# MLP: Multi-Layer Perceptron implementation
class MLPClassifier(BaseEstimator, ClassifierMixin):
    """Multi-Layer Perceptron Classifier"""
    
    def __init__(self, hidden_layers=[128, 64], 
                 epochs=20, batch_size=32, learning_rate=0.001, random_state=0):
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        
        # Set random seed
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
    def _build_model(self, input_dim):
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], activation='relu', input_dim=input_dim))
        model.add(Dropout(0.3))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(0.3))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def fit(self, X, y):
        # Preprocess data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create or reset model
        self.model = self._build_model(X_scaled.shape[1])
        
        # Train model with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_scaled, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        y_pred_prob = self.model.predict(X_scaled)
        return (y_pred_prob > 0.5).astype(int)


# Bidirectional LSTM Classifier
class BiLSTMClassifier(BaseEstimator, ClassifierMixin):
    """Bidirectional LSTM Classifier for tabular data"""
    
    def __init__(self, input_shape=(3, 1), lstm_units=64, 
                 epochs=20, batch_size=32, learning_rate=0.001, random_state=0):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        
        # Set random seed
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
    def _build_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(self.lstm_units, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def fit(self, X, y):
        # Preprocess data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape input for LSTM (samples, timesteps, features)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Create or reset model
        self.model = self._build_model((X_scaled.shape[1], 1))
        
        # Train model with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_reshaped, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        y_pred_prob = self.model.predict(X_reshaped)
        return (y_pred_prob > 0.5).astype(int)


# Wide & Deep Classifier (combines wide linear model with deep network)
class WideDeepClassifier(BaseEstimator, ClassifierMixin):
    """Wide & Deep Classifier for tabular data"""
    
    def __init__(self, input_shape=(3, 1), hidden_layers=[128, 64], 
                 epochs=20, batch_size=32, learning_rate=0.001, random_state=0):
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        
        # Set random seed
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
    def _build_model(self, input_dim):
        # Input layer
        inputs = Input(shape=(input_dim,))
        
        # Wide path (linear model)
        wide = Dense(16, activation='linear', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(inputs)
        
        # Deep path (MLP)
        deep = Dense(self.hidden_layers[0], activation='relu')(inputs)
        deep = BatchNormalization()(deep)
        deep = Dropout(0.3)(deep)
        
        for units in self.hidden_layers[1:]:
            deep = Dense(units, activation='relu')(deep)
            deep = BatchNormalization()(deep)
            deep = Dropout(0.3)(deep)
        
        # Combine wide and deep paths
        combined = Concatenate()([wide, deep])
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(combined)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def fit(self, X, y):
        # Preprocess data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create or reset model
        self.model = self._build_model(X_scaled.shape[1])
        
        # Train model with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_scaled, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        y_pred_prob = self.model.predict(X_scaled)
        return (y_pred_prob > 0.5).astype(int)


def loadData():
    try:
        df = pd.read_csv("2010-capitalbikeshare-tripdata.csv")
        return df.sample(n=min(1000, len(df)), random_state=42)
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame({
            'Duration': np.random.randint(60, 3600, 100),
            'Start station': np.random.randint(1, 100, 100),
            'End station': np.random.randint(1, 100, 100),
            'Member type': np.random.choice(['Member', 'Casual'], 100)
        })


def preprocessing(df):
    X = df.iloc[:, [0, 3, 5]].values  # Duration, Start station, End station
    y = df.iloc[:, -1].values
    
    le = LabelEncoder()
    y = le.fit_transform(y.flatten())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    test_size = min(100, len(X_test))
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]
    
    return X_train, X_test, y_train, y_test, le


def cnn_classifier(X_train, X_test, y_train, y_test):
    print("Training CNN classifier...")
    clf = CNNClassifier(epochs=10, batch_size=32, learning_rate=0.001)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    
    return score, report, clf


def lstm_classifier(X_train, X_test, y_train, y_test):
    print("Training LSTM classifier...")
    clf = LSTMClassifier(epochs=10, batch_size=32, learning_rate=0.001)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    
    return score, report, clf


def mlp_classifier(X_train, X_test, y_train, y_test):
    print("Training MLP classifier...")
    clf = MLPClassifier(hidden_layers=[128, 64, 32], epochs=10, batch_size=32, learning_rate=0.001)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    
    return score, report, clf


def bilstm_classifier(X_train, X_test, y_train, y_test):
    print("Training BiLSTM classifier...")
    clf = BiLSTMClassifier(epochs=10, batch_size=32, learning_rate=0.001)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    
    return score, report, clf


def wide_deep_classifier(X_train, X_test, y_train, y_test):
    print("Training Wide & Deep classifier...")
    clf = WideDeepClassifier(epochs=10, batch_size=32, learning_rate=0.001)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    
    return score, report, clf


def compare_dl_models(X_train, X_test, y_train, y_test):
    print("Comparing Deep Learning models...")
    all_models = {
        "Convolutional Neural Network (CNN)": None,
        "Long Short-Term Memory (LSTM)": None,
        "Multi-Layer Perceptron (MLP)": None,
        "Bidirectional LSTM": None,
        "Wide & Deep": None
    }
    
    print("Training CNN...")
    score, _, _ = cnn_classifier(X_train, X_test, y_train, y_test)
    all_models["Convolutional Neural Network (CNN)"] = score
    
    print("Training LSTM...")
    score, _, _ = lstm_classifier(X_train, X_test, y_train, y_test)
    all_models["Long Short-Term Memory (LSTM)"] = score
    
    print("Training MLP...")
    score, _, _ = mlp_classifier(X_train, X_test, y_train, y_test)
    all_models["Multi-Layer Perceptron (MLP)"] = score
    
    print("Training BiLSTM...")
    score, _, _ = bilstm_classifier(X_train, X_test, y_train, y_test)
    all_models["Bidirectional LSTM"] = score
    
    print("Training Wide & Deep...")
    score, _, _ = wide_deep_classifier(X_train, X_test, y_train, y_test)
    all_models["Wide & Deep"] = score
    
    df_models = pd.DataFrame(list(all_models.items()), columns=['Model', 'Accuracy (%)'])
    df_models = df_models.sort_values(by='Accuracy (%)', ascending=False)
    
    return df_models