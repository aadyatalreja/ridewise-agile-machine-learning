import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from sklearn.base import BaseEstimator, ClassifierMixin

# VQC: Variational Quantum Classifier implementation
class VQCClassifier(BaseEstimator, ClassifierMixin):
    """Variational Quantum Classifier using PennyLane"""
    
    def __init__(self, n_qubits=3, n_layers=1, random_state=0, steps=20, batch_size=20, learning_rate=0.05):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.random_state = random_state
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        np.random.seed(random_state)
        self.weights = np.random.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3))
        
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface="numpy")
        def _circuit(inputs, weights):
            AngleEmbedding(inputs, wires=range(self.n_qubits))
            StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))
        
        self.circuit = _circuit
        
    def _cost(self, weights, X, y):
        predictions = np.array([self.circuit(x, weights) for x in X])
        return np.mean((predictions - y) ** 2)
        
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = X_scaled * np.pi / 2
        
        y_binary = (y > 0).astype(int)
        
        n_samples = min(X_scaled.shape[0], 100)
        sample_indices = np.random.choice(X_scaled.shape[0], n_samples, replace=False)
        X_scaled = X_scaled[sample_indices]
        y_binary = y_binary[sample_indices]
        
        opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        
        for i in range(self.steps):
            batch_indices = np.random.randint(0, n_samples, size=min(self.batch_size, n_samples))
            X_batch = X_scaled[batch_indices]
            y_batch = y_binary[batch_indices]
            
            self.weights = opt.step(lambda w: self._cost(w, X_batch, y_batch), self.weights)
            
            if (i + 1) % 5 == 0:
                cost = self._cost(self.weights, X_scaled, y_binary)
                print(f"Step {i+1}/{self.steps} - Cost: {cost:.4f}")
                
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_scaled = X_scaled * np.pi / 2
        predictions = np.array([self.circuit(x, self.weights) for x in X_scaled])
        return (predictions > 0.0).astype(int)


# QCNN: Quantum Convolutional Neural Network implementation
class QCNNClassifier(BaseEstimator, ClassifierMixin):
    """Quantum Convolutional Neural Network Classifier using PennyLane"""
    
    def __init__(self, n_qubits=4, random_state=0, steps=20, batch_size=20, learning_rate=0.05):
        self.n_qubits = n_qubits
        self.random_state = random_state
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        np.random.seed(random_state)
        
        # Initialize weights for convolutional layer
        self.conv_weights = np.random.uniform(0, 2 * np.pi, size=(n_qubits//2, 4))
        
        # Initialize weights for pooling layer
        self.pool_weights = np.random.uniform(0, 2 * np.pi, size=(n_qubits//4, 2)) 
        
        # Initialize weights for fully connected layer
        self.fc_weights = np.random.uniform(0, 2 * np.pi, size=(2, 3))
        
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface="numpy")
        def _circuit(inputs, conv_weights, pool_weights, fc_weights):
            # Input embedding
            AngleEmbedding(inputs, wires=range(self.n_qubits))
            
            # Convolutional layer
            for i in range(self.n_qubits // 2):
                wires = [2*i, 2*i + 1]
                # Apply 2-qubit convolution
                qml.RX(conv_weights[i, 0], wires=wires[0])
                qml.RY(conv_weights[i, 1], wires=wires[1])
                qml.CNOT(wires=wires)
                qml.RZ(conv_weights[i, 2], wires=wires[0])
                qml.RZ(conv_weights[i, 3], wires=wires[1])
                qml.CNOT(wires=wires)
            
            # Pooling layer
            for i in range(self.n_qubits // 4):
                wires = [i, i + self.n_qubits // 2]
                # Apply 2-qubit pooling
                qml.CRX(pool_weights[i, 0], wires=wires)
                qml.CRY(pool_weights[i, 1], wires=wires)
            
            # Fully connected layer
            qml.RX(fc_weights[0, 0], wires=0)
            qml.RY(fc_weights[0, 1], wires=0)
            qml.RZ(fc_weights[0, 2], wires=0)
            
            qml.RX(fc_weights[1, 0], wires=1)
            qml.RY(fc_weights[1, 1], wires=1)
            qml.RZ(fc_weights[1, 2], wires=1)
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
        
        self.circuit = _circuit
    
    def _cost(self, params, X, y):
        conv_weights, pool_weights, fc_weights = params
        predictions = np.array([self.circuit(x, conv_weights, pool_weights, fc_weights) for x in X])
        return np.mean((predictions - y) ** 2)
        
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = X_scaled * np.pi / 2
        
        # Pad features to match number of qubits if necessary
        if X_scaled.shape[1] < self.n_qubits:
            padding = np.zeros((X_scaled.shape[0], self.n_qubits - X_scaled.shape[1]))
            X_scaled = np.hstack((X_scaled, padding))
        else:
            X_scaled = X_scaled[:, :self.n_qubits]
        
        y_binary = (y > 0).astype(int)
        
        n_samples = min(X_scaled.shape[0], 100)
        sample_indices = np.random.choice(X_scaled.shape[0], n_samples, replace=False)
        X_scaled = X_scaled[sample_indices]
        y_binary = y_binary[sample_indices]
        
        opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        params = (self.conv_weights, self.pool_weights, self.fc_weights)
        
        for i in range(self.steps):
            batch_indices = np.random.randint(0, n_samples, size=min(self.batch_size, n_samples))
            X_batch = X_scaled[batch_indices]
            y_batch = y_binary[batch_indices]
            
            params = opt.step(lambda p: self._cost(p, X_batch, y_batch), params)
            self.conv_weights, self.pool_weights, self.fc_weights = params
            
            if (i + 1) % 5 == 0:
                cost = self._cost(params, X_scaled, y_binary)
                print(f"Step {i+1}/{self.steps} - Cost: {cost:.4f}")
                
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_scaled = X_scaled * np.pi / 2
        
        # Pad features to match number of qubits if necessary
        if X_scaled.shape[1] < self.n_qubits:
            padding = np.zeros((X_scaled.shape[0], self.n_qubits - X_scaled.shape[1]))
            X_scaled = np.hstack((X_scaled, padding))
        else:
            X_scaled = X_scaled[:, :self.n_qubits]
            
        predictions = np.array([
            self.circuit(x, self.conv_weights, self.pool_weights, self.fc_weights) 
            for x in X_scaled
        ])
        return (predictions > 0.0).astype(int)


class QFNNClassifier(BaseEstimator, ClassifierMixin):
    """Quantum Feedforward Neural Network Classifier using PennyLane"""
    
    def __init__(self, n_qubits=3, n_layers=2, random_state=0, steps=20, batch_size=20, learning_rate=0.05):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.random_state = random_state
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        np.random.seed(random_state)
        
        # Initialize weights for input layer
        self.input_weights = np.random.uniform(0, 2 * np.pi, size=(n_qubits, 3))
        
        # Initialize weights for hidden layers
        self.hidden_weights = np.random.uniform(0, 2 * np.pi, size=(n_layers - 1, n_qubits, 3))
        
        # Initialize weights for output layer
        self.output_weights = np.random.uniform(0, 2 * np.pi, size=(2))
        
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface="numpy")
        def _circuit(inputs, input_weights, hidden_weights, output_weights):
            # Input embedding
            AngleEmbedding(inputs, wires=range(self.n_qubits))
            
            # Input layer
            for i in range(self.n_qubits):
                qml.RX(input_weights[i, 0], wires=i)
                qml.RY(input_weights[i, 1], wires=i)
                qml.RZ(input_weights[i, 2], wires=i)
            
            # Create entanglement between qubits
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Hidden layers
            for layer in range(self.n_layers - 1):
                for i in range(self.n_qubits):
                    qml.RX(hidden_weights[layer, i, 0], wires=i)
                    qml.RY(hidden_weights[layer, i, 1], wires=i)
                    qml.RZ(hidden_weights[layer, i, 2], wires=i)
                
                # Create entanglement between qubits in each hidden layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            # Output layer - using first qubit for binary classification
            qml.RX(output_weights[0], wires=0)
            qml.RY(output_weights[1], wires=0)
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
        
        self.circuit = _circuit
    
    def _cost(self, params, X, y):
        input_weights, hidden_weights, output_weights = params
        predictions = np.array([self.circuit(x, input_weights, hidden_weights, output_weights) for x in X])
        return np.mean((predictions - y) ** 2)
        
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = X_scaled * np.pi / 2
        
        # Pad features to match number of qubits if necessary
        if X_scaled.shape[1] < self.n_qubits:
            padding = np.zeros((X_scaled.shape[0], self.n_qubits - X_scaled.shape[1]))
            X_scaled = np.hstack((X_scaled, padding))
        else:
            X_scaled = X_scaled[:, :self.n_qubits]
        
        y_binary = (y > 0).astype(int)
        
        # Limit sample size for faster training
        n_samples = min(X_scaled.shape[0], 100)
        sample_indices = np.random.choice(X_scaled.shape[0], n_samples, replace=False)
        X_scaled = X_scaled[sample_indices]
        y_binary = y_binary[sample_indices]
        
        opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        params = (self.input_weights, self.hidden_weights, self.output_weights)
        
        for i in range(self.steps):
            batch_indices = np.random.randint(0, n_samples, size=min(self.batch_size, n_samples))
            X_batch = X_scaled[batch_indices]
            y_batch = y_binary[batch_indices]
            
            params = opt.step(lambda p: self._cost(p, X_batch, y_batch), params)
            self.input_weights, self.hidden_weights, self.output_weights = params
            
            if (i + 1) % 5 == 0:
                cost = self._cost(params, X_scaled, y_binary)
                print(f"Step {i+1}/{self.steps} - Cost: {cost:.4f}")
                
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_scaled = X_scaled * np.pi / 2
        
        # Pad features to match number of qubits if necessary
        if X_scaled.shape[1] < self.n_qubits:
            padding = np.zeros((X_scaled.shape[0], self.n_qubits - X_scaled.shape[1]))
            X_scaled = np.hstack((X_scaled, padding))
        else:
            X_scaled = X_scaled[:, :self.n_qubits]
            
        predictions = np.array([
            self.circuit(x, self.input_weights, self.hidden_weights, self.output_weights) 
            for x in X_scaled
        ])
        return (predictions > 0.0).astype(int)


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


def vqc_classifier(X_train, X_test, y_train, y_test):
    print("Training VQC classifier...")
    clf = VQCClassifier(n_qubits=3, n_layers=1, steps=20, batch_size=20, learning_rate=0.05)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    
    return score, report, clf


def qcnn_classifier(X_train, X_test, y_train, y_test):
    print("Training QCNN classifier...")
    clf = QCNNClassifier(n_qubits=4, steps=20, batch_size=20, learning_rate=0.05)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    
    return score, report, clf


def qfnn_classifier(X_train, X_test, y_train, y_test):
    print("Training QFNN classifier...")
    clf = QFNNClassifier(n_qubits=3, n_layers=1, steps=20, batch_size=20, learning_rate=0.05)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    
    return score, report, clf


def compare_all_qnn_models(X_train, X_test, y_train, y_test):
    print("Comparing all QNN models...")
    all_models = {
        "Variational Quantum Classifier (VQC)": None,
        "Quantum Convolutional Neural Network (QCNN)": None,
        "Quantum Feedforward Neural Network (QFNN)": None
    }
    
    print("Training VQC...")
    score, _, _ = vqc_classifier(X_train, X_test, y_train, y_test)
    all_models["Variational Quantum Classifier (VQC)"] = score
    
    print("Training QCNN...")
    score, _, _ = qcnn_classifier(X_train, X_test, y_train, y_test)
    all_models["Quantum Convolutional Neural Network (QCNN)"] = score
    
    print("Training QFNN...")
    score, _, _ = qfnn_classifier(X_train, X_test, y_train, y_test)
    all_models["Quantum Feedforward Neural Network (QFNN)"] = score
    
    df_models = pd.DataFrame(list(all_models.items()), columns=['Model', 'Accuracy (%)'])
    df_models = df_models.sort_values(by='Accuracy (%)', ascending=False)
    
    return df_models


def showMap():
    try:
        plotData = pd.read_csv("Trip history with locations.csv")
        plotData = plotData.sample(n=min(500, len(plotData)), random_state=42)
        Data = pd.DataFrame()
        Data['lat'] = plotData['lat']
        Data['lon'] = plotData['lon']
        return Data
    except Exception as e:
        print(f"Error loading map data: {e}")
        return pd.DataFrame({
            'lat': [38.9 + np.random.random(50)/100],
            'lon': [-77.0 + np.random.random(50)/100]
        })


def accept_user_data(duration, start_station, end_station):
    user_prediction_data = np.array([duration, start_station, end_station]).reshape(1, -1)
    return user_prediction_data


def quick_evaluation():
    print("Loading data...")
    df = loadData()
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, le = preprocessing(df)
    
    print("Running quick model comparison...")
    results = compare_all_qnn_models(X_train, X_test, y_train, y_test)
    
    print("\nModel Comparison Results:")
    print(results)
    
    return results


if __name__ == "__main__":
    quick_evaluation()