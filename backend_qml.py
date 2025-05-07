import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.optimize import minimize
from functools import lru_cache
# Optional import for parallel processing
try:
    from joblib import Parallel, delayed
except ImportError:
    pass

# For a real quantum implementation, you would use:
# from qiskit import QuantumCircuit, Aer, execute
# from qiskit.circuit import Parameter
# from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
# from qiskit_machine_learning.neural_networks import SamplerQNN
# from qiskit_machine_learning.algorithms import NeuralNetworkClassifier

# Helper function to load data - vectorized implementation
def loadData():
    # In a real implementation, you would load your actual data
    # For this example, we'll create synthetic data representing bike sharing trips
    np.random.seed(42)
    
    # Create synthetic data
    n_samples = 500
    
    # Features: Duration (seconds), Start station ID, End station ID
    duration = np.random.exponential(scale=900, size=n_samples)  # Trip duration in seconds
    start_station = np.random.randint(1, 50, size=n_samples)     # Start station ID (1-49)
    end_station = np.random.randint(1, 50, size=n_samples)       # End station ID (1-49)
    
    # Target: Member type - Casual or Member
    # We'll create a simple rule: if duration > 900 or stations are far apart, more likely casual
    probs = 1 / (1 + np.exp(-(duration/900 - 1 + 0.05*(np.abs(end_station - start_station)))))
    member_type = np.random.binomial(1, probs)
    member_type_str = np.where(member_type == 1, "Casual", "Member")
    
    # Create DataFrame
    data = pd.DataFrame({
        'Duration': duration,
        'Start station': start_station,
        'End station': end_station,
        'Member type': member_type_str
    })
    
    return data

# Preprocessing function - optimized version
def preprocessing(data):
    # Preprocess the data
    X = data[['Duration', 'Start station', 'End station']].values
    
    # Encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(data['Member type'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Add engineered features - distance between stations (vectorized)
    X_train = np.hstack([X_train, np.abs(X_train[:, 1] - X_train[:, 2]).reshape(-1, 1)])
    X_test = np.hstack([X_test, np.abs(X_test[:, 1] - X_test[:, 2]).reshape(-1, 1)])
    
    return X_train, X_test, y_train, y_test, le

# Quantum Neural Network Classifier - Optimized for speed
class QuantumNeuralNetworkClassifier:
    def __init__(self, n_qubits=4, n_layers=2):  # Reduced layers for speed
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Reduced parameters for faster optimization
        self.n_params = n_qubits * 2 * n_layers + 2 * n_layers
        self.params_ = None
        
    def _encode_data(self, X):
        # Vectorized encoding for batch processing
        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_qubits))
        
        # Map features to qubits with broadcasting
        for i in range(min(X.shape[1], self.n_qubits)):
            encoded[:, i] = np.clip(X[:, i % X.shape[1]], -np.pi, np.pi)
        
        return encoded
    
    def _quantum_circuit_batch(self, params, X_encoded):
        # Process entire batch at once
        n_samples = X_encoded.shape[0]
        state = np.copy(X_encoded)
        
        # Pre-compute layer indices for faster access
        layer_params = []
        for l in range(self.n_layers):
            start_idx = l * (2 * self.n_qubits + 2)
            end_idx = (l + 1) * 2 * self.n_qubits + l * 2
            layer_params.append(params[start_idx:end_idx])
        
        # Apply circuit operations with vectorized operations
        for l in range(self.n_layers):
            rotation_params = layer_params[l][:2 * self.n_qubits]
            
            # Vectorized rotation operations
            for q in range(self.n_qubits):
                if q * 2 + 1 < len(rotation_params):
                    # Apply 2 rotation gates for each qubit (reduced from 3)
                    state[:, q] = np.sin(rotation_params[q * 2] + state[:, q])
                    state[:, q] = np.cos(rotation_params[q * 2 + 1] + state[:, q])
            
            # Vectorized entanglement operations
            if l < self.n_layers - 1:  # Skip in last layer
                entangle_params = layer_params[l][2 * self.n_qubits:]
                if len(entangle_params) >= 2:
                    for i in range(self.n_qubits - 1):
                        # Simplified entanglement effect
                        avg = (state[:, i] + state[:, i + 1]) / 2
                        state[:, i] = avg * np.sin(entangle_params[0])
                        state[:, i+1] = avg * np.cos(entangle_params[1])
        
        # Simplified measurement strategy - vectorized across batch
        prob = np.mean(np.abs(state), axis=1)
        return np.clip(prob, 0.001, 0.999)
    
    def _predict_proba(self, params, X):
        X_encoded = self._encode_data(X)
        probs = self._quantum_circuit_batch(params, X_encoded)
        return np.column_stack([1 - probs, probs])
    
    def _cost(self, params, X, y):
        # Simplified loss calculation
        y_proba = self._predict_proba(params, X)
        y_proba_true = y_proba[np.arange(len(y)), y]
        y_proba_true = np.clip(y_proba_true, 1e-10, 1 - 1e-10)
        
        # Basic cross-entropy with minimal regularization
        loss = -np.mean(np.log(y_proba_true)) + 0.005 * np.sum(params**2)
        return loss
    
    def fit(self, X, y):
        # Simplified optimization with fewer iterations
        initial_params = np.random.uniform(-0.1, 0.1, self.n_params)
        
        self.params_ = minimize(
            lambda params: self._cost(params, X, y),
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': 100}  # Reduced iterations
        ).x
        
        return self
    
    def predict_proba(self, X):
        if self.params_ is None:
            raise ValueError("Model has not been fitted yet")
        return self._predict_proba(self.params_, X)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# Quantum SVM Classifier - Optimized implementation
class QuantumSVMClassifier:
    def __init__(self, C=1.0, gamma=1.0):
        self.C = C
        self.gamma = gamma
        self.X_train_ = None
        self.y_train_ = None
        self.alpha_ = None
        
    @lru_cache(maxsize=128)  # Cache kernel computations
    def _quantum_kernel(self, x1_tuple, x2_tuple):
        # Convert tuples back to arrays for computation
        x1 = np.array(x1_tuple)
        x2 = np.array(x2_tuple)
        
        # Simplified kernel computation
        gamma = self.gamma
        kernel_val = np.exp(-gamma * np.sum((x1 - x2) ** 2))
        return kernel_val
    
    def _compute_kernel_matrix(self, X1, X2):
        n1, n2 = X1.shape[0], X2.shape[0]
        
        # Use numpy's broadcasting for faster computation
        diff_squared = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
        K = np.exp(-self.gamma * diff_squared)
        
        return K
    
    def fit(self, X, y):
        # Store training data - use float32 for memory efficiency
        self.X_train_ = X.astype(np.float32)
        self.y_train_ = y
        
        # Compute kernel matrix for training data
        self.K_train_ = self._compute_kernel_matrix(X, X)
        
        # Simplified SVM weights
        n_samples = X.shape[0]
        class_counts = np.bincount(y)
        class_weights = np.zeros_like(class_counts, dtype=float) 
        
        for c in range(len(class_counts)):
            if class_counts[c] > 0:
                class_weights[c] = 1.0 / class_counts[c]
        
        # Normalize weights
        class_weights = class_weights / np.sum(class_weights) * len(class_weights)
        
        # Assign alpha values based on class weights
        self.alpha_ = np.zeros(n_samples)
        for i in range(n_samples):
            self.alpha_[i] = class_weights[y[i]] * self.C / n_samples
            
        return self
    
    def predict(self, X):
        if self.X_train_ is None or self.y_train_ is None or self.alpha_ is None:
            raise ValueError("Model has not been fitted yet")
        
        # Compute kernel between test and training data
        K_test = self._compute_kernel_matrix(X, self.X_train_)
        
        # Vectorized decision function
        weighted_K = K_test * self.alpha_ * (2 * self.y_train_ - 1)
        f = np.sum(weighted_K, axis=1)
        
        # Return predicted class
        return (f > 0).astype(int)

# Quantum Variational Classifier - Optimized
class QuantumVariationalClassifier:
    def __init__(self, n_qubits=4, n_layers=2):  # Reduced layers
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_qubits * 2 * n_layers  # Fewer parameters
        self.params_ = None
        
    def _encode_data(self, X):
        # Vectorized encoding
        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_qubits))
        
        for i in range(min(X.shape[1], self.n_qubits)):
            encoded[:, i] = np.tanh(X[:, i % X.shape[1]]) * np.pi
            
        return encoded
    
    def _variational_circuit_batch(self, params, X_encoded):
        # Process entire batch at once
        n_samples = X_encoded.shape[0]
        state = np.copy(X_encoded)
        
        # Apply circuit operations
        for l in range(self.n_layers):
            # Apply rotation gates
            for q in range(self.n_qubits):
                idx = l * (self.n_qubits * 2) + q * 2
                if idx + 1 < len(params):
                    # Vectorized operations
                    state[:, q] = np.tanh(np.sin(params[idx] + state[:, q]))
                    state[:, q] = np.tanh(np.cos(params[idx + 1] + state[:, q]))
        
        # Simplified measurement
        prob = np.mean(state, axis=1) * 0.5 + 0.5
        return np.clip(prob, 0.001, 0.999)
    
    def _predict_proba(self, params, X):
        X_encoded = self._encode_data(X)
        probs = self._variational_circuit_batch(params, X_encoded)
        return np.column_stack([1 - probs, probs])
    
    def _cost(self, params, X, y):
        # Simplified cost function
        y_proba = self._predict_proba(params, X)
        y_onehot = np.eye(2)[y]
        
        # Mean squared error with minimal regularization
        loss = np.mean((y_proba - y_onehot) ** 2)
        reg_term = 0.005 * np.sum(params**2)
        
        return loss + reg_term
    
    def fit(self, X, y):
        # Single optimization run instead of multiple starts
        initial_params = np.random.uniform(-0.1, 0.1, self.n_params)
        
        self.params_ = minimize(
            lambda params: self._cost(params, X, y),
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': 100}  # Reduced iterations
        ).x
        
        return self
    
    def predict_proba(self, X):
        if self.params_ is None:
            raise ValueError("Model has not been fitted yet")
        return self._predict_proba(self.params_, X)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# Quantum K-Nearest Neighbors Classifier - Optimized
class QuantumKNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train_ = None
        self.y_train_ = None
        
    def _quantum_distance_matrix(self, X1, X2):
        # Vectorized distance computation using broadcasting
        # Calculate pairwise Euclidean distances
        diff_squared = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
        euclidean_dist = np.sqrt(diff_squared)
        
        # Calculate dot products for quantum term
        dot_products = np.dot(X1, X2.T)
        quantum_term = 0.2 * np.abs(np.sin(dot_products))
        
        # Combined distance metric
        return euclidean_dist - quantum_term
    
    def fit(self, X, y):
        # Store training data - use float32 for memory efficiency
        self.X_train_ = X.astype(np.float32)
        self.y_train_ = y
        return self
    
    def predict(self, X):
        if self.X_train_ is None or self.y_train_ is None:
            raise ValueError("Model has not been fitted yet")
            
        # Compute all distances at once
        distances = self._quantum_distance_matrix(X, self.X_train_)
        
        # Find k nearest neighbors for all test points at once
        nearest_indices = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        
        # Get weights for all neighbors
        weights = 1.0 / (np.take_along_axis(distances, nearest_indices, axis=1) + 1e-5)
        
        # Get labels of neighbors
        nearest_labels = self.y_train_[nearest_indices]
        
        # For each test point, count weighted votes for each class
        y_pred = np.zeros(X.shape[0], dtype=int)
        
        for i in range(X.shape[0]):
            # Count weighted votes by class
            votes = np.zeros(2)  # Binary classification
            for j in range(self.n_neighbors):
                votes[nearest_labels[i, j]] += weights[i, j]
            
            # Predict class with highest weighted votes
            y_pred[i] = np.argmax(votes)
            
        return y_pred

# Quantum Clustering - Optimized
class QuantumClusteringClassifier:
    def __init__(self, n_clusters=2, max_iter=100):  # Reduced iterations
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids_ = None
        self.cluster_labels_ = None
        
    def _quantum_distance_matrix(self, X1, X2):
        # Vectorized distance computation
        # Calculate pairwise Euclidean distances
        diff_squared = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
        euclidean_dist = np.sqrt(diff_squared)
        
        # Calculate dot products for quantum term
        dot_products = np.dot(X1, X2.T)
        phase_term = 0.2 * np.sin(dot_products)
        
        # Combined distance metric
        return euclidean_dist - phase_term
    
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("This implementation requires labels for training")
            
        # Simplified initialization
        # Choose centroids using k-means++ like initialization
        n_samples = X.shape[0]
        
        # First centroid random
        indices = [np.random.randint(0, n_samples)]
        centroids = X[indices].copy()
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Compute distances to existing centroids
            min_distances = np.min([np.sum((X - centroids[i])**2, axis=1) for i in range(len(centroids))], axis=0)
            
            # Choose next centroid with probability proportional to distance
            if np.sum(min_distances) > 0:
                probabilities = min_distances / np.sum(min_distances)
                next_idx = np.random.choice(np.arange(n_samples), p=probabilities)
            else:
                next_idx = np.random.randint(0, n_samples)
                
            centroids = np.vstack([centroids, X[next_idx]])
        
        self.centroids_ = centroids
        
        # Store original labels
        self.original_labels_ = y
        
        # Perform simplified clustering
        for _ in range(self.max_iter):
            # Calculate distances from each point to each centroid
            distances = self._quantum_distance_matrix(X, self.centroids_)
            
            # Assign points to clusters
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(self.centroids_)
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    # If empty cluster, find furthest point
                    new_centroids[k] = X[np.argmax(np.min(distances, axis=1))]
            
            # Check convergence
            if np.allclose(new_centroids, self.centroids_, rtol=1e-3, atol=1e-3):
                break
                
            self.centroids_ = new_centroids
        
        # Assign labels to clusters
        cluster_labels = np.zeros(self.n_clusters, dtype=int)
        for k in range(self.n_clusters):
            cluster_points = np.where(labels == k)[0]
            if len(cluster_points) > 0:
                # Assign most common class
                cluster_labels[k] = np.argmax(np.bincount(y[cluster_points]))
            else:
                # Default to most common class
                cluster_labels[k] = np.argmax(np.bincount(y))
        
        self.cluster_labels_ = cluster_labels
        
        return self
    
    def predict(self, X):
        if self.centroids_ is None or self.cluster_labels_ is None:
            raise ValueError("Model has not been fitted yet")
            
        # Calculate distances from each point to each centroid
        distances = self._quantum_distance_matrix(X, self.centroids_)
        
        # Assign points to clusters
        clusters = np.argmin(distances, axis=1)
        
        # Map cluster to class label
        return self.cluster_labels_[clusters]

# Optimized model functions
def qnn_classifier(X_train, X_test, y_train, y_test):
    # Quantum Neural Network Classifier
    clf = QuantumNeuralNetworkClassifier(n_qubits=4, n_layers=2)  # Reduced layers
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    score = accuracy_score(y_test, y_pred) * 100
    
    # Classification report
    report = classification_report(y_test, y_pred)
    
    return score, report, clf

def qsvm_classifier(X_train, X_test, y_train, y_test):
    # Quantum SVM Classifier
    clf = QuantumSVMClassifier(C=1.0, gamma=0.8)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    score = accuracy_score(y_test, y_pred) * 100
    
    # Classification report
    report = classification_report(y_test, y_pred)
    
    return score, report, clf

def qvqc_classifier(X_train, X_test, y_train, y_test):
    # Quantum Variational Quantum Classifier
    clf = QuantumVariationalClassifier(n_qubits=4, n_layers=2)  # Reduced layers
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    score = accuracy_score(y_test, y_pred) * 100
    
    # Classification report
    report = classification_report(y_test, y_pred)
    
    return score, report, clf

def qknn_classifier(X_train, X_test, y_train, y_test):
    # Quantum KNN Classifier
    clf = QuantumKNNClassifier(n_neighbors=5)  # Reduced neighbors
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    score = accuracy_score(y_test, y_pred) * 100
    
    # Classification report
    report = classification_report(y_test, y_pred)
    
    return score, report, clf

def qqcl_classifier(X_train, X_test, y_train, y_test):
    # Quantum Clustering Classifier
    clf = QuantumClusteringClassifier(n_clusters=2, max_iter=100)  # Reduced parameters
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    score = accuracy_score(y_test, y_pred) * 100
    
    # Classification report
    report = classification_report(y_test, y_pred)
    
    return score, report, clf

# Use parallel processing for model comparison
def compare_qml_models(X_train, X_test, y_train, y_test):
    try:
        # Try to use parallel processing if joblib is available
        from joblib import Parallel, delayed
        
        # Run each model in parallel
        results = Parallel(n_jobs=-1)(
            delayed(func)(X_train, X_test, y_train, y_test) 
            for func in [qnn_classifier, qsvm_classifier, qvqc_classifier, qknn_classifier, qqcl_classifier]
        )
        
        # Extract scores
        model_names = ["Quantum Neural Network (QNN)", "Quantum SVM (QSVM)", 
                       "Quantum Variational Classifier (QVQC)", "Quantum KNN (QKNN)",
                       "Quantum Clustering (QQCL)"]
        scores = [result[0] for result in results]
        
        # Create a dictionary for consistency with the non-parallel approach
        results_dict = {model: score for model, score in zip(model_names, scores)}
        
    except ImportError:
        # Fall back to sequential processing if joblib not available
        results_dict = {}
        
        # Run each model and store results
        qnn_score, _, _ = qnn_classifier(X_train, X_test, y_train, y_test)
        results_dict["Quantum Neural Network (QNN)"] = qnn_score
        
        qsvm_score, _, _ = qsvm_classifier(X_train, X_test, y_train, y_test)
        results_dict["Quantum SVM (QSVM)"] = qsvm_score
        
        qvqc_score, _, _ = qvqc_classifier(X_train, X_test, y_train, y_test)
        results_dict["Quantum Variational Classifier (QVQC)"] = qvqc_score
        
        qknn_score, _, _ = qknn_classifier(X_train, X_test, y_train, y_test)
        results_dict["Quantum KNN (QKNN)"] = qknn_score
        
        qqcl_score, _, _ = qqcl_classifier(X_train, X_test, y_train, y_test)
        results_dict["Quantum Clustering (QQCL)"] = qqcl_score
    
    # Create DataFrame
    df_results = pd.DataFrame({
        'Model': list(results_dict.keys()),
        'Accuracy (%)': list(results_dict.values())
    })
    
    # Sort by accuracy
    df_results = df_results.sort_values('Accuracy (%)', ascending=False).reset_index(drop=True)
    
    return df_results

# Function to run a single model for testing
def run_single_model(model_name, X_train, X_test, y_train, y_test):
    model_functions = {
        'qnn': qnn_classifier,
        'qsvm': qsvm_classifier,
        'qvqc': qvqc_classifier,
        'qknn': qknn_classifier,
        'qqcl': qqcl_classifier
    }
    
    if model_name.lower() in model_functions:
        score, report, _ = model_functions[model_name.lower()](X_train, X_test, y_train, y_test)
        print(f"Model: {model_name}")
        print(f"Accuracy: {score:.2f}%")
        print("Classification Report:")
        print(report)
    else:
        print(f"Model {model_name} not found. Available models: {', '.join(model_functions.keys())}")

# Add a main block to demonstrate usage
if __name__ == "__main__":
    # Load and preprocess data
    data = loadData()
    X_train, X_test, y_train, y_test, le = preprocessing(data)
    
    # Compare all models
    results = compare_qml_models(X_train, X_test, y_train, y_test)
    print("Model Comparison Results:")
    print(results)
    
    # Optional: Run a single model
    # run_single_model('qnn', X_train, X_test, y_train, y_test)