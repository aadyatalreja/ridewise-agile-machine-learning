import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.optimize import minimize

# For a real quantum implementation, you would use:
# from qiskit import QuantumCircuit, Aer, execute
# from qiskit.circuit import Parameter
# from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
# from qiskit_machine_learning.neural_networks import SamplerQNN
# from qiskit_machine_learning.algorithms import NeuralNetworkClassifier

# Helper function to load data
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

# Preprocessing function
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
    
    # Add engineered features - distance between stations
    X_train = np.hstack([X_train, np.abs(X_train[:, 1] - X_train[:, 2]).reshape(-1, 1)])
    X_test = np.hstack([X_test, np.abs(X_test[:, 1] - X_test[:, 2]).reshape(-1, 1)])
    
    return X_train, X_test, y_train, y_test, le

# Quantum Neural Network Classifier - Using classical simulation for demonstration
class QuantumNeuralNetworkClassifier:
    def __init__(self, n_qubits=4, n_layers=3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Number of parameters: 3 rotation angles per qubit per layer + 2 entangling parameters per layer
        self.n_params = n_qubits * 3 * n_layers + 2 * n_layers
        
    def _encode_data(self, x):
        # Improved angle encoding - map normalized features to rotation angles
        # Spread features across all available qubits for better representation
        encoded = np.zeros(self.n_qubits)
        for i in range(min(len(x), self.n_qubits)):
            encoded[i] = np.clip(x[i % len(x)], -np.pi, np.pi)
        return encoded
    
    def _quantum_circuit(self, params, x_encoded):
        # This simulates what would happen in a parameterized quantum circuit
        # In a real QNN, this would apply rotation and entanglement gates
        
        # Reshape params for easier handling
        # Make sure params is 1D first
        params = np.array(params).flatten()
        
        # Initialize circuit state (simplified representation)
        state = np.copy(x_encoded)
        
        for l in range(self.n_layers):
            # Apply rotation gates (simplified)
            rotation_params = params[l * (3 * self.n_qubits + 2) : (l + 1) * 3 * self.n_qubits + l * 2]
            for q in range(self.n_qubits):
                if q * 3 + 2 < len(rotation_params):
                    # Apply 3 rotation gates for each qubit
                    state[q % len(state)] = np.sin(rotation_params[q * 3] + state[q % len(state)])
                    state[q % len(state)] = np.cos(rotation_params[q * 3 + 1] + state[q % len(state)])
                    state[q % len(state)] = np.sin(rotation_params[q * 3 + 2] + state[q % len(state)])
            
            # Apply entangling gates (simplified)
            entangle_params = params[(l + 1) * 3 * self.n_qubits + l * 2 : (l + 1) * (3 * self.n_qubits + 2)]
            if len(entangle_params) >= 2:
                for i in range(len(state) - 1):
                    # Simulate entanglement effect with non-linear interaction
                    avg = (state[i] + state[i + 1]) / 2
                    state[i] = avg * np.sin(entangle_params[0]) + 0.1 * np.cos(state[i] * state[i+1])
                    state[i+1] = avg * np.cos(entangle_params[1]) + 0.1 * np.sin(state[i] * state[i+1])
        
        # Measure the final state (improved measurement strategy)
        weighted_sum = np.sum(np.abs(state) * np.linspace(0.5, 1.0, len(state)))
        prob = weighted_sum / (len(state) * 0.75)  # Normalize
        prob = np.clip(prob, 0.001, 0.999)  # Ensure output is between 0 and 1
        return prob
    
    def _predict_proba_single(self, params, x):
        x_encoded = self._encode_data(x)
        prob = self._quantum_circuit(params, x_encoded)
        return np.array([1 - prob, prob])  # Return probabilities for both classes
    
    def _predict_proba(self, params, X):
        return np.array([self._predict_proba_single(params, x) for x in X])
    
    def _cost(self, params, X, y):
        # Binary cross-entropy loss with regularization
        y_proba = self._predict_proba(params, X)
        # Get probability of true class for each sample
        y_proba_true = y_proba[np.arange(len(y)), y]
        # Avoid numerical issues with log(0)
        y_proba_true = np.clip(y_proba_true, 1e-10, 1 - 1e-10)
        # Compute cross-entropy loss with L2 regularization
        loss = -np.mean(np.log(y_proba_true)) + 0.01 * np.sum(params**2)
        return loss
    
    def fit(self, X, y):
        # Initialize random parameters with better initialization
        initial_params = np.random.uniform(-0.1, 0.1, self.n_params)
        
        # Use scipy optimizer with improved settings
        self.params_ = minimize(
            lambda params: self._cost(params, X, y),
            initial_params,
            method='BFGS',  # Use a method that uses gradients for better convergence
            options={'maxiter': 500}  # Increase iterations for better convergence
        ).x
        
        return self
    
    def predict_proba(self, X):
        return self._predict_proba(self.params_, X)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# Quantum SVM Classifier - Using classical simulation for demonstration
class QuantumSVMClassifier:
    def __init__(self, C=1.0, gamma=1.0):
        self.C = C
        self.gamma = gamma
        
    def _quantum_kernel(self, x1, x2):
        # Improved quantum kernel computation
        # In a real QSVM, this would use a quantum circuit to compute kernel values
        gamma = self.gamma
        # RBF kernel with quantum-inspired nonlinearity
        kernel_val = np.exp(-gamma * np.sum((x1 - x2) ** 2))
        # Add quantum-inspired interference term
        interference = np.cos(np.sum(x1 * x2) * gamma)
        return 0.7 * kernel_val + 0.3 * interference
    
    def _compute_kernel_matrix(self, X1, X2):
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._quantum_kernel(X1[i], X2[j])
        return K
    
    def fit(self, X, y):
        # Store training data
        self.X_train_ = X
        self.y_train_ = y
        
        # Compute kernel matrix for training data
        self.K_train_ = self._compute_kernel_matrix(X, X)
        
        # Improved SVM optimization
        n_samples = X.shape[0]
        self.alpha_ = np.zeros(n_samples)
        
        # More sophisticated heuristic for SVM weights
        for i in range(n_samples):
            # Compute margin-based weights
            margin = 0
            same_class = 0
            diff_class = 0
            
            for j in range(n_samples):
                kernel_val = self._quantum_kernel(X[i], X[j])
                if y[j] == y[i]:
                    same_class += kernel_val
                else:
                    diff_class += kernel_val
            
            # Higher alpha for points near the margin
            margin = same_class - diff_class
            # Sigmoid function to scale alpha values
            self.alpha_[i] = self.C / (1 + np.exp(5 * np.abs(margin)))
            
        # Normalize alpha values
        self.alpha_ = self.alpha_ / np.sum(self.alpha_) * n_samples * self.C * 0.1
            
        return self
    
    def predict(self, X):
        # Compute kernel between test and training data
        K_test = self._compute_kernel_matrix(X, self.X_train_)
        
        # Compute decision function with bias term
        f = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            for j in range(self.X_train_.shape[0]):
                f[i] += self.alpha_[j] * (2 * self.y_train_[j] - 1) * K_test[i, j]
        
        # Add bias term (calculated from support vectors)
        bias = 0
        sv_count = 0
        for i in range(self.X_train_.shape[0]):
            if self.alpha_[i] > 1e-5:
                decision = 0
                for j in range(self.X_train_.shape[0]):
                    decision += self.alpha_[j] * (2 * self.y_train_[j] - 1) * self.K_train_[i, j]
                bias += (2 * self.y_train_[i] - 1) - decision
                sv_count += 1
        
        if sv_count > 0:
            bias /= sv_count
            f += bias
        
        # Return predicted class
        return (f > 0).astype(int)

# Quantum Variational Quantum Classifier
class QuantumVariationalClassifier:
    def __init__(self, n_qubits=4, n_layers=3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_qubits * 2 * n_layers
        
    def _encode_data(self, x):
        # Improved amplitude encoding
        encoded = np.zeros(self.n_qubits)
        for i in range(min(len(x), self.n_qubits)):
            # Non-linear transformation for better separation
            encoded[i] = np.tanh(x[i % len(x)]) * np.pi
        return encoded
    
    def _variational_circuit(self, params, x_encoded):
        # This simulates what would happen in a variational quantum circuit
        # In a real VQC, this would apply parameterized gates
        
        # Reshape params to make them easier to work with
        params = np.array(params).flatten()
        
        # Initialize circuit state
        state = np.copy(x_encoded)
        
        for l in range(self.n_layers):
            # Apply rotation gates with improved parametrization
            for q in range(self.n_qubits):
                idx = l * (self.n_qubits * 2) + q * 2
                if idx + 1 < len(params):
                    # Apply rotation with non-linear activation
                    state[q % len(state)] = np.tanh(np.sin(params[idx] + state[q % len(state)]))
                    state[q % len(state)] = np.tanh(np.cos(params[idx + 1] + state[q % len(state)]))
            
            # Add entanglement-like effects between qubits
            if l < self.n_layers - 1:  # Skip in last layer for better generalization
                temp_state = np.copy(state)
                for q in range(self.n_qubits - 1):
                    # Create interference between adjacent qubits
                    interference = 0.2 * np.sin(temp_state[q] * temp_state[q+1])
                    state[q] += interference
                    state[q+1] += interference
        
        # Improved measurement strategy
        weighted_state = state * np.linspace(0.5, 1.5, len(state))
        prob = np.tanh(np.mean(weighted_state)) * 0.5 + 0.5
        return np.clip(prob, 0.001, 0.999)
    
    def _predict_proba_single(self, params, x):
        x_encoded = self._encode_data(x)
        prob = self._variational_circuit(params, x_encoded)
        return np.array([1 - prob, prob])
    
    def _predict_proba(self, params, X):
        return np.array([self._predict_proba_single(params, x) for x in X])
    
    def _cost(self, params, X, y):
        # Improved loss function with class weighting
        y_proba = self._predict_proba(params, X)
        y_onehot = np.eye(2)[y]
        
        # Class weights to handle class imbalance
        class_counts = np.bincount(y)
        class_weights = np.zeros_like(class_counts, dtype=float)
        for c in range(len(class_counts)):
            if class_counts[c] > 0:
                class_weights[c] = 1.0 / class_counts[c]
        class_weights = class_weights / np.sum(class_weights) * len(class_weights)
        
        # Weighted loss
        sample_weights = np.array([class_weights[c] for c in y])
        weighted_loss = np.sum((y_proba - y_onehot) ** 2, axis=1) * sample_weights
        
        # Add regularization
        reg_term = 0.01 * np.sum(params**2)
        
        return np.mean(weighted_loss) + reg_term
    
    def fit(self, X, y):
        # Better initialization strategy
        initial_params = np.random.uniform(-0.5, 0.5, self.n_params)
        
        # Multi-start optimization for better chance of finding global optimum
        best_loss = float('inf')
        best_params = initial_params
        
        for _ in range(3):  # Try 3 different starting points
            params = np.random.uniform(-0.5, 0.5, self.n_params)
            result = minimize(
                lambda params: self._cost(params, X, y),
                params,
                method='L-BFGS-B',
                options={'maxiter': 300}
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
        
        self.params_ = best_params
        return self
    
    def predict_proba(self, X):
        return self._predict_proba(self.params_, X)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# Quantum K-Nearest Neighbors Classifier
class QuantumKNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        
    def _quantum_distance(self, x1, x2):
        # Improved quantum distance computation
        # Combine Euclidean distance with quantum-inspired interference term
        euclidean_dist = np.sqrt(np.sum((x1 - x2) ** 2))
        # Add quantum-inspired phase term
        quantum_term = np.abs(np.sin(np.dot(x1, x2)))
        # Combined distance metric
        return 0.8 * euclidean_dist - 0.2 * quantum_term
    
    def fit(self, X, y):
        # Store training data
        self.X_train_ = X
        self.y_train_ = y
        return self
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype=int)
        
        for i in range(X.shape[0]):
            # Compute distances to all training points
            distances = np.array([self._quantum_distance(X[i], x_train) for x_train in self.X_train_])
            
            # Find k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_train_[nearest_indices]
            
            # Distance-weighted voting
            weights = 1.0 / (distances[nearest_indices] + 1e-5)
            votes = np.zeros(2)  # Assuming binary classification
            
            for j, label in enumerate(nearest_labels):
                votes[label] += weights[j]
            
            # Weighted majority vote
            y_pred[i] = np.argmax(votes)
            
        return y_pred

# Quantum Quantum Clustering
class QuantumClusteringClassifier:
    def __init__(self, n_clusters=2, max_iter=200):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
    def _quantum_distance(self, x1, x2):
        # Improved quantum distance computation
        euclidean_dist = np.sqrt(np.sum((x1 - x2) ** 2))
        # Add quantum-inspired phase term
        phase_term = 0.2 * np.sin(np.sum(x1 * x2))
        return euclidean_dist - phase_term
    
    def fit(self, X, y=None):
        # Improved initialization - use kmeans++ like strategy
        # Choose first centroid randomly
        indices = [np.random.randint(0, X.shape[0])]
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Compute distances to existing centroids
            distances = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                min_dist = float('inf')
                for idx in indices:
                    dist = self._quantum_distance(X[i], X[idx])
                    min_dist = min(min_dist, dist)
                distances[i] = min_dist
            
            # Choose next centroid with probability proportional to distance
            distances[distances < 0] = 0  # Ensure no negative probabilities
            if np.sum(distances) > 0:
                probabilities = distances / np.sum(distances)
                next_idx = np.random.choice(np.arange(X.shape[0]), p=probabilities)
                indices.append(next_idx)
            else:
                # Fallback to random selection if all distances are zero
                next_idx = np.random.randint(0, X.shape[0])
                while next_idx in indices:
                    next_idx = np.random.randint(0, X.shape[0])
                indices.append(next_idx)
        
        self.centroids_ = X[indices]
        
        # Store original labels for each point
        self.original_labels_ = y
        
        # Perform improved clustering
        for iter_num in range(self.max_iter):
            # Assign points to clusters
            labels = np.zeros(X.shape[0], dtype=int)
            for i in range(X.shape[0]):
                distances = np.array([self._quantum_distance(X[i], centroid) 
                                     for centroid in self.centroids_])
                labels[i] = np.argmin(distances)
            
            # Update centroids with soft assignments
            new_centroids = np.zeros_like(self.centroids_)
            cluster_sizes = np.zeros(self.n_clusters)
            
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    # Regular centroid update
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                    cluster_sizes[k] = len(cluster_points)
                else:
                    # If empty cluster, find the point furthest from its assigned centroid
                    distances_to_centroid = np.zeros(X.shape[0])
                    for i in range(X.shape[0]):
                        assigned_cluster = labels[i]
                        distances_to_centroid[i] = self._quantum_distance(X[i], self.centroids_[assigned_cluster])
                    
                    # Find the point furthest from its centroid
                    furthest_point_idx = np.argmax(distances_to_centroid)
                    new_centroids[k] = X[furthest_point_idx]
                    cluster_sizes[k] = 1
            
            # Early stopping condition - convergence check with tolerance
            centroid_shift = np.sum((new_centroids - self.centroids_) ** 2)
            self.centroids_ = new_centroids
            
            if centroid_shift < 1e-4:
                break
        
        # Assign labels to clusters - weighted by class distribution
        cluster_labels = np.zeros(self.n_clusters, dtype=int)
        for k in range(self.n_clusters):
            cluster_points = np.where(labels == k)[0]
            if len(cluster_points) > 0:
                # Count class occurrences
                class_counts = np.bincount(y[cluster_points])
                # Weighted by inverse of class frequency
                class_freq = np.bincount(y) / len(y)
                class_weights = 1.0 / (class_freq + 1e-5)
                weighted_counts = class_counts * class_weights[:len(class_counts)]
                cluster_labels[k] = np.argmax(weighted_counts)
            else:
                # Default to most common class
                cluster_labels[k] = np.argmax(np.bincount(y))
        
        self.cluster_labels_ = cluster_labels
        
        return self
    
    def predict(self, X):
        # Assign points to clusters
        labels = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            distances = np.array([self._quantum_distance(X[i], centroid) 
                                for centroid in self.centroids_])
            cluster = np.argmin(distances)
            labels[i] = self.cluster_labels_[cluster]
            
        return labels

# Implement actual model functions
def qnn_classifier(X_train, X_test, y_train, y_test):
    # Quantum Neural Network Classifier
    clf = QuantumNeuralNetworkClassifier(n_qubits=4, n_layers=3)
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
    clf = QuantumSVMClassifier(C=1.5, gamma=0.8)
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
    clf = QuantumVariationalClassifier(n_qubits=4, n_layers=3)
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
    clf = QuantumKNNClassifier(n_neighbors=7)
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
    clf = QuantumClusteringClassifier(n_clusters=3, max_iter=200)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    score = accuracy_score(y_test, y_pred) * 100
    
    # Classification report
    report = classification_report(y_test, y_pred)
    
    return score, report, clf

def compare_qml_models(X_train, X_test, y_train, y_test):
    # Dictionary to store results
    results = {}
    
    # Run each model and store results
    qnn_score, _, _ = qnn_classifier(X_train, X_test, y_train, y_test)
    results["Quantum Neural Network (QNN)"] = qnn_score
    
    qsvm_score, _, _ = qsvm_classifier(X_train, X_test, y_train, y_test)
    results["Quantum SVM (QSVM)"] = qsvm_score
    
    qvqc_score, _, _ = qvqc_classifier(X_train, X_test, y_train, y_test)
    results["Quantum Variational Classifier (QVQC)"] = qvqc_score
    
    qknn_score, _, _ = qknn_classifier(X_train, X_test, y_train, y_test)
    results["Quantum KNN (QKNN)"] = qknn_score
    
    qqcl_score, _, _ = qqcl_classifier(X_train, X_test, y_train, y_test)
    results["Quantum Clustering (QQCL)"] = qqcl_score
    
    # Create DataFrame
    df_results = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy (%)': list(results.values())
    })
    
    # Sort by accuracy
    df_results = df_results.sort_values('Accuracy (%)', ascending=False).reset_index(drop=True)
    
    return df_results