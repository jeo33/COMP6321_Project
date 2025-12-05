
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from scipy.sparse import issparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ============================================================================
# LINEAR MODELS (Keep for comparison)
# ============================================================================

class MultiLabelClassifier:
    """
    Multi-label classifier that works with sparse matrices
    """

    def __init__(self,
                 model_type='logistic',
                 max_iter=100,
                 n_jobs=-1,
                 random_state=42,
                 verbose=True):
        """
        Initialize multi-label classifier

        Args:
            model_type: 'logistic', 'sgd', or 'rf'
            max_iter: Maximum iterations
            n_jobs: Number of parallel jobs
            random_state: Random seed
            verbose: Print progress
        """
        self.model_type = model_type
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Create base model
        if model_type == 'logistic':
            base_model = LogisticRegression(
                max_iter=max_iter,
                random_state=random_state,
                verbose=1 if verbose else 0,
                n_jobs=1
            )
        elif model_type == 'sgd':
            base_model = SGDClassifier(
                loss='log_loss',
                max_iter=max_iter,
                random_state=random_state,
                verbose=1 if verbose else 0,
                n_jobs=1
            )
        elif model_type == 'rf':
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=random_state,
                verbose=1 if verbose else 0,
                n_jobs=1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model = MultiOutputClassifier(base_model, n_jobs=n_jobs)
        self.is_fitted = False

    def fit(self, X, y):
        """Train the model"""
        print("\n" + "="*70)
        print("TRAINING LINEAR MODEL")
        print("="*70)
        print(f"Model type: {self.model_type}")
        print(f"Max iterations: {self.max_iter}")

        if issparse(y):
            y = y.toarray()

        print("\nTraining in progress...")
        self.model.fit(X, y)
        self.is_fitted = True
        print(f"\n✓ Training completed!")
        return self

    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def save(self, filepath='models/linear_model.pkl'):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Model saved to {filepath}")

    @staticmethod
    def load(filepath='models/linear_model.pkl'):
        """Load model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {filepath}")
        return model


# ============================================================================
# DEEP NEURAL NETWORK (DNN)
# ============================================================================

class MultiLabelDNN(nn.Module):

    def __init__(self, input_dim, hidden_layers=[512, 256, 128], n_labels=103,
                 activation='relu', dropout=0.3):
        """
        Args:
            input_dim: Number of input features
            hidden_layers: List of neurons per hidden layer [512, 256, 128]
            n_labels: Number of output labels
            activation: 'relu', 'tanh', 'sigmoid', or 'leaky_relu'
            dropout: Dropout rate for regularization
        """
        super(MultiLabelDNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.n_labels = n_labels
        self.activation_name = activation
        self.dropout_rate = dropout

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_layers):

            layers.append(nn.Linear(prev_dim, hidden_dim))

            layers.append(nn.BatchNorm1d(hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            else:
                layers.append(nn.ReLU())  # Default to ReLU

            # Dropout for regularization
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer (no activation, will use BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, n_labels))

        self.network = nn.Sequential(*layers)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*70}")
        print("DNN ARCHITECTURE")
        print(f"{'='*70}")
        print(f"Input Layer:     {input_dim} features")
        for i, h in enumerate(hidden_layers):
            print(f"Hidden Layer {i+1}:  {h} neurons (activation={activation}, dropout={dropout})")
        print(f"Output Layer:    {n_labels} labels (sigmoid)")
        print(f"Total Parameters: {total_params:,}")
        print(f"{'='*70}\n")

    def forward(self, x):
        return self.network(x)


class DNNClassifier:

    def __init__(self, hidden_layers=[512, 256, 128], activation='relu',
                 dropout=0.3, learning_rate=0.001, batch_size=256,
                 epochs=50, device='auto', random_state=42):
        """
        Args:
            hidden_layers: List of neurons per hidden layer [512, 256, 128]
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
            dropout: Dropout rate for regularization (0.0 to 1.0)
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: 'cuda', 'cpu', or 'auto'
            random_state: Random seed
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = None
        self.is_fitted = False
        self.history = None

        print(f"✓ Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Args:
            X: Training features (sparse or dense)
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print("\n" + "="*70)
        print("TRAINING DEEP NEURAL NETWORK")
        print("="*70)

        # Convert sparse to dense
        if issparse(X):
            print("Converting sparse features to dense (this may take time)...")
            X = X.toarray()
        if issparse(y):
            y = y.toarray()

        # Ensure numpy arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        print(f"Training data: {X.shape[0]:,} samples, {X.shape[1]:,} features")
        print(f"Labels: {y.shape[1]} categories")

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        input_dim = X.shape[1]
        n_labels = y.shape[1]

        self.model = MultiLabelDNN(
            input_dim=input_dim,
            hidden_layers=self.hidden_layers,
            n_labels=n_labels,
            activation=self.activation,
            dropout=self.dropout
        ).to(self.device)

        # Calculate positive weights to handle class imbalance
        # weight = number_of_negatives / number_of_positives
        n_samples = y.shape[0]
        n_positives = np.sum(y, axis=0)
        n_negatives = n_samples - n_positives
        # Clamp weights to avoid extreme values for very rare classes
        # Use sqrt to dampen the weights and reduce false positives
        pos_weights = torch.FloatTensor(np.sqrt(n_negatives / (n_positives + 1e-6))).to(self.device)
        
        print(f"Using weighted loss to handle class imbalance (Avg weight: {pos_weights.mean().item():.2f})")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        # Add weight decay for regularization
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        val_loader = None
        if X_val is not None and y_val is not None:
            if issparse(X_val):
                X_val = X_val.toarray()
            if issparse(y_val):
                y_val = y_val.toarray()
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"\nTraining for {self.epochs} epochs...")
        print(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}")
        print("-"*70)

        self.history = {'train_loss': [], 'val_loss': [], 'epoch': []}

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in dataloader:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(dataloader)
            self.history['train_loss'].append(avg_train_loss)
            self.history['epoch'].append(epoch + 1)

            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                self.history['val_loss'].append(avg_val_loss)

                print(f"Epoch [{epoch+1:3d}/{self.epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Early Stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                        break
            else:
                print(f"Epoch [{epoch+1:3d}/{self.epochs}] - Train Loss: {avg_train_loss:.4f}")
        
        # Load best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model with Val Loss: {best_val_loss:.4f}")

        self.is_fitted = True
        print("-"*70)
        print("✓ Training completed!")
        return self.history

    def predict(self, X, threshold=0.5):
        """
        Args:
            X: Features
            threshold: Classification threshold (default 0.5)

        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()

        if issparse(X):
            X = X.toarray()

        X = np.asarray(X, dtype=np.float32)

        predictions = []
        batch_size = self.batch_size

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                outputs = self.model(batch_X)
                probs = torch.sigmoid(outputs)
                batch_pred = (probs > threshold).cpu().numpy()
                predictions.append(batch_pred)

        return np.vstack(predictions)

    def predict_proba(self, X):
        """Get probability predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()

        if issparse(X):
            X = X.toarray()

        X = np.asarray(X, dtype=np.float32)

        probabilities = []
        batch_size = self.batch_size

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                outputs = self.model(batch_X)
                probs = torch.sigmoid(outputs).cpu().numpy()
                probabilities.append(probs)

        return np.vstack(probabilities)

    def save(self, filepath='models/dnn_model.pth'):
        """Save model"""
        torch.save({
            'model_state': self.model.state_dict(),
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'dropout': self.dropout,
            'input_dim': self.model.input_dim,
            'n_labels': self.model.n_labels,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'history': self.history
        }, filepath)
        print(f"✓ DNN model saved to {filepath}")

    @staticmethod
    def load(filepath='models/dnn_model.pth', device='auto'):
        """Load model"""
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        checkpoint = torch.load(filepath, map_location=device)

        classifier = DNNClassifier(
            hidden_layers=checkpoint['hidden_layers'],
            activation=checkpoint['activation'],
            dropout=checkpoint['dropout'],
            learning_rate=checkpoint['learning_rate'],
            batch_size=checkpoint['batch_size'],
            device=device
        )

        classifier.model = MultiLabelDNN(
            input_dim=checkpoint['input_dim'],
            hidden_layers=checkpoint['hidden_layers'],
            n_labels=checkpoint['n_labels'],
            activation=checkpoint['activation'],
            dropout=checkpoint['dropout']
        ).to(device)

        classifier.model.load_state_dict(checkpoint['model_state'])
        classifier.is_fitted = True
        classifier.history = checkpoint.get('history', None)

        print(f"✓ DNN model loaded from {filepath}")
        return classifier