
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
                 activation='relu', dropout=0.5):
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

    def __init__(self, hidden_layers=[2048, 1024, 512], activation='relu',
                 dropout=0.3, learning_rate=0.0005, batch_size=256,
                 epochs=50, device='auto', random_state=42,
                 lr_scheduler='plateau', lr_patience=2, lr_factor=0.5,
                 lr_min=1e-6, lr_t_max=None, early_stopping=True, 
                 early_stopping_patience=3):
        """
        Args:
            hidden_layers: List of neurons per hidden layer [512, 256, 128]
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
            dropout: Dropout rate for regularization (0.0 to 1.0)
            learning_rate: Initial learning rate for Adam optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: 'cuda', 'cpu', or 'auto'
            random_state: Random seed
            lr_scheduler: Learning rate scheduler type ('plateau', 'cosine', 'none')
                - 'plateau': ReduceLROnPlateau (reduces LR when validation loss plateaus)
                - 'cosine': CosineAnnealingLR (cosine annealing schedule)
                - 'none': No learning rate scheduling
            lr_patience: Patience for ReduceLROnPlateau (epochs to wait before reducing)
            lr_factor: Factor by which to reduce learning rate (new_lr = lr * factor)
            lr_min: Minimum learning rate
            lr_t_max: T_max for CosineAnnealingLR (defaults to epochs if None)
            early_stopping: Enable early stopping based on validation loss
            early_stopping_patience: Number of epochs with no improvement before stopping
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        
        # Learning rate scheduler parameters
        self.lr_scheduler_type = lr_scheduler
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.lr_t_max = lr_t_max if lr_t_max is not None else epochs
        
        # Early stopping parameters
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience

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
        self.scheduler = None

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
        pos_weights = torch.FloatTensor(n_negatives / (n_positives + 1e-6)).to(self.device)
        
        print(f"Using weighted loss to handle class imbalance (Avg weight: {pos_weights.mean().item():.2f})")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Initialize learning rate scheduler
        if self.lr_scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=self.lr_factor, 
                patience=self.lr_patience,
                min_lr=self.lr_min,
                verbose=True
            )
            print(f"✓ Using ReduceLROnPlateau scheduler (patience={self.lr_patience}, factor={self.lr_factor}, min_lr={self.lr_min})")
            if X_val is None or y_val is None:
                print("⚠ Warning: ReduceLROnPlateau requires validation data. Scheduler will use training loss.")
        elif self.lr_scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.lr_t_max,
                eta_min=self.lr_min
            )
            print(f"✓ Using CosineAnnealingLR scheduler (T_max={self.lr_t_max}, eta_min={self.lr_min})")
        else:
            self.scheduler = None
            print("✓ No learning rate scheduler")

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
        print(f"Batch size: {self.batch_size}, Initial learning rate: {self.learning_rate}")
        if self.early_stopping and val_loader is not None:
            print(f"Early stopping enabled (patience={self.early_stopping_patience})")
        print("-"*70)

        self.history = {'train_loss': [], 'val_loss': [], 'epoch': [], 'learning_rate': []}
        
        # Early stopping variables
        best_val_loss = float('inf')
        epochs_no_improve = 0
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
            
            # Record current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

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
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                      f"LR: {current_lr:.6f}")
                
                # Early stopping check
                if self.early_stopping:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                        best_model_state = self.model.state_dict().copy()
                        print(f"  → New best validation loss: {best_val_loss:.4f}")
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.early_stopping_patience:
                            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                            print(f"  Best validation loss: {best_val_loss:.4f} at epoch {epoch+1-epochs_no_improve}")
                            if best_model_state is not None:
                                self.model.load_state_dict(best_model_state)
                                print(f"  Restored best model weights")
                            break
                
                # Step scheduler based on type
                if self.scheduler is not None:
                    if self.lr_scheduler_type == 'plateau':
                        self.scheduler.step(avg_val_loss)
                    elif self.lr_scheduler_type == 'cosine':
                        self.scheduler.step()
            else:
                print(f"Epoch [{epoch+1:3d}/{self.epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.6f}")
                
                # Step scheduler for cosine (plateau needs validation loss)
                if self.scheduler is not None and self.lr_scheduler_type == 'cosine':
                    self.scheduler.step()
                elif self.scheduler is not None and self.lr_scheduler_type == 'plateau':
                    # Use training loss if no validation data
                    self.scheduler.step(avg_train_loss)

        self.is_fitted = True
        print("-"*70)
        print("✓ Training completed!")
        
        # Print learning rate schedule summary
        if self.scheduler is not None:
            print(f"\nLearning Rate Schedule:")
            print(f"  Initial LR: {self.history['learning_rate'][0]:.6f}")
            print(f"  Final LR:   {self.history['learning_rate'][-1]:.6f}")
            print(f"  Min LR:     {min(self.history['learning_rate']):.6f}")
            print(f"  Max LR:     {max(self.history['learning_rate']):.6f}")
        
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
            'history': self.history,
            'lr_scheduler_type': self.lr_scheduler_type,
            'lr_patience': self.lr_patience,
            'lr_factor': self.lr_factor,
            'lr_min': self.lr_min,
            'lr_t_max': self.lr_t_max,
            'early_stopping': self.early_stopping,
            'early_stopping_patience': self.early_stopping_patience
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
            device=device,
            lr_scheduler=checkpoint.get('lr_scheduler_type', 'none'),
            lr_patience=checkpoint.get('lr_patience', 3),
            lr_factor=checkpoint.get('lr_factor', 0.5),
            lr_min=checkpoint.get('lr_min', 1e-6),
            lr_t_max=checkpoint.get('lr_t_max', 50),
            early_stopping=checkpoint.get('early_stopping', False),
            early_stopping_patience=checkpoint.get('early_stopping_patience', 5)
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