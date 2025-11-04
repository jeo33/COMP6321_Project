import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from scipy.sparse import issparse
import pickle
import time


class BaselineModel:
    """
    Generic wrapper for baseline multi-label models using scikit-learn.
    Supports: logistic, sgd, rf, nb, svm, dt
    """

    def __init__(self, model_type='logistic', name=None, **kwargs):
        """
        Args:
            model_type: Type of model ('logistic', 'sgd', 'rf', 'nb', 'svm', 'dt')
            name: Custom name for the model
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.name = name or model_type.upper()
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        self.training_time = 0

    def build_model(self):
        """Initialize model based on type."""

        if self.model_type == 'logistic':
            from sklearn.multiclass import OneVsRestClassifier
            self.model = OneVsRestClassifier(
                LogisticRegression(
                    max_iter=self.kwargs.get('max_iter', 1000),
                    random_state=self.kwargs.get('random_state', 42),
                    solver='lbfgs',
                    n_jobs=1
                ),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )

        elif self.model_type == 'sgd':
            from sklearn.multiclass import OneVsRestClassifier
            self.model = OneVsRestClassifier(
                SGDClassifier(
                    loss='log_loss',
                    max_iter=self.kwargs.get('max_iter', 1000),
                    random_state=self.kwargs.get('random_state', 42),
                    n_jobs=1
                ),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )

        elif self.model_type == 'rf':
            base_model = RandomForestClassifier(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', 20),
                random_state=self.kwargs.get('random_state', 42),
                n_jobs=1,
                verbose=0
            )
            self.model = MultiOutputClassifier(base_model, n_jobs=self.kwargs.get('n_jobs', -1))

        elif self.model_type == 'nb':
            base_model = MultinomialNB(alpha=self.kwargs.get('alpha', 1.0))
            self.model = MultiOutputClassifier(base_model, n_jobs=self.kwargs.get('n_jobs', -1))

        elif self.model_type == 'svm':
            from sklearn.multiclass import OneVsRestClassifier
            self.model = OneVsRestClassifier(
                LinearSVC(
                    max_iter=self.kwargs.get('max_iter', 1000),
                    random_state=self.kwargs.get('random_state', 42),
                    dual=False
                ),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )

        elif self.model_type == 'dt':
            base_model = DecisionTreeClassifier(
                max_depth=self.kwargs.get('max_depth', 20),
                random_state=self.kwargs.get('random_state', 42)
            )
            self.model = MultiOutputClassifier(base_model, n_jobs=self.kwargs.get('n_jobs', -1))

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X, y):
        """Train the model on input data."""
        print(f"\nTraining {self.name}...")

        if self.model is None:
            self.build_model()

        # Convert labels to dense if sparse
        if issparse(y):
            print("  Converting labels to dense...")
            y = y.toarray()

        # Naive Bayes needs dense features
        if self.model_type == 'nb' and issparse(X):
            print("  Converting features to dense for Naive Bayes...")
            X = X.toarray()

        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True

        print(f"✓ {self.name} training completed in {self.training_time:.2f}s")
        return self

    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.model_type == 'nb' and issparse(X):
            X = X.toarray()

        start_time = time.time()
        predictions = self.model.predict(X)
        prediction_time = time.time() - start_time

        return predictions, prediction_time

    def save(self, filepath):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ {self.name} saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model


# --------------------------------------------
# Optional: Simple Dummy Baseline for comparison
# --------------------------------------------

class DummyBaseline(BaselineModel):
    """
    A dummy baseline using scikit-learn's DummyClassifier.
    Useful for sanity checks or performance lower bounds.
    """

    def __init__(self, strategy='most_frequent', **kwargs):
        super().__init__(model_type='dummy', name=f"Dummy-{strategy}", **kwargs)
        self.strategy = strategy

    def build_model(self):
        """Use DummyClassifier with a chosen strategy."""
        self.model = MultiOutputClassifier(
            DummyClassifier(strategy=self.strategy, random_state=self.kwargs.get('random_state', 42))
        )
