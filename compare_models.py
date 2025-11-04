import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from data_loader import RCV1DataLoader
from model import DNNClassifier
from baseline_models import BaselineModel
import os
import time
import json


class ModelComparator:
    
    def __init__(self, output_dir='comparison_results'):
        """Initialize comparator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        self.models = {}
        self.results = []
        
    def add_model(self, name, model, skip_training=False):
        """Add a model to the comparison list"""
        self.models[name] = {'model': model, 'skip_training': skip_training}
        print(f"✓ Added model: {name}" + (" (pre-trained)" if skip_training else ""))
    
    def load_data(self, max_train_samples=None):
        """Load data"""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        data_loader = RCV1DataLoader(data_dir='data')
        X_train, y_train = data_loader.load_data('train')
        X_val, y_val = data_loader.load_data('val')
        X_test, y_test = data_loader.load_data('test')
        metadata = data_loader.load_metadata()
        
        if max_train_samples and X_train.shape[0] > max_train_samples:
            print(f"Sampling {max_train_samples:,} training samples...")
            indices = np.random.choice(X_train.shape[0], max_train_samples, replace=False)
            X_train, y_train = X_train[indices], y_train[indices]
        
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.target_names = metadata['target_names']
        
        print(f"✓ Data loaded: train={X_train.shape[0]:,}, val={X_val.shape[0]:,}, test={X_test.shape[0]:,}")
    
    def train_all_models(self):
        """Train models"""
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        for name, model_dict in self.models.items():
            model, skip_training = model_dict['model'], model_dict['skip_training']
            if skip_training:
                print(f"\nSkipping training: {name} (already trained)")
                continue
            
            print(f"\nTraining: {name}")
            try:
                model.fit(self.X_train, self.y_train)
                model_path = os.path.join(self.output_dir, f'{name}_model.pkl')
                model.save(model_path)
            except Exception as e:
                print(f"✗ Failed to train {name}: {e}")
    
    def evaluate_all_models(self):
        """Evaluate models based only on Precision, Recall, and F1"""
        print("\n" + "="*70)
        print("EVALUATING MODELS (Precision, Recall, F1)")
        print("="*70)
        
        self.results = []
        
        for name, model_dict in self.models.items():
            if "dummy" in name.lower():
                continue
            model = model_dict['model']
            print(f"\nEvaluating: {name}")
            
            try:
                start_time = time.time()
                result = model.predict(self.X_test)
                if isinstance(result, tuple):
                    y_pred, pred_time = result
                else:
                    y_pred = result
                    pred_time = 0  # default prediction time
                y_test = self.y_test.toarray() if hasattr(self.y_test, 'toarray') else np.asarray(self.y_test)
                y_pred = np.asarray(y_pred)
                
                metrics = {
                    'model_name': name,
                    'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
                    'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                    'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'precision_micro': precision_score(y_test, y_pred, average='micro', zero_division=0),
                    'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                    'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
                    'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
                    'prediction_time': pred_time,
                    'training_time': getattr(model, 'training_time', 0)
                }
                
                self.results.append(metrics)
                
                print(f"  F1 (Micro): {metrics['f1_micro']:.4f} | Precision: {metrics['precision_micro']:.4f} | Recall: {metrics['recall_micro']:.4f}")
                
            except Exception as e:
                print(f"✗ Error evaluating {name}: {e}")
    
    def save_results(self):
        """Save results to CSV, JSON"""
        print("\nSAVING RESULTS")
        if not self.results:
            print("⚠ No results to save")
            return
        
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.output_dir, 'model_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ CSV saved: {csv_path}")
        
        json_path = os.path.join(self.output_dir, 'model_comparison.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ JSON saved: {json_path}")
    
    # =========================
    # PLOTTING METHODS
    # =========================
    def plot_comparison(self):
        if not self.results:
            print("⚠ No results to plot")
            return
        
        print("\nGENERATING COMPARISON PLOTS")
        df = pd.DataFrame(self.results)
        df = df[~df['model_name'].str.lower().str.contains('dummy')]  # skip dummy models
        
        plot_dir = os.path.join(self.output_dir, 'plots')
        
        # 1️⃣ Precision, Recall, F1 bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.25
        x = np.arange(len(df))
        ax.bar(x - width, df['precision_micro'], width, label='Precision', color='#3498db')
        ax.bar(x, df['recall_micro'], width, label='Recall', color='#e74c3c')
        ax.bar(x + width, df['f1_micro'], width, label='F1-score', color='#2ecc71')
        ax.set_xticks(x)
        ax.set_xticklabels(df['model_name'], rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall, and F1 Comparison')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'precision_recall_f1_comparison.png'), dpi=300)
        plt.close()
        print("✓ Precision-Recall-F1 comparison plot saved")

        # 2️⃣ F1 comparison (micro, macro, weighted)
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df))
        width = 0.25
        ax.bar(x - width, df['f1_micro'], width, label='F1 Micro', color='#1abc9c')
        ax.bar(x, df['f1_macro'], width, label='F1 Macro', color='#9b59b6')
        ax.bar(x + width, df['f1_weighted'], width, label='F1 Weighted', color='#f39c12')
        ax.set_xticks(x)
        ax.set_xticklabels(df['model_name'], rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title('F1 Scores (Micro, Macro, Weighted)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'f1_comparison.png'), dpi=300)
        plt.close()
        print("✓ F1 score comparison plot saved")

        # 3️⃣ Metrics heatmap
        metrics_cols = ['precision_micro', 'recall_micro', 'f1_micro', 'f1_macro', 'f1_weighted']
        heatmap_data = df.set_index('model_name')[metrics_cols]
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt=".3f")
        plt.title("Metrics Heatmap Across Models")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'metrics_heatmap.png'), dpi=300)
        plt.close()
        print("✓ Metrics heatmap saved")

        # 4️⃣ Time vs performance scatter
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='training_time', y='f1_micro', data=df, s=120, color='#2980b9')
        for i, row in df.iterrows():
            plt.text(row['training_time'], row['f1_micro'] + 0.005, row['model_name'], ha='center', fontsize=9)
        plt.xlabel('Training Time (s)')
        plt.ylabel('F1 (Micro)')
        plt.title('Training Time vs F1 Performance')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'time_vs_performance.png'), dpi=300)
        plt.close()
        print("✓ Time vs Performance plot saved")
    
    def print_summary(self):
        """Print ranking summary"""
        if not self.results:
            print("⚠ No results available")
            return
        
        print("\n" + "="*70)
        print("COMPARISON SUMMARY (Precision, Recall, F1)")
        print("="*70)
        
        df = pd.DataFrame(self.results).sort_values('f1_micro', ascending=False)
        for i, row in enumerate(df.itertuples(), 1):
            print(f"{i}. {row.model_name:25s}  "
                  f"F1: {row.f1_micro:.4f}  "
                  f"P: {row.precision_micro:.4f}  "
                  f"R: {row.recall_micro:.4f}  "
                  f"Train: {row.training_time:.2f}s  Pred: {row.prediction_time:.2f}s")


def main():
    print("\n" + "="*70)
    print(" MODEL COMPARISON: PRECISION, RECALL, F1 ONLY ")
    print("="*70)
    
    comparator = ModelComparator(output_dir='comparison_results')
    comparator.load_data(max_train_samples=50000)
    
    comparator.add_model('Logistic-Regression', BaselineModel('logistic', max_iter=1000, n_jobs=-1))
    comparator.add_model('SGD-Classifier', BaselineModel('sgd', max_iter=1000, n_jobs=-1))
    comparator.add_model('Random-Forest', BaselineModel('rf', n_estimators=100, max_depth=20, n_jobs=-1))
    comparator.add_model('Naive-Bayes', BaselineModel('nb', alpha=1.0, n_jobs=-1))
    comparator.add_model('Decision-Tree', BaselineModel('dt', max_depth=20, n_jobs=-1))
    
    try:
        dnn_model = DNNClassifier.load('models/dnn_model.pth')
        comparator.add_model('DNN (Our Model)', dnn_model, skip_training=True)
    except:
        print("⚠ DNN model not found. Train it first with main.py")
    
    comparator.train_all_models()
    comparator.evaluate_all_models()
    comparator.save_results()
    comparator.plot_comparison()
    comparator.print_summary()
    
    print("\nComparison completed! Results saved in 'comparison_results/'\n")


if __name__ == '__main__':
    main()
