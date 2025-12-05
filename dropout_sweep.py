
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import torch
import warnings
from data_loader import RCV1DataLoader
from model import DNNClassifier
from evaluator import MultiLabelEvaluator

warnings.filterwarnings('ignore')

def main():
    print("\n" + "="*70)
    print(" DROPOUT RATE SWEEP FOR DNN")
    print("="*70)

    # 1. Load Data
    data_loader = RCV1DataLoader(data_dir='data')
    evaluator = MultiLabelEvaluator(output_dir='results/dropout_sweep')
    os.makedirs('results/dropout_sweep', exist_ok=True)

    print("\nLoading Data...")
    try:
        X_train, y_train = data_loader.load_data('train')
        
        # Check if data is RCV1 (103 categories)
        if y_train.shape[1] == 103:
            if X_train.shape[0] < 100000:
                 print(f"Existing data is a sampled subset ({X_train.shape[0]} samples). Switching to full dataset...")
                 raise FileNotFoundError("Sampled data found, forcing full download")
            
            print("✓ Loaded existing RCV1 data")
            X_val, y_val = data_loader.load_data('val')
            X_test, y_test = data_loader.load_data('test')
            metadata = data_loader.load_metadata()
            target_names = metadata['target_names']
        else:
            print(f"Existing data has {y_train.shape[1]} categories (expected 103). Regenerating...")
            raise FileNotFoundError("Data mismatch")
            
    except FileNotFoundError:
        print("Downloading and processing RCV1 dataset...")
        X_train, y_train, X_val, y_val, X_test, y_test, target_names = \
            data_loader.download_and_split(
                test_size=0.2,
                val_size=0.1,
                sample_size=None
            )

    # Sample training data for efficiency if needed
    max_train_samples = 200000 if torch.cuda.is_available() else 50000
    if X_train.shape[0] > max_train_samples:
        indices = np.random.choice(X_train.shape[0], max_train_samples, replace=False)
        X_train_dnn = X_train[indices]
        y_train_dnn = y_train[indices]
        print(f"✓ Sampled {max_train_samples:,} training samples")
    else:
        X_train_dnn = X_train
        y_train_dnn = y_train

    # 2. Define Sweep Parameters
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    results = []

    print("\n" + "="*70)
    print(f"STARTING SWEEP: Dropout Rates {dropout_rates}")
    print("="*70)

    for dropout in dropout_rates:
        print(f"\nTesting Dropout Rate: {dropout}")
        print("-" * 30)

        # Initialize model
        model = DNNClassifier(
            hidden_layers=[512, 128],
            activation='relu',
            dropout=dropout,
            learning_rate=0.001,
            batch_size=512,
            epochs=20,  # Reduced epochs for sweep speed, adjust if needed
            device='auto',
            random_state=42
        )

        # Train
        model.fit(X_train_dnn, y_train_dnn, X_val, y_val)

        # Evaluate on Validation Set
        pred_threshold = 0.5
        y_val_pred = model.predict(X_val, threshold=pred_threshold)
        
        # Calculate metrics manually or use evaluator
        # Using evaluator's method but capturing the dict
        # We need to suppress print output from evaluator if possible, or just accept it
        metrics = evaluator.evaluate(y_val, y_val_pred, target_names, f'Val_Dropout_{dropout}')
        
        # Store results
        result_entry = {
            'dropout_rate': dropout,
            'f1_micro': metrics['f1_micro'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'precision_micro': metrics['precision_micro'],
            'recall_micro': metrics['recall_micro'],
            'subset_accuracy': metrics['subset_accuracy'],
            'hamming_loss': metrics['hamming_loss']
        }
        results.append(result_entry)
        
        # Save intermediate results
        with open('results/dropout_sweep/sweep_results_intermediate.json', 'w') as f:
            json.dump(results, f, indent=2)

    # 3. Save Final Results
    with open('results/dropout_sweep/sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Sweep results saved to results/dropout_sweep/sweep_results.json")

    # 4. Plot Results
    plot_sweep_results(results)

def plot_sweep_results(results):
    rates = [r['dropout_rate'] for r in results]
    f1_micro = [r['f1_micro'] for r in results]
    f1_macro = [r['f1_macro'] for r in results]
    precision = [r['precision_micro'] for r in results]
    recall = [r['recall_micro'] for r in results]

    plt.figure(figsize=(12, 8))
    
    plt.plot(rates, f1_micro, marker='o', linewidth=2, label='F1 Micro')
    plt.plot(rates, f1_macro, marker='s', linewidth=2, label='F1 Macro')
    plt.plot(rates, precision, marker='^', linewidth=2, linestyle='--', label='Precision (Micro)')
    plt.plot(rates, recall, marker='v', linewidth=2, linestyle='--', label='Recall (Micro)')

    plt.title('Impact of Dropout Rate on Model Performance', fontsize=16)
    plt.xlabel('Dropout Rate', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rates)
    
    # Annotate best F1 Micro
    best_idx = np.argmax(f1_micro)
    best_rate = rates[best_idx]
    best_score = f1_micro[best_idx]
    plt.annotate(f'Best: {best_score:.4f}', 
                 xy=(best_rate, best_score), 
                 xytext=(best_rate, best_score + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 ha='center')

    plt.tight_layout()
    plt.savefig('results/dropout_sweep/dropout_performance.png', dpi=300)
    print("✓ Plot saved to results/dropout_sweep/dropout_performance.png")
    plt.close()

if __name__ == '__main__':
    main()
