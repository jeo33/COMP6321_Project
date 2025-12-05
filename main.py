
from data_loader import RCV1DataLoader
from model import DNNClassifier, MultiLabelClassifier
from evaluator import MultiLabelEvaluator
import numpy as np
import warnings
import torch

warnings.filterwarnings('ignore')


def main():

    print("\n" + "="*70)
    print(" MULTI-LABEL CLASSIFICATION WITH DEEP NEURAL NETWORK")
    print("="*70)

    print("\nGPU Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠ WARNING: No GPU detected, will use CPU")

    data_loader = RCV1DataLoader(data_dir='data')
    evaluator = MultiLabelEvaluator(output_dir='results/plots')

    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)

    try:
        X_train, y_train = data_loader.load_data('train')
        
        # Check if data is RCV1 (103 categories)
        if y_train.shape[1] == 103:
            # Check if it's the full dataset (approx > 100k samples)
            if X_train.shape[0] < 100000:
                print(f"Existing data is a sampled subset ({X_train.shape[0]} samples). Switching to full dataset...")
                raise FileNotFoundError("Sampled data found, forcing full download")

            print("✓ Loaded existing RCV1 data")
            X_val, y_val = data_loader.load_data('val')
            X_test, y_test = data_loader.load_data('test')
            metadata = data_loader.load_metadata()
            target_names = metadata['target_names']
        else:
            print(f"Existing data has {y_train.shape[1]} categories (expected 103 for RCV1). Regenerating...")
            raise FileNotFoundError("Data mismatch")
            
    except FileNotFoundError:
        print("Downloading and processing RCV1 dataset...")
        # Using RCV1 dataset (Multi-label, TF-IDF features)
        X_train, y_train, X_val, y_val, X_test, y_test, target_names = \
            data_loader.download_and_split(
                test_size=0.2,
                val_size=0.1,
                sample_size=None  # Use full dataset
            )

    # Sample training data for DNN
    print("\n" + "="*70)
    print("SAMPLING DATA FOR DNN")
    print("="*70)
    print("Note: DNN requires dense matrices which use more memory.")
    print("Sampling a subset for efficient training...")

    # Use more samples with GPU
    max_train_samples = 200000 if torch.cuda.is_available() else 50000
    if X_train.shape[0] > max_train_samples:
        indices = np.random.choice(X_train.shape[0], max_train_samples, replace=False)
        X_train_dnn = X_train[indices]
        y_train_dnn = y_train[indices]
        print(f"✓ Sampled {max_train_samples:,} training samples")
    else:
        X_train_dnn = X_train
        y_train_dnn = y_train

    print("\n" + "="*70)
    print("STEP 2: TRAINING DNN MODEL ON GPU")
    print("="*70)

    model = DNNClassifier(
        hidden_layers=[512, 128],   # 2 hidden layers 512,128
        #hidden_layers=[32768,8192,2048,512, 256, 128],  # 6 hidden layers
        activation='leaky_relu',                # ReLU activation
        dropout=0.5,                      # 50% dropout
        learning_rate=0.001,              
        batch_size=512,                   
        epochs=50,                        
        device='auto',                    
        random_state=42
    )

    history = model.fit(X_train_dnn, y_train_dnn, X_val, y_val)
    model.save('models/dnn_model.pth')

    print("\n" + "="*70)
    print("STEP 3-5: COMPREHENSIVE EVALUATION")
    print("="*70)

    # Use a higher threshold to reduce false positives
    pred_threshold = 0.5
    print(f"Using prediction threshold: {pred_threshold}")

    print("\nEvaluating on training set...")
    y_train_pred = model.predict(X_train_dnn, threshold=pred_threshold)
    train_metrics = evaluator.evaluate(y_train_dnn, y_train_pred,
                                      target_names, 'Training')

    print("\nEvaluating on validation set...")
    y_val_pred = model.predict(X_val, threshold=pred_threshold)
    val_metrics = evaluator.evaluate(y_val, y_val_pred, target_names, 'Validation')

    print("\nEvaluating on test set...")
    y_test_pred = model.predict(X_test, threshold=pred_threshold)
    test_metrics = evaluator.evaluate(y_test, y_test_pred, target_names, 'Test')


    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS FROM TEST SET")
    print("="*70)
    
    evaluator.print_sample_predictions(
        X_test, y_test, y_test_pred, 
        target_names=target_names, 
        n_samples=10
    )
    
    evaluator.print_detailed_predictions(
        X_test, y_test, y_test_pred,
        target_names=target_names,
        sample_indices=[0, 10, 50, 100, 500]
    )
    
    evaluator.compare_predictions_summary(
        y_test, y_test_pred,
        target_names=target_names
    )

    print("\n" + "="*70)
    print("STEP 6: GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*70)

    evaluator.plot_all_analyses(
        y_train_dnn, y_train_pred,
        y_val, y_val_pred,
        y_test, y_test_pred,
        train_metrics, val_metrics, test_metrics,
        target_names
    )

    if history:
        plot_training_history(history)

    
    print("\n" + "="*70)
    print("PROJECT SUMMARY")
    print("="*70)
    print(f"\n✓ DNN Training completed successfully!")
    print(f"✓ Device used: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"✓ Model saved to: models/dnn_model.pth")
    print(f"✓ All plots saved to: results/plots/")

    print(f"\n{'─'*70}")
    print("FINAL TEST RESULTS")
    print(f"{'─'*70}")
    print(f"F1 (Micro):        {test_metrics['f1_micro']:.4f}")
    print(f"F1 (Macro):        {test_metrics['f1_macro']:.4f}")
    print(f"F1 (Weighted):     {test_metrics['f1_weighted']:.4f}")
    print(f"Precision (Micro): {test_metrics['precision_micro']:.4f}")
    print(f"Recall (Micro):    {test_metrics['recall_micro']:.4f}")
    print(f"Subset Accuracy:   {test_metrics['subset_accuracy']:.4f}")
    print(f"Jaccard Score:     {test_metrics['jaccard_score']:.4f}")
    print(f"Hamming Loss:      {test_metrics['hamming_loss']:.4f}")

    print("\n" + "="*70)
    print(" PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")


def plot_training_history(history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], label='Training Loss', marker='o')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['epoch'], history['val_loss'], label='Validation Loss', marker='s')

    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig('results/plots/00_training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: results/plots/00_training_history.png")
    plt.close()


if __name__ == '__main__':
    main()