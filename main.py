
from data_loader import RCV1DataLoader
from model import DNNClassifier, MultiLabelClassifier
from evaluator import MultiLabelEvaluator
import numpy as np
import warnings
import torch
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')


def main():
    # ============= 配置区域 =============
    USE_RCV1_BALANCED = True   # True: 使用平衡后的 RCV1, False: 使用 20 Newsgroups
    BALANCE_STRATEGY = 'drop'  # 'drop': 删除稀有类, 'oversample': 过采样
    # ===================================

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

    evaluator = MultiLabelEvaluator(output_dir='results/plots')

    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)

    if USE_RCV1_BALANCED:
        # 使用平衡后的 RCV1 数据
        if BALANCE_STRATEGY == 'drop':
            data_dir = 'data/balanced_drop'
            print("🔹 Using Balanced RCV1 (Drop Rare Classes Strategy)")
        else:
            data_dir = 'data/balanced_oversample'
            print("🔹 Using Balanced RCV1 (Oversampling Strategy)")
        
        data_loader = RCV1DataLoader(data_dir=data_dir)
        X_train, y_train = data_loader.load_data('train')
        X_val, y_val = data_loader.load_data('val')
        X_test, y_test = data_loader.load_data('test')
        metadata = data_loader.load_metadata()
        target_names = metadata['target_names']
        
        print(f"✓ Loaded balanced RCV1 data")
        print(f"  Classes: {len(target_names)}")
        print(f"  Training samples: {X_train.shape[0]:,}")
        print(f"  Features: {X_train.shape[1]:,}")
        
        # RCV1 是稀疏矩阵，需要转换为密集格式给 DNN
        from scipy.sparse import issparse
        if issparse(X_train):
            print("\nConverting sparse matrices to dense (this may take time)...")
            X_train = X_train.toarray()
            X_val = X_val.toarray()
            X_test = X_test.toarray()
            print("✓ Converted to dense format")
    
    else:
        # 使用 20 Newsgroups + BERT embeddings
        data_loader = RCV1DataLoader(data_dir='data')
        print("🔹 Using 20 Newsgroups + BERT Embeddings")
        
        try:
            X_train, y_train = data_loader.load_data('train')
            
            # Check if data is likely BERT (768 features)
            if X_train.shape[1] == 768:
                print("✓ Loaded existing BERT embeddings")
                X_val, y_val = data_loader.load_data('val')
                X_test, y_test = data_loader.load_data('test')
                metadata = data_loader.load_metadata()
                target_names = metadata['target_names']
            else:
                print(f"Existing data has {X_train.shape[1]} features (expected 768 for BERT). Regenerating...")
                raise FileNotFoundError("Data mismatch")
                
        except FileNotFoundError:
            print("Generating BERT embeddings from raw text...")
            # Using 20 Newsgroups dataset with BERT embeddings
            X_train, y_train, X_val, y_val, X_test, y_test, target_names = \
                data_loader.download_raw_and_embed(
                    test_size=0.2,
                    val_size=0.1,
                    max_samples=500000  # Adjust this based on your GPU/CPU capabilities
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

    # 根据数据集选择网络配置
    if USE_RCV1_BALANCED:
        # RCV1: 47,236 维 TF-IDF 特征，需要更深的网络
        print("📊 Configuring DNN for RCV1 (high-dimensional TF-IDF features)")
        model = DNNClassifier(
            hidden_layers=[2048, 1024, 512],  # 更深的网络适应高维输入
            activation='leaky_relu',
            dropout=0.3,                             # 降低 dropout
            learning_rate=0.0005,                    # 降低初始学习率
            batch_size=256,                          # 更小的 batch
            epochs=30,
            device='auto',
            random_state=42,
            lr_scheduler='plateau',
            lr_patience=2,
            lr_factor=0.5,
            lr_min=1e-6,
            early_stopping=True,                     # 启用 early stopping
            early_stopping_patience=3                # 3个epoch无改善则停止
        )
    else:
        # 20 Newsgroups: 768 维 BERT embeddings
        print("📊 Configuring DNN for 20 Newsgroups (BERT embeddings)")
        model = DNNClassifier(
            hidden_layers=[512, 256, 128],
            activation='relu',
            dropout=0.5,
            learning_rate=0.001,
            batch_size=512,
            epochs=30,
            device='auto',
            random_state=42,
            lr_scheduler='plateau',
            lr_patience=2,
            lr_factor=0.5,
            lr_min=1e-6
        )

    history = model.fit(X_train_dnn, y_train_dnn, X_val, y_val)
    model.save('models/dnn_model.pth')

    print("\n" + "="*70)
    print("STEP 3: FINDING OPTIMAL THRESHOLD")
    print("="*70)
    
    # Search for optimal threshold on validation set
    print("Searching for optimal prediction threshold on validation set...")
    best_threshold = 0.5
    best_f1 = 0
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    
    for thresh in thresholds:
        y_val_pred_temp = model.predict(X_val, threshold=thresh)
        f1 = f1_score(y_val, y_val_pred_temp, average='micro', zero_division=0)
        print(f"  Threshold {thresh:.2f}: F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    pred_threshold = best_threshold
    print(f"\n✓ Optimal threshold: {pred_threshold:.2f} (Validation F1: {best_f1:.4f})")
    
    # Save threshold for compare_models.py
    import json
    with open('models/best_threshold.json', 'w') as f:
        json.dump({'threshold': pred_threshold, 'val_f1': best_f1}, f)
    print(f"✓ Saved threshold to models/best_threshold.json")

    print("\n" + "="*70)
    print("STEP 4-6: COMPREHENSIVE EVALUATION")
    print("="*70)
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

    # Check if learning rate is available
    has_lr = 'learning_rate' in history and history['learning_rate']
    
    if has_lr:
        # Create figure with two subplots (loss and learning rate)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot 1: Loss curves
        ax1.plot(history['epoch'], history['train_loss'], label='Training Loss', marker='o', linewidth=2)
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(history['epoch'], history['val_loss'], label='Validation Loss', marker='s', linewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3, linestyle='--')
        
        # Plot 2: Learning rate schedule
        ax2.plot(history['epoch'], history['learning_rate'], label='Learning Rate', 
                marker='o', linewidth=2, color='green')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_yscale('log')  # Use log scale for better visualization
        
        plt.tight_layout()
    else:
        # Fallback to single plot if no learning rate data
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