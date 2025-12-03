#!/bin/bash
# Generate data in background with logging

cd /home/bingxu97/links/scratch/COMP6321_Project
source /scratch/bingxu97/comp6321_env/bin/activate

echo "Starting data generation at $(date)"
echo "This will take approximately 20-30 minutes..."
echo "Monitor progress: tail -f /tmp/generate_data_$(whoami).log"

python -c "
from data_loader import RCV1DataLoader
import sys

print('='*70)
print('Generating full 20 Newsgroups dataset with BERT embeddings...')
print('='*70)

data_loader = RCV1DataLoader(data_dir='data')

try:
    X_train, y_train, X_val, y_val, X_test, y_test, target_names = \
        data_loader.download_raw_and_embed(
            test_size=0.2,
            val_size=0.1,
            max_samples=500000
        )
    
    print('\n' + '='*70)
    print('SUCCESS! Dataset generated:')
    print('='*70)
    print(f'Training: {X_train.shape[0]:,} samples')
    print(f'Validation: {X_val.shape[0]:,} samples')
    print(f'Test: {X_test.shape[0]:,} samples')
    print(f'Features: {X_train.shape[1]} (BERT embeddings)')
    print('='*70)
    
except Exception as e:
    print(f'\nERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee /tmp/generate_data_$(whoami).log

echo "Data generation completed at $(date)"
