#!/bin/bash
#SBATCH --job-name=compare_models
#SBATCH --output=/scratch/bingxu97/logs/compare_models_%j.out
#SBATCH --error=/scratch/bingxu97/logs/compare_models_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug

# Set resource limits to avoid OpenBLAS threading issues
ulimit -u 16384  # Increase max user processes
ulimit -s unlimited  # Set unlimited stack size

# OpenBLAS environment variables to control threading
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# Prevent excessive thread spawning
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

# Additional OpenBLAS settings
export OPENBLAS_VERBOSE=0
export OPENBLAS_MAIN_FREE=1

# Set working directory
PROJECT_ROOT="/home/bingxu97/links/scratch/COMP6321_Project"
cd $PROJECT_ROOT

# Load required modules for shared libraries  
module load StdEnv/2023
module load gcc/12.3
module load python/3.10
module load cuda/12.9
module load python/3.10
module load cuda/12.9

# Create necessary directories in /scratch
mkdir -p "/scratch/bingxu97/logs"
mkdir -p "/scratch/bingxu97/models"
mkdir -p "/scratch/bingxu97/results/plots"

# Activate virtual environment
VENV_DIR="/scratch/bingxu97/comp6321_env"
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Set environment variables for data caching
export SCIKIT_LEARN_DATA="/scratch/bingxu97/scikit_learn_data"
export TORCH_HOME="/scratch/bingxu97/torch_cache"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Verify environment activation
echo "Python path: $(which python)"

echo "Starting model comparison at $(date)"
echo "Running on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: ${SLURM_MEM_PER_NODE}MB"
echo ""

# Check resource limits
echo "Current resource limits:"
ulimit -a
echo ""

# Check Python and NumPy installation
echo "Active conda environment:"
conda env list | grep '*'
echo ""

echo "Python version and location:"
which python
python --version
echo ""

echo "Checking NumPy installation:"
python -c "import sys; print('Python executable:', sys.executable)"
python -c "import numpy; print('NumPy version:', numpy.__version__); print('NumPy location:', numpy.__file__)" || echo "NumPy not found!"
echo ""

echo "Checking other required packages:"
python -c "import pandas; print('Pandas version:', pandas.__version__)" || echo "Pandas not found!"
python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)" || echo "Scikit-learn not found!"
echo ""
cd /home/bingxu97/links/scratch/COMP6321_Project
# Run the Python script
echo "Running main.py to train DNN..."
python main.py
if [ $? -ne 0 ]; then
    echo "main.py failed! Exiting."
    exit 1
fi

echo "Running compare_models.py..."
python compare_models.py

echo ""
echo "Model comparison completed at $(date)"
echo "Exit code: $?"