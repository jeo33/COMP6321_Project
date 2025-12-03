#!/bin/bash
#SBATCH --job-name=compare_models
#SBATCH --output=compare_models_%j.out
#SBATCH --error=compare_models_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=compute

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
cd /home/gjp1993/links/scratch/COMP6321_Project

# Load required modules for shared libraries  
module load StdEnv/2023
module load gcc/12.3

# Initialize conda properly
export CONDA_BASE_ENV="/home/gjp1993/miniconda3"
source $CONDA_BASE_ENV/etc/profile.d/conda.sh

# Activate ml_project environment
echo "Activating ml_project conda environment..."
conda deactivate 2>/dev/null || true
conda activate ml_project

# Verify environment activation
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# Fix library path issues
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Clean Python environment variables to avoid conflicts
unset PYTHONPATH
export PYTHONNOUSERSITE=1

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