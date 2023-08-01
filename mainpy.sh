#!/bin/bash -l

#SBATCH --account=courses0101
#SBATCH --job-name=superExecution
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --export=none
#SBATCH --exclusive


# Defining the working dir
workingDir="$MYSCRATCH/data/FFA-IR-main/code"


# Entering the working dir
cd $workingDir

#Load singularity
#module load singularity/3.8.6

module load python/3.9.15
python get-pip.py



export PATH=$PATH:/home/lbassi/.local/bin
export PYTHONPATH=$PYTHONPATH:/scratch/courses0101/lbassi

pip install torch
pip install torchvision
pip install opencv-python

export TORCH_HOME=/$MYSCRATCH:/

#Load container
#singularity exec -B $MYSCRATCH:/home  python_3.10.9-slim.sif pip list


# Activate the environment
#if [ -d "env" ]; then
#    source env/bin/activate
#else
#    echo "Virtual environment doesn't exist. Please create one and rerun the script."
#    exit 1
#fi

# Print the list of python packages in the container
#singularity exec -B $MYSCRATCH:/mnt python_3.10.9-slim.sif pip list

# Supercomputing execution
srun python3 main.py
#srun singularity exec -B $MYSCRATCH:/mnt python_3.10.9-slim.sif python3 main.py


# Deactivate the environment
#deactivate

# Successfully finished
echo "Done"
exit 0