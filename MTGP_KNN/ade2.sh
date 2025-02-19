#!/bin/bash
#SBATCH --job-name=mgp-part-lsvm
#SBATCH --output=out_array_%A_%a.out
#SBATCH --error=out_array_%A_%a.err
#SBATCH --array=1-30
#SBATCH --time=2-35:00
#SBATCH --partition=parallel
#SBATCH --mem-per-cpu=5000M
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wangpeng@ecs.vuw.ac.nz

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

file_path=/nfs/scratch/wangpe/GPFC/algorithms2
echo $SLURM_JOB_NODELIST

module load python/3.8.1
source /nfs/home/wangpe/python_test/mytest/bin/activate
python $file_path/GPFC.py $1 $SLURM_ARRAY_TASK_ID
# echo $right

mv *.txt  /nfs/scratch/wangpe/GPFC/results/MGP-part-lsvm/$1
mv *.npy  /nfs/scratch/wangpe/GPFC/results/MGP-part-lsvm/$1
mv *.pickle  /nfs/scratch/wangpe/GPFC/results/MGP-part-lsvm/$1
