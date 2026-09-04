#!/bin/bash
#SBATCH --job-name=meg_decode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4            # 4 CPUs per subject (used by SlidingEstimator n_jobs=-1)
#SBATCH --mem=32GB                   # MEG epochs + decoding — 32GB is safe
#SBATCH --time=02:00:00              # 2 hrs per subject is generous; adjust down after first run
#SBATCH --array=0-47%5               # 48 subjects, max 5 running at once
#SBATCH --output=/scratch/djn8068/meg_project/logs/decode_v3_%A_%a.out
#SBATCH --error=/scratch/djn8068/meg_project/logs/decode_v3_%A_%a.err
#SBATCH --mail-type=ARRAY_TASKS_DONE,FAIL
#SBATCH --mail-user=djn8068@nyu.edu
#SBATCH --account=torch_pr_279_tandon_priority

# =============================================================================
# SUBJECT LIST
# Must match the order in batch_autoreject.py exactly.
# SLURM_ARRAY_TASK_ID (0-29) indexes into this list.
# =============================================================================

SUBJECTS=(
    R2312 R2538 R2611 R2612 R2621 R2649 R2650 R2652
    R2657 R2662 R2663 R2664 R2665 R2667 R2669 R2671
    R2672 R2674 R2675 R2676 R2678 R2679 R2682 R2683
    R2684 R2688 R2700 R2701 R2702 R2726 R2728 R2733
    R2734 R2743 R2749 R2760 R2763 R2765 R2772 R2775
    R2776 R2780 R2781 R2787 R2790 R2792 R2795 R2796
)

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "=============================="
echo "Job array index: $SLURM_ARRAY_TASK_ID"
echo "Subject:         $SUBJECT"
echo "Node:            $HOSTNAME"
echo "Start time:      $(date)"
echo "=============================="

# =============================================================================
# PATHS
# =============================================================================

PROJECT_DIR=/scratch/djn8068/meg_project
SCRIPT=${PROJECT_DIR}/scripts/decode_bw_single_subject.py
DATA_DIR=${PROJECT_DIR}/data

# =============================================================================
# ENSURE LOG DIRECTORY EXISTS
# =============================================================================

mkdir -p ${PROJECT_DIR}/logs

# =============================================================================
# ACTIVATE CONDA ENVIRONMENT
# =============================================================================

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
source activate /scratch/djn8068/conda_envs/meg_env
export PATH=/scratch/djn8068/conda_envs/meg_env/bin:$PATH
export PYTHONNOUSERSITE=True

echo "Python: $(which python)"
echo "MNE version: $(python -c 'import mne; print(mne.__version__)')"

# =============================================================================
# RUN DECODING
# =============================================================================

python ${SCRIPT} \
    --subject ${SUBJECT} \
    --base_dir ${DATA_DIR} \
    --trial_type localizer \
    --tmin -0.1 \
    --tmax 0.75 \
    --n_splits 5

EXIT_CODE=$?

echo "=============================="
echo "Subject ${SUBJECT} finished with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=============================="

exit $EXIT_CODE