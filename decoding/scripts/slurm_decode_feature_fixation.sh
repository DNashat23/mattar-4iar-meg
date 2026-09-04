#!/bin/bash
#SBATCH --job-name=decode_feature_fixation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=02:00:00              # 4-class decode, 3 fixation streams (onset/median/offset) per subject
#SBATCH --array=0-44%5               # 45 subjects, max 5 running at once
#SBATCH --output=/scratch/djn8068/meg_project/logs/decode_feature_fixation_%A_%a.out
#SBATCH --error=/scratch/djn8068/meg_project/logs/decode_feature_fixation_%A_%a.err
#SBATCH --mail-type=ARRAY_TASKS_DONE,FAIL
#SBATCH --mail-user=djn8068@nyu.edu
#SBATCH --account=torch_pr_279_tandon_priority

# =============================================================================
# SUBJECT LIST
# =============================================================================
# Same 45-subject fixation-epoch cohort as slurm_ar_v3_independent_fixation.sh
# (48-subject independent-pipeline cohort minus R2652, R2667, R2760 -- no
# *_localizer_fixations.csv for those three).

SUBJECTS=(
    R2662 R2665 R2671 R2675 R2726 R2728 R2312 R2538
    R2611 R2612 R2621 R2649 R2650 R2657 R2663 R2669
    R2672 R2674 R2676 R2678 R2679 R2682 R2683 R2684
    R2688 R2700 R2701 R2702 R2733 R2734 R2743 R2749
    R2763 R2765 R2772 R2775 R2776 R2780 R2781 R2787
    R2792 R2795 R2796 R2737 R2664
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
SCRIPT=${PROJECT_DIR}/scripts/decode_feature_fixation.py
DATA_DIR=${PROJECT_DIR}/data

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

# =============================================================================
# RUN DECODING -- 4-CLASS PROMPT_FEATURE, FIXATION EPOCHS
# =============================================================================
# Script internally loops over all three fixation streams (onset/median/
# offset) for this subject, skipping any stream whose results already exist.

python ${SCRIPT} \
    --subject ${SUBJECT} \
    --base_dir ${DATA_DIR}

EXIT_CODE=$?

echo "=============================="
echo "Subject ${SUBJECT} finished with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=============================="

exit $EXIT_CODE
