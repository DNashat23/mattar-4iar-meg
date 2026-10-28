#!/bin/bash
#SBATCH --job-name=lowhp_localizer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:30:00
#SBATCH --array=0-47%10              # 48 subjects, max 10 running at once
#SBATCH --output=/scratch/djn8068/meg_project/logs/%x_%A_%a.out
#SBATCH --error=/scratch/djn8068/meg_project/logs/%x_%A_%a.err
#SBATCH --account=torch_pr_279_tandon_priority

# =============================================================================
# 0.1 Hz HIGH-PASS REBUILD OF THE LOCALIZER, one array task = one subject.
# STEP is passed at submission:
#   sbatch --export=ALL,STEP=check       -J lowhp_check       slurm_lowhp_localizer.sh
#   sbatch --export=ALL,STEP=reconstruct -J lowhp_reconstruct slurm_lowhp_localizer.sh
#   sbatch --export=ALL,STEP=fixation    -J lowhp_fixation    slurm_lowhp_localizer.sh
#   sbatch --export=ALL,STEP=onset       -J lowhp_onset       slurm_lowhp_localizer.sh
# (add --array=0 to run only the first subject, R2312)
#
#   check        find the _nr runs and the cleaned 1-40 Hz file, choose runs (no filtering; cheap)
#   reconstruct  rebuild the cleaned continuous localizer with a 0.1-40 Hz band (lowhp_reconstruct.py)
#   fixation     fixation-locked epochs from the rebuilt data, reusing the 1-40 Hz AutoReject decisions
#   onset        onset/response epochs from the rebuilt data, reusing the 1-40 Hz AutoReject decisions
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
echo "Step:            $STEP"
echo "Node:            $HOSTNAME"
echo "Start time:      $(date)"
echo "=============================="

# =============================================================================
# PATHS
# =============================================================================

PROJECT_DIR=/scratch/djn8068/meg_project
SCRIPTS=${PROJECT_DIR}/scripts
DATA_DIR=${PROJECT_DIR}/data              # existing 1-40 Hz data and epochs (read only)
SQD_DIR=${PROJECT_DIR}/sqd                # <subject>/<...>_localizer<k>_..._nr.sqd, uploaded from Box
LOWHP_DIR=${PROJECT_DIR}/data_lowhp       # new 0.1-40 Hz data and epochs

mkdir -p ${PROJECT_DIR}/logs ${LOWHP_DIR}

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
# RUN
# =============================================================================

case "$STEP" in
  check)
    python ${SCRIPTS}/lowhp_reconstruct.py --task localizer --subject ${SUBJECT} --base_dir ${DATA_DIR} \
        --raw_root ${SQD_DIR} --out_base ${LOWHP_DIR} --hp 0.1 --check_only ;;
  reconstruct)
    python ${SCRIPTS}/lowhp_reconstruct.py --task localizer --subject ${SUBJECT} --base_dir ${DATA_DIR} \
        --raw_root ${SQD_DIR} --out_base ${LOWHP_DIR} --hp 0.1 ;;
  fixation)
    if ! ls ${LOWHP_DIR}/${SUBJECT}/${SUBJECT}_*_localizer_fixations.csv > /dev/null 2>&1; then
        echo "No fixations CSV for ${SUBJECT}: skipped (as in the 1-40 Hz run)"; exit 0
    fi
    python ${SCRIPTS}/single_subject_ar_v3_independent_fixation.py --subject ${SUBJECT} \
        --base_dir ${LOWHP_DIR} --reuse_ar_dir ${DATA_DIR} ;;
  onset)
    python ${SCRIPTS}/single_subject_ar_v3_independent.py --subject ${SUBJECT} \
        --base_dir ${LOWHP_DIR} --reuse_ar_dir ${DATA_DIR} ;;
  *)
    echo "Unknown STEP '$STEP' (use check, reconstruct, fixation or onset)"; exit 2 ;;
esac

EXIT_CODE=$?

echo "=============================="
echo "Subject ${SUBJECT} step ${STEP} finished with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=============================="

exit $EXIT_CODE
