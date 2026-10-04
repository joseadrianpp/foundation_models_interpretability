#!/usr/bin/env bash
# Full run of datasets A and B: model tasks, OLS tasks and post-hoc analyses.
# The post-hoc tasks of A (CPU) run while B's model tasks use the GPUs.
#
#   nohup bash scripts/run_full.sh > /dev/null 2>&1 &
#   tail -f logs/run_full.log
#
# Environment overrides:
#   PYTHON       interpreter (default: python of the active environment)
#   GPUS         comma-separated GPU ids (default: every visible GPU, at most 4,
#                since --gpus gives each worker whole cell types)
#   DISTANCES    GO term similarities for the post-hoc tasks (default: lin SimRel wang)
#   GO3_THREADS  CPU threads of the post-hoc GO3 tasks (default: every core)
#   LOG_DIR      log folder (default: logs/)
#   PROGRESS_INTERVAL  seconds between progress lines in the task logs (default: 30)
#
# Each task logs to $LOG_DIR/<dataset>_<task>.log; follow one with tail -f.
set -u
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
if [ -z "${GPUS:-}" ]; then
  N=$(nvidia-smi -L | wc -l)
  N=$(( N > 4 ? 4 : N ))
  GPUS=$(seq -s, 0 $(( N - 1 )))
fi
DISTANCES=${DISTANCES:-"lin SimRel wang"}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
RUN="$PYTHON -u run_tasks.py"
stamp() { echo "$(date '+%F %T') $*" | tee -a "$LOG_DIR/run_full.log"; }

models() {  # $1 = dataset
  for T in attention_scgpt attention_geneformer ig_lora_scgpt attention_lora_geneformer; do
    stamp "start $1 $T (GPUs $GPUS) -> $LOG_DIR/$1_$T.log"
    $RUN --task $T --dataset $1 --gpus "$GPUS" > "$LOG_DIR/$1_$T.log" 2>&1
    stamp "end $1 $T (exit $?)"
  done
  for T in attention_residualized_ols_scgpt attention_residualized_ols_geneformer \
           ig_residualized_ols_lora_scgpt attention_residualized_ols_lora_geneformer; do
    stamp "start $1 $T -> $LOG_DIR/$1_$T.log"
    $RUN --task $T --dataset $1 > "$LOG_DIR/$1_$T.log" 2>&1
    stamp "end $1 $T (exit $?)"
  done
}

post() {  # $1 = dataset
  stamp "start $1 postprocess (distances: $DISTANCES) -> $LOG_DIR/$1_postprocess.log"
  $RUN --task postprocess --dataset $1 --distances $DISTANCES > "$LOG_DIR/$1_postprocess.log" 2>&1
  stamp "end $1 postprocess (exit $?)"
}

models A
post A &
models B
wait
post B
stamp "all done"
