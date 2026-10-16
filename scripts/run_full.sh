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
#   DATASETS     datasets, in order (default: A B)
#   STAGES       any of: model ols post (default: all three). model: the model
#                tasks (GPU); ols: every OLS task (CPU, from the model tasks'
#                output); post: the post-hoc analyses (CPU)
#   MODEL_TASKS  model tasks of the model stage (default: all nine, every model)
#   SKIP_DONE    1: model tasks keep the cell types an earlier run finished
#                (--skip-done); IG resumes from its checkpoint either way
#   DISTANCES    GO term similarities for the post-hoc tasks (default: lin SimRel wang)
#   GO3_THREADS  CPU threads of the post-hoc GO3 tasks (default: every core)
#   LOG_DIR      log folder (default: logs/)
#   PROGRESS_INTERVAL  seconds between progress lines in the task logs (default: 30)
#
# Each task logs to $LOG_DIR/<dataset>_<task>.log; follow one with tail -f.
set -u
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"A B"}
STAGES=${STAGES:-"model ols post"}
MODEL_TASKS=${MODEL_TASKS:-"attention_scgpt attention_lora_scgpt ig_lora_scgpt
  attention_geneformer attention_lora_geneformer ig_lora_geneformer
  attention_sccello attention_lora_sccello ig_lora_sccello"}
OLS_TASKS="attention_residualized_ols_scgpt attention_residualized_ols_lora_scgpt ig_residualized_ols_lora_scgpt
  attention_residualized_ols_geneformer attention_residualized_ols_lora_geneformer ig_residualized_ols_lora_geneformer
  attention_residualized_ols_sccello attention_residualized_ols_lora_sccello ig_residualized_ols_lora_sccello"
DISTANCES=${DISTANCES:-"lin SimRel wang"}
LOG_DIR=${LOG_DIR:-logs}
SKIP=""
[ "${SKIP_DONE:-0}" = 1 ] && SKIP="--skip-done"
mkdir -p "$LOG_DIR"
RUN="$PYTHON -u run_tasks.py"
stamp() { echo "$(date '+%F %T') $*" | tee -a "$LOG_DIR/run_full.log"; }
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

if has_stage model && [ -z "${GPUS:-}" ]; then
  N=$(nvidia-smi -L | wc -l)
  N=$(( N > 4 ? 4 : N ))
  GPUS=$(seq -s, 0 $(( N - 1 )))
fi

models() {  # $1 = dataset
  for T in $MODEL_TASKS; do
    stamp "start $1 $T (GPUs $GPUS) -> $LOG_DIR/$1_$T.log"
    $RUN --task $T --dataset $1 --gpus "$GPUS" $SKIP > "$LOG_DIR/$1_$T.log" 2>&1
    stamp "end $1 $T (exit $?)"
  done
}

ols() {  # $1 = dataset
  for T in $OLS_TASKS; do
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

stamp "datasets: $DATASETS; stages: $STAGES"
# The post-hoc tasks of a dataset run in the background, next to the model
# tasks of the next one; two of them never run at once (they share the GO3 cache)
POST_PID=""
for D in $DATASETS; do
  has_stage model && models $D
  has_stage ols && ols $D
  if has_stage post; then
    [ -n "$POST_PID" ] && wait "$POST_PID"
    post $D &
    POST_PID=$!
  fi
done
wait
stamp "all done"
