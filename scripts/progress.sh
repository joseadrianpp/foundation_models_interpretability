#!/usr/bin/env bash
# Progress of a scripts/run_full.sh run: its latest start/end stamps and, for
# every task log, the latest line of each progress bar (one per GPU worker).
#
#   watch -n 30 bash scripts/progress.sh
#
# LOG_DIR as in run_full.sh (default: logs/).
cd "$(dirname "$0")/.."
LOG_DIR=${LOG_DIR:-logs}

tail -n 6 "$LOG_DIR/run_full.log" 2>/dev/null
for f in "$LOG_DIR"/[AB]_*.log; do
  [ -e "$f" ] || continue
  echo "== $f"
  # Bars are keyed by their description (text before ": "); keep the last 4
  grep -a '%|' "$f" | awk -F': ' '
    !($1 in last) { order[++n] = $1 }
    { last[$1] = $0 }
    END { for (i = 1; i <= n; i++) print "  " last[order[i]] }' | tail -n 4
done
