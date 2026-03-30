#!/usr/bin/env bash

file="$1"
shift

job_name="$(basename "$file" .py)"

# Allocate more time for train
if [ "$job_name" = "train" ]; then
    time_req="3-00:00:00"
else
    time_req="1-00:00:00"
fi

sbatch \
    --job-name="MuZeroSMT_$job_name" \
    --output="%x_%j.out" \
    --error="%x_%j.err" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=100G \
    --time="$time_req" \
    --wrap="source venv/bin/activate && python -u \"$file\" $*"
