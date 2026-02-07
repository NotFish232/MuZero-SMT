#!/usr/bin/env bash

file="$1"
shift

job_name="$(basename "$file" .py)"

sbatch \
    --job-name="MuZeroSMT_$job_name" \
    --output="%x_%j.out" \
    --error="%x_%j.err" \
    --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=192 \
    --time=2-00:00:00 \
    --wrap="source venv/bin/activate && python -u \"$file\" $*"
