#!/usr/bin/env bash

file="$1"
shift

job_name="$(basename "$file" .py)"

sbatch \
    --job-name="MuZeroSMT_$job_name" \
    --output="%x_%j.out" \
    --error="%x_%j.err" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --mem=128000M \
    --time=1-00:00:00 \
    --wrap="source venv/bin/activate && python -u \"$file\" $*"
