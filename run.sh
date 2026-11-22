#!/usr/bin/env bash
# Full pipeline: fetch the dataset (if needed), build, and run.
# Any extra arguments are passed to the program, e.g.:
#   ./run.sh --streams 8 --block 32 --save_all
set -euo pipefail
cd "$(dirname "$0")"

if ! find data -name '*.pgm' -print -quit | grep -q .; then
  bash data/download_data.sh
fi

make build
mkdir -p output
./bin/image_batch_processor --input data --output output --streams 4 \
  --samples 6 "$@" 2>&1 | tee output/log.txt
