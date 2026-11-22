#!/usr/bin/env bash
# Downloads the CMU Face Images dataset (UCI Machine Learning Repository):
# 20 people x 32 poses/expressions, stored as PGM files at 3 resolutions
# (128x120, 64x60, 32x30). Mitchell, T. (1997), doi:10.24432/C5JC79
set -uo pipefail
cd "$(dirname "$0")"

URL="https://kdd.ics.uci.edu/databases/faces/faces.tar.gz"
ARCHIVE="faces.tar.gz"

if find . -name '*.pgm' -print -quit | grep -q .; then
  echo "PGM images already present in data/, skipping download."
  exit 0
fi

echo "Downloading ${URL} ..."
if command -v wget > /dev/null; then
  wget -q -O "${ARCHIVE}" "${URL}"
elif command -v curl > /dev/null; then
  curl -fsSL -o "${ARCHIVE}" "${URL}"
fi

if [ ! -s "${ARCHIVE}" ]; then
  echo "ERROR: download failed (no internet access?)."
  echo "Download ${URL} manually, put ${ARCHIVE} in the data/ folder,"
  echo "then run this script again."
  exit 1
fi

tar -xzf "${ARCHIVE}"
echo "Extracted $(find . -name '*.pgm' | wc -l) PGM images into data/."
