#!/bin/bash
cd "$(dirname "$0")"
for s in 0 1 2; do for l in A2 A3 A4; do echo "$l $s"; done; done | \
  xargs -P 5 -L 1 bash -c '.venv/bin/python run_one.py $0 $1 1200 > ../results/log_$0_$1.txt 2>&1'
echo done > ../results/PESQDONE
