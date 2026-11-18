#!/bin/bash
cd "$(dirname "$0")"
for s in 0 1 2; do for l in X-LIN X-C0.7 X-C0.5 R0 R1 R2 R3 R4 R5 R6; do echo "$l $s"; done; done | \
  xargs -P 5 -L 1 bash -c '.venv/bin/python run_one.py $0 $1 1200 > ../results/log_$0_$1.txt 2>&1'
echo ALLDONE > ../results/ALLDONE
