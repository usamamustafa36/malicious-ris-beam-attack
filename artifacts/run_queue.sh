#!/bin/bash
# Runs the remaining revision experiments sequentially once the PHY-baseline run ends.
cd /home/ubuntu/Projects/THESIS/completed/usama/paper2026
while pgrep -f "python3 stage_baselines_phy.py" >/dev/null; do sleep 30; done
export BEAM_DEVICE=cpu OMP_NUM_THREADS=8
python3 stage_tradeoff.py > artifacts/stage_tradeoff.log 2>&1
python3 stage_multipath.py > artifacts/stage_multipath.log 2>&1
python3 stage_rsrp_generality.py > artifacts/stage_rsrp_generality.log 2>&1
echo QUEUE_DONE >> artifacts/stage_rsrp_generality.log
