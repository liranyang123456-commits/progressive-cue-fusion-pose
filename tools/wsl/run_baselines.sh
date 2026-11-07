#!/bin/bash
# GoTrack (refine from the T0 bins, from the proposed estimate, from the selection over all hypotheses) and MegaPose
# (RGB-D and RGB refiners from T0 and from the proposed estimate; RGB-D estimator from the box of the visible mask) on
# both BOP samples. Run from the repository root after tools/wsl/setup_envs.sh; BOP_ROOT points to the BOP datasets.
#   REPO (default: current directory), RESULTS (default: $REPO/results)
source "$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONUNBUFFERED=1
REPO=${REPO:-$(pwd)}
RESULTS=${RESULTS:-$REPO/results}
cd "$REPO"
conda activate gotrack
for root in bop6dof_results bop6dof_results_second; do
  for ds in lmo lm tless; do
    python baselines/run_gotrack_bop.py --root "$RESULTS/$root/$ds" --mode refine
    python baselines/run_gotrack_bop.py --root "$RESULTS/$root/$ds" --mode refine-from-ours
    python baselines/run_gotrack_bop.py --root "$RESULTS/$root/$ds" --mode refine-from-ours \
      --ours-file results_msf_select.jsonl --out-name baseline_gotrack_from_msf.jsonl
  done
done
conda deactivate
conda activate happypose
for root in bop6dof_results bop6dof_results_second; do
  for ds in lmo lm tless; do
    python baselines/run_megapose_bop.py --root "$RESULTS/$root/$ds" --mode refine
    python baselines/run_megapose_bop.py --root "$RESULTS/$root/$ds" --mode refine-from-ours
    python baselines/run_megapose_bop.py --root "$RESULTS/$root/$ds" --mode refine --model megapose-1.0-RGB
    python baselines/run_megapose_bop.py --root "$RESULTS/$root/$ds" --mode refine-from-ours --model megapose-1.0-RGB
    python baselines/run_megapose_bop.py --root "$RESULTS/$root/$ds" --mode full
  done
done
