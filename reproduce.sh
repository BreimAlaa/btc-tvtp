#!/usr/bin/env bash
# Reproduce every number in the paper from scratch. Roughly 45 minutes on a laptop; numba compiles on first use.
set -euo pipefail
cd "$(dirname "$0")"
for d in results/btc results/btc_k3 results/btc_t; do   # a finished walk-forward is replayed from its checkpoints, not re-estimated
  [ -d "$d/checkpoints" ] && echo "note: $d/checkpoints exists and will be replayed; delete it to re-estimate the walk-forward from scratch"
done
python -m pytest -q
python scripts/run_synthetic.py                                    # results/synthetic/
python scripts/run_btc.py                                          # results/btc/      (K=2, Gaussian, main results)
python scripts/run_btc.py --K 3 --skip_fullsample --no_polish --n_restarts 3 --out results/btc_k3   # three-regime robustness
python scripts/run_btc.py --emission student_t --out results/btc_t                                  # Student-t robustness
python scripts/make_paper_tables.py                                # paper/generated/*.tex
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null   # separate lines so that set -e aborts on any LaTeX/BibTeX error
bibtex main >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null
cd ..
echo "done: paper/main.pdf"
