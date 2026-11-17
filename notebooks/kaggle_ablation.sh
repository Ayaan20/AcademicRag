#!/usr/bin/env bash
# Retrieval ablation on a Kaggle GPU notebook (Accelerator: GPU T4, Internet: on).
# Attach a dataset containing scholarrag_part1.zip / scholarrag_part2.zip
# (MinerU-parsed papers + evaluation sets), then run in a cell:
#   !git clone -q https://github.com/Ayaan20/AcademicRag /kaggle/working/AcademicRag
#   !bash /kaggle/working/AcademicRag/notebooks/kaggle_ablation.sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0          # one T4 is plenty; avoids multi-GPU worker pools
cd /kaggle/working/AcademicRag
nvidia-smi -L || echo "no GPU visible"
pip -q install FlagEmbedding faiss-cpu python-dotenv 2>&1 | tail -2 || true

# ---- data: Kaggle may keep the zips or auto-extract them; handle both
mkdir -p parsed_pdf
for z in $(find /kaggle/input -name 'scholarrag_part*.zip' 2>/dev/null); do unzip -q -o "$z" -d .; done
find /kaggle/input -name '*_content_list_v2.json' -exec cp -n {} parsed_pdf/ \; 2>/dev/null || true
for f in evaluation_set_100papers.json train_set_100papers.json; do
  [ -f "$f" ] || find /kaggle/input -name "$f" -exec cp {} . \; 2>/dev/null || true
done
echo "papers: $(ls parsed_pdf/*_content_list_v2.json | wc -l)"; ls -la evaluation_set_100papers.json

# ---- 1. chunk + BGE-M3 dense/sparse embeddings + FAISS
python rebuild_index.py 2>&1 | tr '\r' '\n' | grep -v -E 'it/s\]|s/it\]|B/s' | tail -30

# ---- 2. ablation (4 configs x 204 held-out questions)
python run_ablation.py --dataset evaluation_set_100papers.json > /kaggle/working/ablation_log.txt 2>&1 \
  || { tail -50 /kaggle/working/ablation_log.txt; exit 1; }
grep -E 'Ablation|ERROR|Traceback' /kaggle/working/ablation_log.txt | head -20 || true
cp -r eval_results /kaggle/working/eval_results_out
echo "===== ABLATION RESULTS ====="
cat eval_results/ablation_retrieval.md
cat eval_results/ablation_retrieval.csv
