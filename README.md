# IPSD Basic

Minimal IPSD rollout demo for `s1k` examples with Qwen3-8B.

This is intentionally smaller than `../ipsd`: no SQLite DB, no SFT training,
no full eval suite, and no verifier filtering. It preserves the core rollout
mechanism:

1. Load `simplescaling/s1K-1.1_tokenized`.
2. Build a student prompt from the question.
3. Build a self-teacher prompt from the question plus `deepseek_attempt`.
4. Run two vLLM engines with the same model.
5. At each token, accept the teacher token when student ENS is below the
   calibrated fixed threshold; otherwise use the student token.
6. Score the selected raw `s1k` traces under the student and save surprisal,
   entropy, and ENS min/max/mean/median.
7. Save SFT-ready traces, token-level probability data, a correctness report,
   trend audit, and a compact HTML visualization.
8. After all IPSD traces are generated, score those generated traces under the
   student and add posthoc surprisal/entropy/ENS stats plus per-position plots.

Default threshold sweep:

```text
1/2, 3/4, 7/8, 15/16, 31/32
```

Default token budget:

```text
max_seq_len / max_model_len: 16384
max_prompt_len: 6144
max_gen_len: 10240
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 python ipsd_basic/run_ipsd_basic.py \
  --model Qwen/Qwen3-8B \
  --num-examples 10 \
  --calibration-limit 10 \
  --max-seq-len 16384 \
  --output-dir ipsd_basic/outputs/latest \
  --overwrite
```

Outputs:

- `sft_traces.jsonl`: SFT-ready generated traces.
- `token_prob_data.jsonl`: per-trace token probability data used by the HTML.
- `correctness_report.json`: correctness by threshold and by example, with a
  trend audit for the threshold sweep.
- `raw_trace_stats.jsonl`: student-scored raw `s1k` trace arrays and stats.
- `raw_trace_report.json`: compact raw-trace scoring summary.
- `calibration.json`: ENS threshold values.
- `ipsd_basic_visualization.html`: simple trace/token viewer with per-trace
  posthoc surprisal/entropy line plots over token position.

The script reports correctness after generation. It does not discard incorrect
or malformed traces.
