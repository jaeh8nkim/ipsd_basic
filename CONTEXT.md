# IPSD Basic Context

## Teacher Privileged Context Mechanism

`ipsd_basic/run_ipsd_basic.py` keeps two separate forms of dataset-provided expert text:

- `raw_trace`: the original/generated reasoning trace used for raw-trace baseline scoring.
- `expert_demo`: the stripped final response used as the self-teacher privileged context.

The student prompt only contains the problem. The teacher prompt contains the same problem plus `Expert response:\n{expert_demo}`. The self-teacher therefore gets final-answer/solution destination information, not the hidden reasoning trace.

DeepMath handling:

- DeepMath rows use `r1_solution_1` as the DeepSeek-R1-style source output.
- `deepmath_split_solution(r1_solution_1)` stores the full R1-style text in `raw_trace`.
- If `r1_solution_1` contains `</think>`, only the substring after `</think>` is assigned to `raw_expert`.
- `expert_demo = strip_expert_demo_reasoning(raw_expert)` then removes any remaining `<think>...</think>` spans and strips leftover think markers.
- Result: for DeepMath, the teacher privileged context is the post-`</think>` final response only.

s1k handling:

- s1k rows do not use `r1_solution_1`.
- `raw_trace` is built from `text` when present, otherwise from `trace`, `solution`, or `response`.
- `raw_expert` is selected from `deepseek_attempt`, `expert_demo`, `answer`, or `final_answer`.
- `expert_demo = strip_expert_demo_reasoning(raw_expert)` removes any text before and including `</think>` when think tags are present, and removes any remaining `<think>...</think>` spans.
- Result: for s1k, the teacher privileged context is also the final response only, not the reasoning section.

In both datasets, `build_teacher_prompt(...)` uses `expert_demo`, while generated SFT traces are produced from the student-side problem-only prompt plus the IPSD-selected rollout.

## 2026-06-11 100x5 Pass@4 Runs

Shared run settings:

- Model: `Qwen/Qwen3-8B`
- Thresholds: `1/2`, `3/4`, `7/8`, `15/16`, `31/32`
- Examples per threshold: 100
- Generated IPSD traces per dataset: 500
- Calibration rows: 100
- `max_seq_len`: 16,384
- `max_prompt_len`: 6,144
- `max_gen_len`: 10,240
- `generation_concurrency`: 16
- `max_attempts`: 4
- Pass@4 policy: try up to four rollouts for each query/threshold and stop early once a correct trace is found. If no correct trace is found, keep the fourth attempt as the selected trace.
- No correctness filtering is applied to the final selected traces; correctness is reported.

### s1k 100x5 pass@4

Output directory:

`/work/ipsd/ipsd_basic/outputs/pass4_16k_s1k_100x5`

Dataset:

`simplescaling/s1K-1.1_tokenized`

Runtime:

- 134,036.22 seconds, about 37.23 hours

Raw trace baseline:

- Raw rows scored: 100
- Avg raw surprisal mean: 1.0334
- Avg raw entropy mean: 0.4085
- Raw trace length min / median / mean / max: 1,367 / 9,198.0 / 9,654.7 / 17,908
- Context-truncated raw traces: 10/100

Correctness and generated-trace stats:

| ENS threshold | Calibrated ENS | Correct | Pass@4 | Avg trials | Attempts total | Avg teacher accept | Avg trace length | Cap-hit traces | Mean surprisal | Mean entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1/2` | 0.1477 | 35/100 | 0.35 | 3.15 | 315 | 0.578 | 8877.4 | 39 | 0.1644 | 0.2611 |
| `3/4` | 0.6001 | 37/100 | 0.37 | 3.15 | 315 | 0.878 | 8802.6 | 37 | 0.1450 | 0.2563 |
| `7/8` | 2.4819 | 40/100 | 0.40 | 2.89 | 289 | 0.979 | 7969.1 | 31 | 0.1763 | 0.2622 |
| `15/16` | 7.2034 | 43/100 | 0.43 | 2.78 | 278 | 0.991 | 7120.3 | 29 | 0.2141 | 0.2666 |
| `31/32` | 18.0196 | 49/100 | 0.49 | 2.68 | 268 | 0.996 | 6460.3 | 21 | 0.2363 | 0.2632 |

Attempt distribution:

| ENS threshold | Attempts used distribution | Correct by selected attempt |
|---|---|---|
| `1/2` | 1:22 2:8 3:3 4:67 | 1:22 2:8 3:3 4:2 |
| `3/4` | 1:24 2:6 3:1 4:69 | 1:24 2:6 3:1 4:6 |
| `7/8` | 1:32 2:9 3:3 4:56 | 1:32 2:7 3:0 4:1 |
| `15/16` | 1:39 2:8 3:2 4:51 | 1:39 2:2 3:0 4:2 |
| `31/32` | 1:45 2:11 3:7 4:37 | 1:45 2:2 3:2 4:0 |

Trend notes:

- Correctness increased monotonically with ENS threshold: 35%, 37%, 40%, 43%, 49%.
- Average trials decreased monotonically: 3.15, 3.15, 2.89, 2.78, 2.68.
- Teacher acceptance increased monotonically: 0.578, 0.878, 0.979, 0.991, 0.996.
- Trace length and cap hits fell as the threshold became less restrictive.
- Generated IPSD traces had much lower student surprisal than raw s1k traces, but generated-trace surprisal rose at higher thresholds.

### DeepMath 100x5 pass@4

Output directory:

`/work/ipsd/ipsd_basic/outputs/pass4_16k_deepmath_100x5`

Dataset:

`zwhe99/DeepMath-103K`

Status:

- Completed cleanly.

Runtime:

- 53,499.98 seconds, about 14.86 hours

Raw trace baseline:

- Raw rows scored: 100
- Avg raw surprisal mean: 0.5207
- Avg raw entropy mean: 0.3123
- Raw trace length min / median / mean / max: 894 / 4,520.0 / 5,147.8 / 13,649
- Context-truncated raw traces: 0/100

Correctness and generated-trace stats:

| ENS threshold | Calibrated ENS | Correct | Pass@4 | Avg trials | Attempts total | Avg teacher accept | Avg trace length | Cap-hit traces | Mean surprisal | Mean entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1/2` | 0.1303 | 81/100 | 0.81 | 1.71 | 171 | 0.551 | 6190.1 | 13 | 0.1589 | 0.2523 |
| `3/4` | 0.4240 | 82/100 | 0.82 | 1.76 | 176 | 0.827 | 6345.6 | 8 | 0.1497 | 0.2473 |
| `7/8` | 1.3458 | 87/100 | 0.87 | 1.58 | 158 | 0.954 | 5832.3 | 6 | 0.1409 | 0.2433 |
| `15/16` | 3.7708 | 94/100 | 0.94 | 1.39 | 139 | 0.984 | 4848.4 | 2 | 0.1826 | 0.2494 |
| `31/32` | 9.2774 | 93/100 | 0.93 | 1.49 | 149 | 0.991 | 4188.4 | 1 | 0.2100 | 0.2524 |

Attempt distribution:

| ENS threshold | Attempts used distribution | Correct by selected attempt |
|---|---|---|
| `1/2` | 1:73 2:3 3:4 4:20 | 1:73 2:3 3:4 4:1 |
| `3/4` | 1:69 2:6 3:5 4:20 | 1:69 2:6 3:5 4:2 |
| `7/8` | 1:74 2:8 3:4 4:14 | 1:74 2:8 3:4 4:1 |
| `15/16` | 1:80 2:9 3:3 4:8 | 1:80 2:9 3:3 4:2 |
| `31/32` | 1:78 2:5 3:7 4:10 | 1:78 2:5 3:7 4:3 |

Trend notes:

- Correctness increased from `1/2` through `15/16`, then slightly dipped at `31/32`: 81%, 82%, 87%, 94%, 93%.
- Average trials mostly decreased as the threshold increased: 1.71, 1.76, 1.58, 1.39, 1.49.
- Teacher acceptance increased monotonically: 0.551, 0.827, 0.954, 0.984, 0.991.
- Trace length and cap hits fell as the threshold increased.
- DeepMath was much easier than s1k on this slice under Qwen3-8B pass@4, and the pass@4 gains after attempt 2 were smaller than the s1k case.

## 2026-06-12 Pass@2 Model Sweep

This section tracks the requested 100x5 pass@2 model-selection runs. Shared intended settings unless noted:

- Examples per threshold: 100
- Generated IPSD traces per dataset: 500
- Thresholds: `1/2`, `3/4`, `7/8`, `15/16`, `31/32`
- Calibration rows: 100
- `max_seq_len`: 16,384
- `max_prompt_len`: 6,144
- `max_gen_len`: 10,240
- `max_attempts`: 2
- No correctness filtering is applied to selected traces; correctness is reported.

### s1k Qwen3-0.6B 100x5 pass@2

Output directory:

`/work/ipsd/ipsd_basic/outputs/pass2_16k_s1k_qwen3_0p6b_100x5`

Dataset:

`simplescaling/s1K-1.1_tokenized`

Model:

`Qwen/Qwen3-0.6B`

Runtime:

- Shell runtime: 65,878.58 seconds, about 18.30 hours
- `run_summary.elapsed_seconds`: 65,006.83 seconds, about 18.06 hours

Raw trace baseline:

- Raw rows scored: 100
- Avg raw surprisal mean: 1.0964
- Avg raw entropy mean: 0.6157
- Raw trace length min / median / mean / max: 1,367 / 9,198.0 / 9,654.7 / 17,908
- Context-truncated raw traces: 10/100

Correctness and generated-trace stats:

| ENS threshold | Calibrated ENS | Correct | Pass@2 | Avg trials | Attempts total | Avg teacher accept | Avg trace length | Cap-hit traces | Mean surprisal | Mean entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1/2` | 0.2316 | 9/100 | 0.09 | 1.94 | 194 | 0.573 | 7430.6 | 43 | 0.2824 | 0.4857 |
| `3/4` | 0.7678 | 6/100 | 0.06 | 1.97 | 197 | 0.918 | 7807.7 | 51 | 0.2059 | 0.4169 |
| `7/8` | 2.0428 | 20/100 | 0.20 | 1.88 | 188 | 0.986 | 7045.1 | 32 | 0.2790 | 0.4715 |
| `15/16` | 4.1279 | 21/100 | 0.21 | 1.83 | 183 | 0.995 | 6563.6 | 29 | 0.3011 | 0.4680 |
| `31/32` | 7.6919 | 23/100 | 0.23 | 1.84 | 184 | 0.997 | 6496.8 | 29 | 0.3207 | 0.4707 |

Attempt distribution:

| ENS threshold | Attempts used distribution | Correct by selected attempt |
|---|---|---|
| `1/2` | 1:6 2:94 | 1:6 2:3 |
| `3/4` | 1:3 2:97 | 1:3 2:3 |
| `7/8` | 1:12 2:88 | 1:12 2:8 |
| `15/16` | 1:17 2:83 | 1:17 2:4 |
| `31/32` | 1:16 2:84 | 1:16 2:7 |

Trend notes:

- Qwen3-0.6B was much weaker than Qwen3-8B on the same s1k slice.
- Correctness improved at higher ENS thresholds but remained low: 9%, 6%, 20%, 21%, 23%.
- Most selected traces needed the second attempt, and many still failed after pass@2.
- High teacher acceptance did not imply correctness for 0.6B. At `31/32`, average teacher acceptance was 0.997 but correctness was only 23/100.
- Cap-hit rates were high across all thresholds, especially `1/2` and `3/4`; this explains the long runtime and indicates frequent non-terminating or unproductive reasoning loops.
- For subsequent runs, throughput settings should be made more aggressive than the conservative 16-concurrency setup, while monitoring for OOM or scheduler instability.

## 2026-06-09 16k Smoke Tests

Two 16k smoke tests were completed with Qwen3-8B using `ipsd_basic/run_ipsd_basic.py`.

Shared run settings:

- Model: `Qwen/Qwen3-8B`
- Thresholds: `1/2`, `3/4`, `7/8`, `15/16`, `31/32`
- Examples per threshold: 10
- Generated IPSD traces per dataset: 50
- Calibration rows: 10
- `max_seq_len`: 16,384
- `max_prompt_len`: 6,144
- `max_gen_len`: 10,240
- `generation_concurrency`: 4
- No correctness filtering was applied; incorrect traces are retained and reported.
- The teacher-acceptance trend was monotonic upward for both datasets.
- Correctness was not monotonic across thresholds.

### s1k

Output directory:

`/work/ipsd/ipsd_basic/outputs/smoke_16k_s1k_10x5`

Dataset:

`simplescaling/s1K-1.1_tokenized`

Runtime:

- 5,362.97 seconds, about 89 minutes

Correctness and trace stats:

| ENS threshold | Correct | Accuracy | Avg teacher accept | Avg trace length | Mean surprisal | Mean entropy |
|---|---:|---:|---:|---:|---:|---:|
| `1/2` | 4/10 | 0.40 | 0.579 | 8111.6 | 0.1677 | 0.2668 |
| `3/4` | 6/10 | 0.60 | 0.872 | 8309.9 | 0.1506 | 0.2645 |
| `7/8` | 5/10 | 0.50 | 0.978 | 7467.8 | 0.1861 | 0.2766 |
| `15/16` | 5/10 | 0.50 | 0.991 | 7128.0 | 0.2151 | 0.2765 |
| `31/32` | 5/10 | 0.50 | 0.994 | 4942.5 | 0.2562 | 0.2810 |

Raw trace assessment:

- Raw rows scored: 10
- Avg raw surprisal mean: 1.0292
- Avg raw entropy mean: 0.4202
- One raw trace hit the 16k context limit during scoring.

Saved artifacts:

- `sft_traces.jsonl`: 50 rows
- `token_prob_data.jsonl`: 50 rows
- `raw_trace_stats.jsonl`: 10 rows
- `correctness_report.json`
- `run_summary.json`
- `ipsd_basic_visualization.html`

Notes:

- Automated correctness peaked at `3/4` with 6/10.
- Higher thresholds increased teacher acceptance but did not improve correctness on this 10-example slice.
- Several incorrect traces reached the 10,240 generated-token cap.

### DeepMath

Output directory:

`/work/ipsd/ipsd_basic/outputs/smoke_16k_deepmath_10x5`

Dataset:

`zwhe99/DeepMath-103K`

Runtime:

- 3,652.30 seconds, about 61 minutes

Correctness and trace stats:

| ENS threshold | Correct | Accuracy | Avg teacher accept | Avg trace length | Mean surprisal | Mean entropy |
|---|---:|---:|---:|---:|---:|---:|
| `1/2` | 5/10 | 0.50 | 0.574 | 6133.1 | 0.1379 | 0.2190 |
| `3/4` | 4/10 | 0.40 | 0.828 | 5314.2 | 0.1347 | 0.2165 |
| `7/8` | 4/10 | 0.40 | 0.952 | 5425.8 | 0.1257 | 0.2215 |
| `15/16` | 5/10 | 0.50 | 0.983 | 5024.1 | 0.1595 | 0.2221 |
| `31/32` | 5/10 | 0.50 | 0.991 | 4313.5 | 0.1720 | 0.2081 |

Raw trace assessment:

- Raw rows scored: 10
- Avg raw surprisal mean: 0.4826
- Avg raw entropy mean: 0.2793

Saved artifacts:

- `sft_traces.jsonl`: 50 rows
- `token_prob_data.jsonl`: 50 rows
- `raw_trace_stats.jsonl`: 10 rows
- `correctness_report.json`
- `run_summary.json`
- `ipsd_basic_visualization.html`

Notes:

- Automated correctness was flat/non-monotonic: 50%, 40%, 40%, 50%, 50%.
- Teacher acceptance increased monotonically with threshold.
- DeepMath ran faster than s1k because generated and raw traces were shorter on this selected slice.

## Interpretation

The 16k max sequence length fixed the earlier under-generation issue from the old 2k cap. The resulting correctness is now in a plausible range around 40-60% for these 10-example slices.

The expected monotonic trend holds for teacher acceptance, not for correctness. This is reasonable for a small sample because each threshold changes the generated prefix distribution; higher self-teacher acceptance can shorten traces and reduce student surprise while still producing wrong or incomplete final answers. Correctness should therefore be treated as an empirical metric per threshold, not a guaranteed monotonic function of ENS threshold.

Manual review is still appropriate for failed examples with symbolic/proof-style answers because the current verifier is mostly rule-based and may under-credit equivalent mathematical forms.
