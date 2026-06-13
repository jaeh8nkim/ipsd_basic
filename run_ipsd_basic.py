#!/usr/bin/env python3
"""
Minimal IPSD generation and visualization.

Example:

  CUDA_VISIBLE_DEVICES=0,1 python ipsd_basic/run_ipsd_basic.py \
      --model Qwen/Qwen3-8B \
      --num-examples 10 \
      --calibration-limit 10 \
      --max-seq-len 16384 \
      --output-dir ipsd_basic/outputs/latest \
      --overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import html
import json
import math
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "You must respond to every query in the following manner:\n"
    "First, provide a step-by-step logical exploration of the problem.\n"
    "Then, provide a clear and direct response based on your reasoning, "
    "with the final answer enclosed in \\boxed{}."
)

THINK_START = "<think>"
THINK_END = "</think>"
EPS = 1e-6
DEFAULT_MAX_SEQ_LEN = 16_384
DEFAULT_MAX_PROMPT_LEN = 6_144
DEFAULT_MAX_GEN_LEN = DEFAULT_MAX_SEQ_LEN - DEFAULT_MAX_PROMPT_LEN
THRESHOLD_KEYS = ["1/2", "3/4", "7/8", "15/16", "31/32"]
THRESHOLD_QUANTILES = {
    "1/2": 0.5,
    "3/4": 0.75,
    "7/8": 0.875,
    "15/16": 0.9375,
    "31/32": 0.96875,
}


@dataclass
class BasicRow:
    row_index: int
    source_id: str
    question: str
    answer: str
    expert_demo: str
    raw_trace: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal IPSD threshold sweep")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="simplescaling/s1K-1.1_tokenized")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--dataset-offset", type=int, default=0)
    parser.add_argument("--calibration-limit", type=int, default=10)
    parser.add_argument("--max-calibration-tokens", type=int, default=1024)
    parser.add_argument("--thresholds", default=",".join(THRESHOLD_KEYS))
    parser.add_argument("--output-dir", default="ipsd_basic/outputs/latest")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-gpu", default=None)
    parser.add_argument("--student-gpu", default=None)
    parser.add_argument("--teacher-gpu-mem-util", type=float, default=0.45)
    parser.add_argument("--student-gpu-mem-util", type=float, default=0.45)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-gen-len", type=int, default=DEFAULT_MAX_GEN_LEN)
    parser.add_argument("--max-prompt-len", type=int, default=DEFAULT_MAX_PROMPT_LEN)
    parser.add_argument("--max-model-len", "--max-seq-len", dest="max_model_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--logprobs", type=int, default=128)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--allow-missing-answer", action="store_true")
    parser.add_argument("--generation-concurrency", type=int, default=4)
    parser.add_argument("--assessment-concurrency", type=int, default=8)
    parser.add_argument(
        "--assessment-use-both-engines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use both identical model engines for raw/generated student-prompt scoring. "
            "This only speeds assessment; generation still uses separate teacher/student roles."
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--html-max-records", type=int, default=100)
    return parser.parse_args()


def load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except Exception:
        return


def first_nonempty(item: dict[str, Any], names: list[str], default: str = "") -> str:
    for name in names:
        value = item.get(name)
        if value is not None and str(value).strip():
            return str(value)
    return default


def normalize_trace_text(text: str) -> str:
    return (
        (text or "")
        .replace("<|im_start|>think", THINK_START)
        .replace("<|im_start|>answer", THINK_END)
        .replace("<|im_end|>", "")
        .strip()
    )


def ensure_trace_wrapped(text: str) -> str:
    text = normalize_trace_text(text)
    if not text:
        return ""
    if THINK_START not in text:
        text = f"{THINK_START}\n{text}"
    if THINK_END not in text:
        text = f"{text}\n{THINK_END}\n"
    return text


def extract_trace_from_text(text: str) -> str:
    text = normalize_trace_text(text)
    pos = text.find(THINK_START)
    if pos >= 0:
        return text[pos:].strip()
    return ensure_trace_wrapped(text)


def extract_answer_part(completion: str) -> str:
    if THINK_END in completion:
        return completion.split(THINK_END, 1)[1].strip()
    return completion.strip()


def extract_boxed_answer(text: str) -> str | None:
    matches: list[str] = []
    i = 0
    while i < len(text):
        start = text.find("\\boxed{", i)
        if start < 0:
            break
        j = start + len("\\boxed{")
        depth = 1
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            matches.append(text[start + len("\\boxed{") : j - 1])
        i = start + 1
    return matches[-1].strip() if matches else None


def strip_expert_demo_reasoning(text: str) -> str:
    text = normalize_trace_text(text)
    if not text:
        return ""
    if THINK_END in text:
        text = text.split(THINK_END, 1)[1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if THINK_START in text:
        text = text.split(THINK_START, 1)[0]
    return text.replace(THINK_END, "").strip()


def deepmath_split_solution(r1_solution: str) -> tuple[str, str]:
    raw = (r1_solution or "").strip()
    trace = raw if raw.startswith(THINK_START) else f"{THINK_START}{raw}"
    expert_demo = ""
    if THINK_END in raw:
        expert_demo = raw.split(THINK_END, 1)[1].strip()
    if not expert_demo:
        expert_demo = extract_answer_part(trace)
    return trace, expert_demo


def build_student_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_teacher_prompt(tokenizer, question: str, expert_demo: str) -> str:
    teacher_system = (
        DEFAULT_SYSTEM_PROMPT
        + "\n\nYou are also given an expert response to the same problem. "
        "Use it only as privileged destination information while producing your own reasoning."
    )
    user = (
        f"Problem:\n{question}\n\n"
        f"Expert response:\n{expert_demo.strip()}\n\n"
        "Solve the problem. Think step by step, then give the final answer."
    )
    messages = [
        {"role": "system", "content": teacher_system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_training_text(tokenizer, question: str, trace: str) -> str:
    prompt = build_student_prompt(tokenizer, question)
    prompt = re.sub(r"(<think>\s*</think>\s*)$", "", prompt)
    return prompt + trace


def normalize_example(item: dict[str, Any], index: int) -> BasicRow | None:
    source_id = first_nonempty(item, ["id", "source_id", "uid"], default=str(index))
    question = first_nonempty(item, ["question", "problem", "prompt"])
    answer = first_nonempty(item, ["answer", "final_answer", "solution", "target"])
    if item.get("r1_solution_1"):
        raw_trace, raw_expert = deepmath_split_solution(first_nonempty(item, ["r1_solution_1"]))
        answer = first_nonempty(item, ["final_answer", "answer", "target"])
    else:
        raw_trace = extract_trace_from_text(str(item["text"])) if item.get("text") else ensure_trace_wrapped(
            first_nonempty(item, ["trace", "solution", "response"])
        )
        raw_expert = first_nonempty(item, ["deepseek_attempt", "expert_demo", "answer", "final_answer"])
    expert_demo = strip_expert_demo_reasoning(raw_expert)
    if not answer:
        answer = extract_boxed_answer(expert_demo) or extract_boxed_answer(extract_answer_part(raw_trace)) or ""
    if not question or not expert_demo:
        return None
    return BasicRow(
        row_index=index,
        source_id=str(source_id),
        question=question,
        answer=answer,
        expert_demo=expert_demo,
        raw_trace=raw_trace,
    )


def load_rows(args: argparse.Namespace) -> list[BasicRow]:
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.dataset_split, trust_remote_code=True)
    rows: list[BasicRow] = []
    i = max(0, args.dataset_offset)
    while i < len(ds) and len(rows) < max(args.num_examples, args.calibration_limit):
        row = normalize_example(dict(ds[i]), i)
        if row is not None:
            rows.append(row)
        i += 1
    if len(rows) < args.num_examples:
        raise RuntimeError(f"Only found {len(rows)} usable rows; requested {args.num_examples}")
    if not args.allow_missing_answer:
        missing = [row.source_id for row in rows[: args.num_examples] if not row.answer]
        if missing:
            raise RuntimeError(
                "Some selected rows have no extractable answer for correctness reporting: "
                + ", ".join(missing[:5])
                + ". Pass --allow-missing-answer to keep them."
            )
    return rows


@contextlib.contextmanager
def visible_gpus(devices: str | None):
    if devices is None:
        yield
        return
    original = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    try:
        yield
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = original


def default_gpus(args: argparse.Namespace) -> tuple[str, str]:
    if args.teacher_gpu and args.student_gpu:
        return args.teacher_gpu, args.student_gpu
    visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",") if x.strip()]
    if len(visible) < 2:
        raise ValueError(
            "Two vLLM engines need two visible GPU entries. Set CUDA_VISIBLE_DEVICES=0,1 "
            "or pass --teacher-gpu and --student-gpu."
        )
    return args.teacher_gpu or visible[0], args.student_gpu or visible[1]


async def setup_async_engine(
    model_path: str,
    gpu: str | None,
    max_model_len: int,
    gpu_memory_utilization: float,
    dtype: str,
    max_logprobs: int,
):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    resolved = model_path if os.path.exists(model_path) else snapshot_download(model_path)
    with visible_gpus(gpu):
        engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=resolved,
                tensor_parallel_size=1,
                max_model_len=max_model_len,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                enable_prefix_caching=True,
                max_logprobs=max_logprobs,
            ),
            start_engine_loop=True,
        )
    tokenizer = engine.get_tokenizer()
    if hasattr(tokenizer, "__await__"):
        tokenizer = await tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    return engine, tokenizer, resolved


async def one_step(engine, context_ids: list[int], sampling_params):
    from vllm import TokensPrompt

    generator = engine.generate(
        TokensPrompt(prompt_token_ids=context_ids),
        sampling_params,
        request_id=str(uuid.uuid4()),
    )
    first = await anext(generator)
    return first.outputs[0]


def completion_token_ids(output) -> list[int]:
    token_ids = getattr(output, "token_ids", None)
    return list(token_ids or [])


def logprob_value(obj: Any) -> float:
    return float(getattr(obj, "logprob", obj))


def entropy_from_logprobs(logprobs_dict: dict[int, Any], vocab_size: int | None) -> float:
    logps: list[float] = []
    for obj in logprobs_dict.values():
        lp = logprob_value(obj)
        if math.isfinite(lp):
            logps.append(lp)
    if not logps:
        return 0.0
    probs = [math.exp(lp) for lp in logps]
    entropy = -sum(p * lp for p, lp in zip(probs, logps))
    if vocab_size and vocab_size > len(probs):
        tail = max(0.0, 1.0 - sum(probs))
        if tail > 0.0:
            tail_p = tail / (vocab_size - len(probs))
            entropy += -tail * math.log(tail_p)
    return float(entropy)


def token_surprisal_from_logprobs(token_id: int, logprobs_dict: dict[int, Any], vocab_size: int | None) -> float:
    if token_id in logprobs_dict:
        return float(-logprob_value(logprobs_dict[token_id]))
    if vocab_size and vocab_size > len(logprobs_dict):
        probs = []
        for obj in logprobs_dict.values():
            lp = logprob_value(obj)
            if math.isfinite(lp):
                probs.append(math.exp(lp))
        tail = max(0.0, 1.0 - sum(probs))
        if tail > 0.0:
            return float(-math.log(tail / (vocab_size - len(logprobs_dict))))
    return float("inf")


def entropy_and_tail_surprisal(logprobs_dict: dict[int, Any], vocab_size: int | None) -> tuple[float, float | None]:
    logps: list[float] = []
    for obj in logprobs_dict.values():
        lp = logprob_value(obj)
        if math.isfinite(lp):
            logps.append(lp)
    probs = [math.exp(lp) for lp in logps]
    entropy = -sum(p * lp for p, lp in zip(probs, logps)) if logps else 0.0
    tail = max(0.0, 1.0 - sum(probs))
    if vocab_size and vocab_size > len(probs) and tail > 0.0:
        tail_p = tail / (vocab_size - len(probs))
        entropy += -tail * math.log(tail_p)

    tail_surprisal = None
    if vocab_size and vocab_size > len(logprobs_dict) and tail > 0.0:
        tail_surprisal = float(-math.log(tail / (vocab_size - len(logprobs_dict))))
    return float(entropy), tail_surprisal


def token_surprisal_with_tail(
    token_id: int,
    logprobs_dict: dict[int, Any],
    tail_surprisal: float | None,
) -> float:
    if token_id in logprobs_dict:
        return float(-logprob_value(logprobs_dict[token_id]))
    if tail_surprisal is not None:
        return tail_surprisal
    return float("inf")


def finite_surprisal(value: float) -> float:
    return float(value) if math.isfinite(value) else 1e9


def metrics_from_logprobs(token_id: int, logprobs_dict: dict[int, Any], vocab_size: int) -> tuple[float, float, float]:
    entropy, tail_surprisal = entropy_and_tail_surprisal(logprobs_dict, vocab_size)
    surprisal = finite_surprisal(token_surprisal_with_tail(token_id, logprobs_dict, tail_surprisal))
    ens = float(surprisal / (entropy + EPS))
    return float(surprisal), float(entropy), ens


def quantile_thresholds(values: list[float]) -> dict[str, float]:
    finite = np.array([v for v in values if math.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        raise RuntimeError("No finite ENS values collected for calibration")
    return {key: float(np.quantile(finite, q)) for key, q in THRESHOLD_QUANTILES.items()}


async def score_prompt_logprobs(
    engine,
    prompt_ids: list[int],
    completion_ids: list[int],
    sampling_params,
    vocab_size: int,
) -> tuple[list[float], list[float], list[float]] | None:
    if not completion_ids:
        return [], [], []
    try:
        from vllm import TokensPrompt

        full_ids = list(prompt_ids) + list(completion_ids)
        generator = engine.generate(
            TokensPrompt(prompt_token_ids=full_ids),
            sampling_params,
            request_id=str(uuid.uuid4()),
        )
        final_output = None
        async for output in generator:
            final_output = output
            if output.finished:
                break
        prompt_logprobs = getattr(final_output, "prompt_logprobs", None) if final_output is not None else None
        if not prompt_logprobs:
            return None

        start = len(prompt_ids)
        surprisals: list[float] = []
        entropies: list[float] = []
        ens_values: list[float] = []
        saw_completion_logprobs = False
        for offset, target_id in enumerate(completion_ids):
            pos = start + offset
            lp_dict = prompt_logprobs[pos] if pos < len(prompt_logprobs) and prompt_logprobs[pos] else {}
            saw_completion_logprobs = saw_completion_logprobs or bool(lp_dict)
            surprisal, entropy, ens = metrics_from_logprobs(int(target_id), lp_dict, vocab_size)
            surprisals.append(surprisal)
            entropies.append(entropy)
            ens_values.append(ens)
        if not saw_completion_logprobs:
            return None
        return surprisals, entropies, ens_values
    except Exception:
        return None


async def score_one_step_loop(
    engine,
    prompt_ids: list[int],
    completion_ids: list[int],
    sampling_params,
    vocab_size: int,
) -> tuple[list[float], list[float], list[float]]:
    context_ids = list(prompt_ids)
    surprisals: list[float] = []
    entropies: list[float] = []
    ens_values: list[float] = []
    for target_id in completion_ids:
        output = await one_step(engine, context_ids, sampling_params)
        lp_dict = output.logprobs[0] if output.logprobs else {}
        surprisal, entropy, ens = metrics_from_logprobs(int(target_id), lp_dict, vocab_size)
        surprisals.append(surprisal)
        entropies.append(entropy)
        ens_values.append(ens)
        context_ids.append(int(target_id))
    return surprisals, entropies, ens_values


async def calibrate_thresholds(rows: list[BasicRow], student_engine, tokenizer, args: argparse.Namespace) -> dict[str, float]:
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    step_params = SamplingParams(
        max_tokens=1,
        temperature=args.temperature,
        top_p=args.top_p,
        logprobs=args.logprobs,
        output_kind=RequestOutputKind.DELTA,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    prompt_params = SamplingParams(
        max_tokens=1,
        temperature=args.temperature,
        top_p=args.top_p,
        logprobs=args.logprobs,
        prompt_logprobs=args.logprobs,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    all_ens: list[float] = []
    for idx, row in enumerate(rows[: args.calibration_limit], start=1):
        prompt_ids = tokenizer.encode(build_student_prompt(tokenizer, row.question), add_special_tokens=False)
        completion_ids = tokenizer.encode(row.raw_trace, add_special_tokens=False)[: args.max_calibration_tokens]
        metrics = await score_prompt_logprobs(student_engine, prompt_ids, completion_ids, prompt_params, len(tokenizer))
        path = "prompt_logprobs"
        if metrics is None:
            metrics = await score_one_step_loop(student_engine, prompt_ids, completion_ids, step_params, len(tokenizer))
            path = "one_step_loop"
        all_ens.extend(v for v in metrics[2] if math.isfinite(v))
        print(
            f"calibration {idx}/{min(args.calibration_limit, len(rows))}: "
            f"{len(completion_ids)} tokens via {path}",
            flush=True,
        )
    thresholds = quantile_thresholds(all_ens)
    print("calibrated ENS thresholds: " + json.dumps(thresholds, sort_keys=True), flush=True)
    return thresholds


def cheap_equivalence(gold: str, answer: str) -> bool:
    if not gold:
        return False
    boxed = extract_boxed_answer(answer) or answer

    def norm(s: str) -> str:
        return re.sub(r"\s+", "", str(s)).strip().lower()

    return norm(gold) == norm(boxed)


def accuracy_reward_check(gold: str, answer_part: str) -> bool:
    if not gold:
        return False
    try:
        from trl.rewards import accuracy_reward

        reward = accuracy_reward(completions=[[{"content": answer_part}]], solution=[gold])[0]
        return reward is not None and reward > 0
    except Exception:
        return cheap_equivalence(gold, answer_part)


def grade_completion(gold: str, completion: str) -> dict[str, Any]:
    answer_part = extract_answer_part(completion)
    extracted = extract_boxed_answer(answer_part) or answer_part
    if not gold:
        return {"correct": None, "grader": "missing_gold", "extracted_answer": extracted}
    if accuracy_reward_check(gold, answer_part):
        return {"correct": True, "grader": "accuracy_reward", "extracted_answer": extracted}
    if cheap_equivalence(gold, answer_part):
        return {"correct": True, "grader": "cheap_equivalence", "extracted_answer": extracted}
    return {"correct": False, "grader": "failed", "extracted_answer": extracted}


def has_complete_boxed_answer(completion: str) -> bool:
    if THINK_END not in completion:
        return False
    answer_part = extract_answer_part(completion)
    return extract_boxed_answer(answer_part) is not None


def finite_mean(values: list[float]) -> float | None:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def finite_stats(values: list[float]) -> dict[str, float | None]:
    finite = np.array(
        [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
    }


def active_max_model_len(args: argparse.Namespace) -> int:
    return int(args.max_model_len or (args.max_prompt_len + args.max_gen_len))


async def score_trace_under_student(
    row: BasicRow,
    token_ids: list[int],
    student_engine,
    tokenizer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from vllm.sampling_params import SamplingParams

    if not token_ids:
        return {
            "path": "empty",
            "surprisal": [],
            "entropy": [],
            "ens": [],
            "surprisal_stats": finite_stats([]),
            "entropy_stats": finite_stats([]),
            "ens_stats": finite_stats([]),
            "total_tokens": 0,
            "scored_tokens": 0,
            "unscored_tokens": 0,
            "truncated_by_context": False,
        }

    prompt_ids = tokenizer.encode(build_student_prompt(tokenizer, row.question), add_special_tokens=False)
    max_context_len = active_max_model_len(args)
    max_scored_tokens = max(0, max_context_len - len(prompt_ids) - 1)
    scored_token_ids = token_ids[:max_scored_tokens]
    truncated_by_context = len(scored_token_ids) < len(token_ids)
    if not scored_token_ids:
        return {
            "path": "context_too_short",
            "surprisal": [],
            "entropy": [],
            "ens": [],
            "surprisal_stats": finite_stats([]),
            "entropy_stats": finite_stats([]),
            "ens_stats": finite_stats([]),
            "total_tokens": len(token_ids),
            "scored_tokens": 0,
            "unscored_tokens": len(token_ids),
            "truncated_by_context": truncated_by_context,
        }
    prompt_params = SamplingParams(
        max_tokens=1,
        temperature=args.temperature,
        top_p=args.top_p,
        logprobs=args.logprobs,
        prompt_logprobs=args.logprobs,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    step_params = SamplingParams(
        max_tokens=1,
        temperature=args.temperature,
        top_p=args.top_p,
        logprobs=args.logprobs,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    metrics = await score_prompt_logprobs(student_engine, prompt_ids, scored_token_ids, prompt_params, len(tokenizer))
    path = "prompt_logprobs"
    if metrics is None:
        metrics = await score_one_step_loop(student_engine, prompt_ids, scored_token_ids, step_params, len(tokenizer))
        path = "one_step_loop"
    surprisals, entropies, ens_values = metrics
    return {
        "path": path,
        "surprisal": surprisals,
        "entropy": entropies,
        "ens": ens_values,
        "surprisal_stats": finite_stats(surprisals),
        "entropy_stats": finite_stats(entropies),
        "ens_stats": finite_stats(ens_values),
        "total_tokens": len(token_ids),
        "scored_tokens": len(scored_token_ids),
        "unscored_tokens": len(token_ids) - len(scored_token_ids),
        "truncated_by_context": truncated_by_context,
    }


async def generate_ipsd_trace(
    row: BasicRow,
    threshold_key: str,
    threshold_value: float,
    teacher_engine,
    student_engine,
    tokenizer,
    args: argparse.Namespace,
    attempt_index: int = 1,
) -> dict[str, Any]:
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    teacher_prompt = build_teacher_prompt(tokenizer, row.question, row.expert_demo)
    student_prompt = build_student_prompt(tokenizer, row.question)
    teacher_context = tokenizer.encode(teacher_prompt, add_special_tokens=False)
    student_context = tokenizer.encode(student_prompt, add_special_tokens=False)
    if len(teacher_context) > args.max_prompt_len:
        raise RuntimeError(f"teacher prompt over cap for source_id={row.source_id}: {len(teacher_context)}")
    if len(student_context) > args.max_prompt_len:
        raise RuntimeError(f"student prompt over cap for source_id={row.source_id}: {len(student_context)}")

    max_context_len = active_max_model_len(args)
    vocab_size = len(tokenizer)
    chosen_ids: list[int] = []
    tokens: list[dict[str, Any]] = []
    source_mask: list[int] = []
    surprisals: list[float] = []
    entropies: list[float] = []
    ens_values: list[float] = []

    base_seed = (
        args.seed
        + row.row_index * 1_000_000
        + THRESHOLD_KEYS.index(threshold_key) * 100_000
        + (attempt_index - 1) * 10_000_000
    )
    stopped_by_context = False
    for step in range(args.max_gen_len):
        if len(teacher_context) >= max_context_len - 1 or len(student_context) >= max_context_len - 1:
            stopped_by_context = True
            break
        teacher_params = SamplingParams(
            max_tokens=1,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=base_seed + step * 2,
            logprobs=args.logprobs,
            output_kind=RequestOutputKind.DELTA,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )
        student_params = SamplingParams(
            max_tokens=1,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=base_seed + step * 2 + 1,
            logprobs=args.logprobs,
            output_kind=RequestOutputKind.DELTA,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )
        teacher_out, student_out = await asyncio.gather(
            one_step(teacher_engine, teacher_context, teacher_params),
            one_step(student_engine, student_context, student_params),
        )
        teacher_ids = completion_token_ids(teacher_out)
        student_ids = completion_token_ids(student_out)
        if not teacher_ids or not student_ids:
            break
        teacher_id = int(teacher_ids[-1])
        student_id = int(student_ids[-1])
        student_lp = student_out.logprobs[0] if student_out.logprobs else {}
        teacher_lp = teacher_out.logprobs[0] if teacher_out.logprobs else {}

        student_entropy, student_tail_surprisal = entropy_and_tail_surprisal(student_lp, vocab_size)
        _teacher_entropy, teacher_tail_surprisal = entropy_and_tail_surprisal(teacher_lp, vocab_size)
        teacher_under_student_surprisal = finite_surprisal(
            token_surprisal_with_tail(teacher_id, student_lp, student_tail_surprisal)
        )
        teacher_under_student_entropy = student_entropy
        teacher_under_student_ens = float(teacher_under_student_surprisal / (student_entropy + EPS))
        accepted = teacher_under_student_ens <= threshold_value
        chosen_id = teacher_id if accepted else student_id
        source = "teacher" if accepted else "student"
        if accepted:
            chosen_student_surprisal = teacher_under_student_surprisal
            chosen_student_entropy = teacher_under_student_entropy
            chosen_student_ens = teacher_under_student_ens
        else:
            chosen_student_surprisal = finite_surprisal(
                token_surprisal_with_tail(chosen_id, student_lp, student_tail_surprisal)
            )
            chosen_student_entropy = student_entropy
            chosen_student_ens = float(chosen_student_surprisal / (student_entropy + EPS))
        chosen_teacher_surprisal = finite_surprisal(
            token_surprisal_with_tail(chosen_id, teacher_lp, teacher_tail_surprisal)
        )
        teacher_candidate_teacher_surprisal = finite_surprisal(
            token_surprisal_with_tail(teacher_id, teacher_lp, teacher_tail_surprisal)
        )
        student_candidate_student_surprisal = finite_surprisal(
            token_surprisal_with_tail(student_id, student_lp, student_tail_surprisal)
        )

        chosen_token = tokenizer.decode([chosen_id], skip_special_tokens=False)
        teacher_token = tokenizer.decode([teacher_id], skip_special_tokens=False)
        student_token = tokenizer.decode([student_id], skip_special_tokens=False)
        student_logprob = -chosen_student_surprisal if math.isfinite(chosen_student_surprisal) else None
        teacher_logprob = -chosen_teacher_surprisal if math.isfinite(chosen_teacher_surprisal) else None
        logprob_delta = (
            float(teacher_logprob - student_logprob)
            if teacher_logprob is not None and student_logprob is not None
            else None
        )
        record = {
            "position": step,
            "token_id": chosen_id,
            "token": chosen_token,
            "source": source,
            "accepted_teacher": bool(accepted),
            "teacher_token_id": teacher_id,
            "teacher_token": teacher_token,
            "student_token_id": student_id,
            "student_token": student_token,
            "student_surprisal": chosen_student_surprisal,
            "student_entropy": chosen_student_entropy,
            "student_ens": chosen_student_ens,
            "teacher_token_student_ens": teacher_under_student_ens,
            "ens_threshold": threshold_value,
            "ens_margin": threshold_value - teacher_under_student_ens,
            "student_logprob_chosen": student_logprob,
            "teacher_logprob_chosen": teacher_logprob,
            "teacher_minus_student_logprob": logprob_delta,
            "teacher_candidate_teacher_logprob": -teacher_candidate_teacher_surprisal,
            "student_candidate_student_logprob": -student_candidate_student_surprisal,
        }
        tokens.append(record)
        chosen_ids.append(chosen_id)
        source_mask.append(1 if accepted else 0)
        surprisals.append(chosen_student_surprisal)
        entropies.append(chosen_student_entropy)
        ens_values.append(chosen_student_ens)
        teacher_context.append(chosen_id)
        student_context.append(chosen_id)

        if chosen_id == tokenizer.eos_token_id:
            break
        if step % 16 == 0 or chosen_token in {"}", "\n", "."}:
            partial_trace = tokenizer.decode(chosen_ids, skip_special_tokens=False)
            if has_complete_boxed_answer(partial_trace):
                break

    trace = tokenizer.decode(chosen_ids, skip_special_tokens=False)
    grade = grade_completion(row.answer, trace)
    teacher_tokens = sum(source_mask)
    trace_perplexity = None
    mean_surprisal = finite_mean(surprisals)
    if mean_surprisal is not None:
        trace_perplexity = float(math.exp(mean_surprisal))
    return {
        "threshold_key": threshold_key,
        "threshold_value": threshold_value,
        "attempt_index": attempt_index,
        "row_index": row.row_index,
        "source_id": row.source_id,
        "question": row.question,
        "answer": row.answer,
        "expert_demo": row.expert_demo,
        "trace": trace,
        "text": build_training_text(tokenizer, row.question, trace),
        "correct": grade["correct"],
        "grader": grade["grader"],
        "extracted_answer": grade["extracted_answer"],
        "trace_length": len(chosen_ids),
        "teacher_tokens": teacher_tokens,
        "student_tokens": len(source_mask) - teacher_tokens,
        "teacher_accept_rate": teacher_tokens / len(source_mask) if source_mask else 0.0,
        "trace_perplexity": trace_perplexity,
        "has_think_end": THINK_END in trace,
        "stopped_by_context": stopped_by_context,
        "max_model_len": max_context_len,
        "source_mask": source_mask,
        "token_ids": chosen_ids,
        "token_surprisal": surprisals,
        "token_entropy": entropies,
        "token_ens": ens_values,
        "tokens": tokens,
    }


def known_output_paths(output_dir: Path) -> list[Path]:
    return [
        output_dir / "sft_traces.jsonl",
        output_dir / "token_prob_data.jsonl",
        output_dir / "correctness_report.json",
        output_dir / "raw_trace_stats.jsonl",
        output_dir / "raw_trace_report.json",
        output_dir / "calibration.json",
        output_dir / "run_summary.json",
        output_dir / "ipsd_basic_visualization.html",
    ]


def prepare_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in known_output_paths(output_dir):
            if path.exists():
                path.unlink()
    return output_dir


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


async def assess_raw_traces(
    rows: list[BasicRow],
    student_engine,
    teacher_engine,
    tokenizer,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    raw_records: list[dict[str, Any] | None] = [None] * len(rows)
    score_engines = [student_engine]
    if args.assessment_use_both_engines:
        score_engines.append(teacher_engine)
    sem = asyncio.Semaphore(max(1, int(args.assessment_concurrency)))

    async def score_row(idx: int, row: BasicRow) -> tuple[int, dict[str, Any]]:
        async with sem:
            token_ids = tokenizer.encode(row.raw_trace, add_special_tokens=False)
            engine = score_engines[(idx - 1) % len(score_engines)]
            print(
                f"raw assessment start {idx}/{len(rows)}: "
                f"row_index={row.row_index} source_id={row.source_id} tokens={len(token_ids)}",
                flush=True,
            )
            score = await score_trace_under_student(row, token_ids, engine, tokenizer, args)
        record = {
            "row_index": row.row_index,
            "source_id": row.source_id,
            "question": row.question,
            "answer": row.answer,
            "trace_length": len(token_ids),
            "scoring_path": score["path"],
            "surprisal": score["surprisal"],
            "entropy": score["entropy"],
            "ens": score["ens"],
            "surprisal_stats": score["surprisal_stats"],
            "entropy_stats": score["entropy_stats"],
            "ens_stats": score["ens_stats"],
            "scored_tokens": score["scored_tokens"],
            "unscored_tokens": score["unscored_tokens"],
            "truncated_by_context": score["truncated_by_context"],
        }
        return idx, record

    done = 0
    tasks = [score_row(idx, row) for idx, row in enumerate(rows, start=1)]
    for task in asyncio.as_completed(tasks):
        idx, record = await task
        raw_records[idx - 1] = record
        done += 1
        print(
            f"completed raw assessment {done}/{len(rows)}: "
            f"row_index={record['row_index']} source_id={record['source_id']} "
            f"tokens={record['trace_length']}",
            flush=True,
        )
    ordered_records = [record for record in raw_records if record is not None]
    for record in ordered_records:
        append_jsonl(output_dir / "raw_trace_stats.jsonl", record)
    report = {
        "rows": len(ordered_records),
        "avg_surprisal_mean": float(np.mean([r["surprisal_stats"]["mean"] for r in ordered_records if r["surprisal_stats"]["mean"] is not None])) if ordered_records else None,
        "avg_entropy_mean": float(np.mean([r["entropy_stats"]["mean"] for r in ordered_records if r["entropy_stats"]["mean"] is not None])) if ordered_records else None,
        "examples": [
            {
                "row_index": r["row_index"],
                "source_id": r["source_id"],
                "trace_length": r["trace_length"],
                "surprisal_stats": r["surprisal_stats"],
                "entropy_stats": r["entropy_stats"],
                "ens_stats": r["ens_stats"],
                "scored_tokens": r["scored_tokens"],
                "unscored_tokens": r["unscored_tokens"],
                "truncated_by_context": r["truncated_by_context"],
            }
            for r in ordered_records
        ],
    }
    write_json(output_dir / "raw_trace_report.json", report)
    return ordered_records


async def assess_generated_traces(
    results: list[dict[str, Any]],
    row_by_index: dict[int, BasicRow],
    student_engine,
    teacher_engine,
    tokenizer,
    args: argparse.Namespace,
) -> None:
    score_engines = [student_engine]
    if args.assessment_use_both_engines:
        score_engines.append(teacher_engine)
    sem = asyncio.Semaphore(max(1, int(args.assessment_concurrency)))

    async def score_result(idx: int, result: dict[str, Any]) -> int:
        async with sem:
            row = row_by_index[result["row_index"]]
            engine = score_engines[(idx - 1) % len(score_engines)]
            print(
                f"generated assessment start {idx}/{len(results)}: threshold={result['threshold_key']} "
                f"row_index={row.row_index} source_id={row.source_id} tokens={len(result['token_ids'])}",
                flush=True,
            )
            posthoc = await score_trace_under_student(row, result["token_ids"], engine, tokenizer, args)
        for pos, token_record in enumerate(result["tokens"]):
            if pos < len(posthoc["surprisal"]):
                token_record["posthoc_student_surprisal"] = posthoc["surprisal"][pos]
            if pos < len(posthoc["entropy"]):
                token_record["posthoc_student_entropy"] = posthoc["entropy"][pos]
            if pos < len(posthoc["ens"]):
                token_record["posthoc_student_ens"] = posthoc["ens"][pos]
        result["posthoc_student_surprisal"] = posthoc["surprisal"]
        result["posthoc_student_entropy"] = posthoc["entropy"]
        result["posthoc_student_ens"] = posthoc["ens"]
        result["posthoc_student_stats"] = {
            "scoring_path": posthoc["path"],
            "surprisal": posthoc["surprisal_stats"],
            "entropy": posthoc["entropy_stats"],
            "ens": posthoc["ens_stats"],
            "total_tokens": posthoc["total_tokens"],
            "scored_tokens": posthoc["scored_tokens"],
            "unscored_tokens": posthoc["unscored_tokens"],
            "truncated_by_context": posthoc["truncated_by_context"],
        }
        return idx

    done = 0
    tasks = [score_result(idx, result) for idx, result in enumerate(results, start=1)]
    for task in asyncio.as_completed(tasks):
        idx = await task
        done += 1
        print(f"completed generated assessment {done}/{len(results)}: result_index={idx}", flush=True)


def compact_sft_record(result: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "threshold_key",
        "threshold_value",
        "row_index",
        "source_id",
        "attempt_index",
        "attempts_used",
        "max_attempts",
        "pass_at_k_correct",
        "selected_attempt",
        "attempts",
        "question",
        "answer",
        "trace",
        "text",
        "correct",
        "grader",
        "extracted_answer",
        "trace_length",
        "teacher_tokens",
        "student_tokens",
        "teacher_accept_rate",
        "trace_perplexity",
        "has_think_end",
        "posthoc_student_stats",
    ]
    return {key: result.get(key) for key in keys}


def token_prob_record(result: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "threshold_key",
        "threshold_value",
        "row_index",
        "source_id",
        "attempt_index",
        "attempts_used",
        "max_attempts",
        "pass_at_k_correct",
        "selected_attempt",
        "attempts",
        "question",
        "answer",
        "correct",
        "trace_length",
        "teacher_accept_rate",
        "trace",
        "posthoc_student_surprisal",
        "posthoc_student_entropy",
        "posthoc_student_ens",
        "posthoc_student_stats",
        "tokens",
    ]
    return {key: result.get(key) for key in keys}

def summarize_results(results: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["threshold_key"]].append(result)
    threshold_rows = []
    for key in THRESHOLD_KEYS:
        rows = grouped.get(key, [])
        graded = [row for row in rows if row.get("correct") is not None]
        correct = sum(1 for row in graded if row.get("correct") is True)
        attempts_used = [int(row.get("attempts_used") or row.get("attempt_index") or 1) for row in rows]
        threshold_rows.append(
            {
                "threshold_key": key,
                "ens_value": thresholds.get(key),
                "traces": len(rows),
                "queries": len(rows),
                "graded": len(graded),
                "correct": correct,
                "accuracy": correct / len(graded) if graded else None,
                "pass_at_k": correct / len(graded) if graded else None,
                "attempts_total": int(sum(attempts_used)) if attempts_used else 0,
                "avg_trials": float(np.mean(attempts_used)) if attempts_used else None,
                "max_trials_used": int(max(attempts_used)) if attempts_used else 0,
                "avg_teacher_accept_rate": float(np.mean([r["teacher_accept_rate"] for r in rows])) if rows else None,
                "avg_trace_length": float(np.mean([r["trace_length"] for r in rows])) if rows else None,
                "surprisal_mean_avg": float(np.mean([
                    r["posthoc_student_stats"]["surprisal"]["mean"]
                    for r in rows
                    if r.get("posthoc_student_stats", {}).get("surprisal", {}).get("mean") is not None
                ])) if any(
                    r.get("posthoc_student_stats", {}).get("surprisal", {}).get("mean") is not None
                    for r in rows
                ) else None,
                "entropy_mean_avg": float(np.mean([
                    r["posthoc_student_stats"]["entropy"]["mean"]
                    for r in rows
                    if r.get("posthoc_student_stats", {}).get("entropy", {}).get("mean") is not None
                ])) if any(
                    r.get("posthoc_student_stats", {}).get("entropy", {}).get("mean") is not None
                    for r in rows
                ) else None,
            }
        )
    examples = [
        {
            "threshold_key": result["threshold_key"],
            "source_id": result["source_id"],
            "row_index": result["row_index"],
            "correct": result["correct"],
            "grader": result["grader"],
            "extracted_answer": result["extracted_answer"],
            "answer": result["answer"],
            "trace_length": result["trace_length"],
            "teacher_accept_rate": result["teacher_accept_rate"],
            "attempt_index": result.get("attempt_index"),
            "attempts_used": result.get("attempts_used"),
            "max_attempts": result.get("max_attempts"),
            "pass_at_k_correct": result.get("pass_at_k_correct"),
            "selected_attempt": result.get("selected_attempt"),
            "attempts": result.get("attempts"),
            "has_think_end": result["has_think_end"],
            "posthoc_student_stats": result["posthoc_student_stats"],
        }
        for result in results
    ]
    audit = audit_threshold_trends(threshold_rows)
    return {
        "thresholds": threshold_rows,
        "examples": examples,
        "trend_audit": audit,
    }


def audit_threshold_trends(threshold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in threshold_rows if row.get("traces", 0) > 0]
    notes = []
    accept_rates = [row.get("avg_teacher_accept_rate") for row in rows]
    accept_monotonic = all(
        accept_rates[i] is None
        or accept_rates[i + 1] is None
        or float(accept_rates[i]) <= float(accept_rates[i + 1]) + 1e-12
        for i in range(len(accept_rates) - 1)
    )
    if not accept_monotonic:
        notes.append(
            "Teacher acceptance rate is not monotonic across the fixed ENS threshold sweep. "
            "This is unexpected for identical prefixes, but generated prefixes differ by threshold, "
            "so later positions are not direct apples-to-apples comparisons. Inspect token_prob_data.jsonl."
        )
    else:
        notes.append("Teacher acceptance rate is monotonic nondecreasing across completed thresholds.")

    notes.append(
        "Posthoc surprisal/entropy are reported as descriptive trace statistics. They are not required "
        "to be monotonic because each threshold changes the generated prefix and therefore the future "
        "student distribution."
    )
    return {
        "expected_acceptance_trend": "nondecreasing from 1/2 to 31/32",
        "acceptance_monotonic_non_decreasing": accept_monotonic,
        "accept_rates": [
            {"threshold_key": row.get("threshold_key"), "avg_teacher_accept_rate": row.get("avg_teacher_accept_rate")}
            for row in rows
        ],
        "notes": notes,
    }


def section_windows(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tokens:
        return []
    windows: list[tuple[str, int, int]] = []
    n = len(tokens)
    windows.append(("start", 0, min(n, 180)))
    deltas = [
        (i, t.get("teacher_minus_student_logprob"))
        for i, t in enumerate(tokens)
        if isinstance(t.get("teacher_minus_student_logprob"), (int, float))
        and math.isfinite(float(t["teacher_minus_student_logprob"]))
    ]
    if deltas:
        up = max(deltas, key=lambda x: x[1])[0]
        down = min(deltas, key=lambda x: x[1])[0]
        windows.append(("strongest upweight", max(0, up - 40), min(n, up + 41)))
        windows.append(("strongest downweight", max(0, down - 40), min(n, down + 41)))
    cumulative = ""
    think_end_pos = None
    for i, token in enumerate(tokens):
        cumulative += token.get("token") or ""
        if THINK_END in cumulative:
            think_end_pos = i
            break
    if think_end_pos is not None:
        windows.append(("around </think>", max(0, think_end_pos - 60), min(n, think_end_pos + 100)))
    windows.append(("end", max(0, n - 180), n))

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for name, start, end in windows:
        if start >= end or (start, end) in seen:
            continue
        seen.add((start, end))
        deduped.append({"name": name, "start": start, "end": end})
    return deduped


def make_html(
    results: list[dict[str, Any]],
    report: dict[str, Any],
    raw_records: list[dict[str, Any]],
    output_path: Path,
    max_records: int | None = None,
) -> None:
    html_results = results
    if max_records is not None and max_records > 0 and len(results) > max_records:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            grouped[result["threshold_key"]].append(result)
        per_threshold = max(1, max_records // max(1, len(grouped)))
        html_results = []
        for key in THRESHOLD_KEYS:
            rows = grouped.get(key, [])
            if not rows:
                continue
            # Include a deterministic mix of failed/correct rows for inspection.
            failed = [row for row in rows if row.get("correct") is not True]
            correct = [row for row in rows if row.get("correct") is True]
            picked = (failed[: per_threshold // 2] + correct[: per_threshold - per_threshold // 2])[:per_threshold]
            if len(picked) < per_threshold:
                seen = {id(row) for row in picked}
                picked.extend([row for row in rows if id(row) not in seen][: per_threshold - len(picked)])
            html_results.extend(picked)
    report = dict(report)
    report["html_records"] = len(html_results)
    report["html_total_records"] = len(results)
    viz_records = []
    for result in html_results:
        viz_records.append(
            {
                "id": f"{result['threshold_key']}::{result['source_id']}",
                "threshold_key": result["threshold_key"],
                "threshold_value": result["threshold_value"],
                "source_id": result["source_id"],
                "row_index": result["row_index"],
                "attempts_used": result.get("attempts_used"),
                "max_attempts": result.get("max_attempts"),
                "pass_at_k_correct": result.get("pass_at_k_correct"),
                "selected_attempt": result.get("selected_attempt"),
                "attempts": result.get("attempts"),
                "question": result["question"],
                "answer": result["answer"],
                "correct": result["correct"],
                "extracted_answer": result["extracted_answer"],
                "trace_length": result["trace_length"],
                "teacher_accept_rate": result["teacher_accept_rate"],
                "posthoc_student_surprisal": result["posthoc_student_surprisal"],
                "posthoc_student_entropy": result["posthoc_student_entropy"],
                "posthoc_student_stats": result["posthoc_student_stats"],
                "tokens": result["tokens"],
                "sections": section_windows(result["tokens"]),
            }
        )
    data_json = json.dumps({"records": viz_records, "report": report, "raw_records": raw_records}, ensure_ascii=False)
    escaped_data = html.escape(data_json, quote=False)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IPSD Basic Visualization</title>
<style>
body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: #1f2933; background: #f7f7f4; }}
header {{ padding: 16px 20px; background: #20242b; color: white; }}
h1 {{ font-size: 18px; margin: 0 0 8px; }}
.summary {{ display: flex; gap: 12px; flex-wrap: wrap; font-size: 13px; }}
.summary span {{ background: rgba(255,255,255,.12); padding: 4px 8px; border-radius: 4px; }}
main {{ padding: 16px 20px 32px; }}
.controls {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 14px; }}
select {{ font: inherit; padding: 5px 8px; border: 1px solid #a8adb5; border-radius: 4px; background: white; }}
.meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; margin-bottom: 14px; font-size: 13px; }}
.meta div {{ background: white; border: 1px solid #ddd8cc; padding: 8px; border-radius: 4px; }}
.question {{ background: white; border: 1px solid #ddd8cc; padding: 10px; border-radius: 4px; margin-bottom: 14px; white-space: pre-wrap; }}
.section {{ margin: 14px 0; background: white; border: 1px solid #ddd8cc; border-radius: 4px; }}
.section h2 {{ margin: 0; padding: 8px 10px; font-size: 13px; background: #ece8dc; border-bottom: 1px solid #ddd8cc; }}
.tokens {{ padding: 10px; line-height: 1.9; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; white-space: pre-wrap; overflow-wrap: anywhere; }}
.tok {{ border-radius: 3px; padding: 1px 2px; margin: 0 1px; cursor: default; border-bottom: 2px solid transparent; }}
.tok.teacher {{ background-color: rgba(231, 111, 81, var(--alpha)); }}
.tok.student {{ background-color: rgba(42, 125, 176, var(--alpha)); }}
.tok.up {{ border-bottom-color: rgba(32, 135, 86, .8); }}
.tok.down {{ border-bottom-color: rgba(190, 55, 55, .8); }}
.legend {{ font-size: 12px; color: #4b5563; margin-bottom: 10px; }}
.chart {{ width: 100%; height: 260px; background: white; border: 1px solid #ddd8cc; border-radius: 4px; margin: 12px 0 14px; }}
.chart svg {{ width: 100%; height: 100%; display: block; }}
.chart .axis {{ stroke: #9aa0a6; stroke-width: 1; }}
.chart .surprisal {{ fill: none; stroke: #b63d2b; stroke-width: 1.5; }}
.chart .entropy {{ fill: none; stroke: #2368a0; stroke-width: 1.5; }}
.chart text {{ font-size: 11px; fill: #4b5563; }}
</style>
</head>
<body>
<header>
  <h1>IPSD Basic Visualization</h1>
  <div class="summary" id="summary"></div>
</header>
<main>
  <div class="controls">
    <label>Threshold <select id="threshold"></select></label>
    <label>Example <select id="example"></select></label>
  </div>
  <div class="legend">Orange = accepted self-teacher token. Blue = student fallback token. Green/red underline = teacher-vs-student logprob delta direction.</div>
  <div class="meta" id="meta"></div>
  <div class="chart" id="chart"></div>
  <div class="question" id="question"></div>
  <div id="sections"></div>
</main>
<script id="payload" type="application/json">{escaped_data}</script>
<script>
const payload = JSON.parse(document.getElementById('payload').textContent);
const records = payload.records;
const byThreshold = new Map();
for (const r of records) {{
  if (!byThreshold.has(r.threshold_key)) byThreshold.set(r.threshold_key, []);
  byThreshold.get(r.threshold_key).push(r);
}}
const thresholdSel = document.getElementById('threshold');
const exampleSel = document.getElementById('example');
const order = ["1/2","3/4","7/8","15/16","31/32"].filter(k => byThreshold.has(k));
for (const key of order) {{
  const opt = document.createElement('option');
  opt.value = key;
  opt.textContent = key;
  thresholdSel.appendChild(opt);
}}
function esc(s) {{ return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function fmt(x, digits=4) {{ return typeof x === 'number' && Number.isFinite(x) ? x.toFixed(digits) : 'n/a'; }}
function alpha(token) {{
  const delta = token.teacher_minus_student_logprob;
  const ens = token.teacher_token_student_ens;
  const scale = typeof delta === 'number' ? Math.min(1, Math.abs(delta) / 5) : Math.min(1, Math.abs(ens || 0) / 5);
  return (0.18 + 0.55 * scale).toFixed(3);
}}
function points(values, width, height, pad, maxY) {{
  const n = values.length;
  if (!n || !maxY) return '';
  return values.map((v, i) => {{
    const x = pad + (n === 1 ? 0 : i * (width - 2 * pad) / (n - 1));
    const y = height - pad - Math.max(0, Math.min(1, v / maxY)) * (height - 2 * pad);
    return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
  }}).join(' ');
}}
function renderChart(record) {{
  const width = 900, height = 260, pad = 32;
  const sVals = (record.posthoc_student_surprisal || []).filter(Number.isFinite);
  const eVals = (record.posthoc_student_entropy || []).filter(Number.isFinite);
  const maxY = Math.max(1e-6, ...sVals, ...eVals);
  const sPts = points(sVals, width, height, pad, maxY);
  const ePts = points(eVals, width, height, pad, maxY);
  const stats = record.posthoc_student_stats || {{}};
  document.getElementById('chart').innerHTML = `
    <svg viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none" role="img" aria-label="posthoc student surprisal and entropy by token position">
      <line class="axis" x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}"></line>
      <line class="axis" x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height-pad}}"></line>
      <polyline class="surprisal" points="${{sPts}}"></polyline>
      <polyline class="entropy" points="${{ePts}}"></polyline>
      <text x="${{pad}}" y="18">posthoc student stats: surprisal mean=${{fmt(stats.surprisal?.mean)}} median=${{fmt(stats.surprisal?.median)}} min=${{fmt(stats.surprisal?.min)}} max=${{fmt(stats.surprisal?.max)}}</text>
      <text x="${{pad}}" y="34">entropy mean=${{fmt(stats.entropy?.mean)}} median=${{fmt(stats.entropy?.median)}} min=${{fmt(stats.entropy?.min)}} max=${{fmt(stats.entropy?.max)}}</text>
      <text x="${{width-pad-180}}" y="18" fill="#b63d2b">surprisal</text>
      <text x="${{width-pad-180}}" y="34" fill="#2368a0">entropy</text>
      <text x="${{width/2-45}}" y="${{height-6}}">token position</text>
    </svg>`;
}}
function populateExamples() {{
  exampleSel.innerHTML = '';
  for (const r of byThreshold.get(thresholdSel.value) || []) {{
    const opt = document.createElement('option');
    opt.value = r.id;
    opt.textContent = `${{r.row_index}} / ${{r.source_id}} / correct=${{r.correct}}`;
    exampleSel.appendChild(opt);
  }}
}}
function render() {{
  const record = records.find(r => r.id === exampleSel.value) || (byThreshold.get(thresholdSel.value) || [])[0];
  if (!record) return;
  document.getElementById('meta').innerHTML = `
    <div><b>threshold</b><br>${{esc(record.threshold_key)}} = ${{fmt(record.threshold_value)}}</div>
    <div><b>correct</b><br>${{esc(record.correct)}}</div>
    <div><b>extracted</b><br>${{esc(record.extracted_answer)}}</div>
    <div><b>gold</b><br>${{esc(record.answer)}}</div>
    <div><b>trace length</b><br>${{record.trace_length}}</div>
    <div><b>teacher accept rate</b><br>${{fmt(record.teacher_accept_rate, 3)}}</div>
  `;
  renderChart(record);
  document.getElementById('question').textContent = record.question;
  const sections = document.getElementById('sections');
  sections.innerHTML = '';
  for (const section of record.sections) {{
    const box = document.createElement('div');
    box.className = 'section';
    const h = document.createElement('h2');
    h.textContent = `${{section.name}} [${{section.start}}, ${{section.end}})`;
    const toks = document.createElement('div');
    toks.className = 'tokens';
    for (const token of record.tokens.slice(section.start, section.end)) {{
      const span = document.createElement('span');
      const delta = token.teacher_minus_student_logprob;
      span.className = `tok ${{token.source}} ${{typeof delta === 'number' && delta >= 0 ? 'up' : 'down'}}`;
      span.style.setProperty('--alpha', alpha(token));
      span.textContent = token.token;
      span.title = [
        `pos=${{token.position}} source=${{token.source}}`,
        `ENS=${{fmt(token.teacher_token_student_ens)}} threshold=${{fmt(token.ens_threshold)}} margin=${{fmt(token.ens_margin)}}`,
        `surprisal=${{fmt(token.student_surprisal)}} entropy=${{fmt(token.student_entropy)}}`,
        `teacher-student logprob delta=${{fmt(delta)}}`,
        `teacher proposed=${{JSON.stringify(token.teacher_token)}}`,
        `student fallback=${{JSON.stringify(token.student_token)}}`
      ].join('\\n');
      toks.appendChild(span);
    }}
    box.appendChild(h);
    box.appendChild(toks);
    sections.appendChild(box);
  }}
}}
function renderSummary() {{
  const container = document.getElementById('summary');
  container.innerHTML = '';
  for (const row of payload.report.thresholds) {{
    const span = document.createElement('span');
    span.textContent = `${{row.threshold_key}}: ${{row.correct}}/${{row.graded}} correct, accept=${{fmt(row.avg_teacher_accept_rate, 3)}}`;
    container.appendChild(span);
  }}
}}
thresholdSel.addEventListener('change', () => {{ populateExamples(); render(); }});
exampleSel.addEventListener('change', render);
renderSummary();
populateExamples();
render();
</script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


async def async_main() -> None:
    load_dotenv_if_present()
    args = parse_args()
    threshold_keys = [x.strip() for x in args.thresholds.split(",") if x.strip()]
    unknown = [key for key in threshold_keys if key not in THRESHOLD_QUANTILES]
    if unknown:
        raise ValueError(f"Unknown threshold keys: {unknown}. Supported: {sorted(THRESHOLD_QUANTILES)}")
    output_dir = prepare_output_dir(args)
    rows = load_rows(args)
    selected_rows = rows[: args.num_examples]
    calibration_rows = rows[: args.calibration_limit]
    teacher_gpu, student_gpu = default_gpus(args)
    max_model_len = args.max_model_len or (args.max_prompt_len + args.max_gen_len)

    print(f"loaded {len(selected_rows)} generation rows and {len(calibration_rows)} calibration rows", flush=True)
    print(f"launching teacher engine on GPU {teacher_gpu}; student engine on GPU {student_gpu}", flush=True)
    teacher_engine, teacher_tokenizer, resolved_model = await setup_async_engine(
        args.model,
        gpu=teacher_gpu,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.teacher_gpu_mem_util,
        dtype=args.dtype,
        max_logprobs=args.logprobs,
    )
    student_engine, student_tokenizer, _ = await setup_async_engine(
        args.model,
        gpu=student_gpu,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.student_gpu_mem_util,
        dtype=args.dtype,
        max_logprobs=args.logprobs,
    )
    if len(teacher_tokenizer) != len(student_tokenizer):
        raise RuntimeError("Teacher and student tokenizers differ unexpectedly")

    thresholds = await calibrate_thresholds(calibration_rows, student_engine, student_tokenizer, args)
    write_json(
        output_dir / "calibration.json",
        {
            "model": args.model,
            "resolved_model": resolved_model,
            "dataset": args.dataset,
            "thresholds": thresholds,
            "threshold_keys": threshold_keys,
            "calibration_limit": args.calibration_limit,
            "max_calibration_tokens": args.max_calibration_tokens,
            "logprobs": args.logprobs,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    )

    raw_records = await assess_raw_traces(selected_rows, student_engine, teacher_engine, student_tokenizer, args, output_dir)

    results: list[dict[str, Any]] = []
    total = len(threshold_keys) * len(selected_rows)
    done = 0
    started = time.time()
    generation_concurrency = max(1, int(args.generation_concurrency))
    max_attempts = max(1, int(args.max_attempts))
    for threshold_key in threshold_keys:
        threshold_value = thresholds[threshold_key]
        sem = asyncio.Semaphore(generation_concurrency)

        async def run_generation(row: BasicRow) -> dict[str, Any]:
            async with sem:
                attempt_summaries: list[dict[str, Any]] = []
                selected: dict[str, Any] | None = None
                for attempt_index in range(1, max_attempts + 1):
                    print(
                        f"generation attempt: threshold={threshold_key} "
                        f"row_index={row.row_index} source_id={row.source_id} "
                        f"attempt={attempt_index}/{max_attempts}",
                        flush=True,
                    )
                    result = await generate_ipsd_trace(
                        row,
                        threshold_key,
                        threshold_value,
                        teacher_engine,
                        student_engine,
                        student_tokenizer,
                        args,
                        attempt_index=attempt_index,
                    )
                    attempt_summaries.append(
                        {
                            "attempt_index": attempt_index,
                            "correct": result.get("correct"),
                            "grader": result.get("grader"),
                            "extracted_answer": result.get("extracted_answer"),
                            "trace_length": result.get("trace_length"),
                            "teacher_accept_rate": result.get("teacher_accept_rate"),
                            "has_think_end": result.get("has_think_end"),
                            "stopped_by_context": result.get("stopped_by_context"),
                        }
                    )
                    print(
                        f"completed attempt: threshold={threshold_key} "
                        f"source_id={row.source_id} attempt={attempt_index}/{max_attempts} "
                        f"correct={result['correct']} length={result['trace_length']} "
                        f"accept_rate={result['teacher_accept_rate']:.3f}",
                        flush=True,
                    )
                    selected = result
                    if result.get("correct") is True:
                        break
                assert selected is not None
                selected["attempts"] = attempt_summaries
                selected["attempts_used"] = len(attempt_summaries)
                selected["max_attempts"] = max_attempts
                selected["pass_at_k_correct"] = any(row.get("correct") is True for row in attempt_summaries)
                selected["selected_attempt"] = selected.get("attempt_index")
                return selected

        for task in asyncio.as_completed([run_generation(row) for row in selected_rows]):
            result = await task
            done += 1
            results.append(result)
            print(
                f"completed generation {done}/{total}: threshold={threshold_key} source_id={result['source_id']}: "
                f"pass_at_{max_attempts}={result['pass_at_k_correct']} attempts={result['attempts_used']} "
                f"selected_attempt={result['selected_attempt']} correct={result['correct']} "
                f"length={result['trace_length']} accept_rate={result['teacher_accept_rate']:.3f}",
                flush=True,
            )

    row_by_index = {row.row_index: row for row in selected_rows}
    await assess_generated_traces(results, row_by_index, student_engine, teacher_engine, student_tokenizer, args)
    report = summarize_results(results, thresholds)
    for result in results:
        append_jsonl(output_dir / "sft_traces.jsonl", compact_sft_record(result))
        append_jsonl(output_dir / "token_prob_data.jsonl", token_prob_record(result))
    write_json(output_dir / "correctness_report.json", report)
    make_html(results, report, raw_records, output_dir / "ipsd_basic_visualization.html", args.html_max_records)
    write_json(
        output_dir / "run_summary.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.time() - started,
            "model": args.model,
            "resolved_model": resolved_model,
            "dataset": args.dataset,
            "num_examples": args.num_examples,
            "threshold_keys": threshold_keys,
            "max_seq_len": max_model_len,
            "max_model_len": max_model_len,
            "max_prompt_len": args.max_prompt_len,
            "max_gen_len": args.max_gen_len,
            "generation_concurrency": generation_concurrency,
            "assessment_concurrency": max(1, int(args.assessment_concurrency)),
            "assessment_use_both_engines": bool(args.assessment_use_both_engines),
            "max_attempts": max_attempts,
            "html_max_records": args.html_max_records,
            "outputs": {
                "sft_traces": str((output_dir / "sft_traces.jsonl").resolve()),
                "token_prob_data": str((output_dir / "token_prob_data.jsonl").resolve()),
                "correctness_report": str((output_dir / "correctness_report.json").resolve()),
                "raw_trace_stats": str((output_dir / "raw_trace_stats.jsonl").resolve()),
                "raw_trace_report": str((output_dir / "raw_trace_report.json").resolve()),
                "html": str((output_dir / "ipsd_basic_visualization.html").resolve()),
            },
            "correctness": report["thresholds"],
            "trend_audit": report.get("trend_audit"),
        },
    )
    print(json.dumps(report["thresholds"], indent=2), flush=True)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
