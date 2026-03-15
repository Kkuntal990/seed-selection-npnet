#!/usr/bin/env python3
"""RTN (W8A16) Quantization via bitsandbytes — Full Pipeline (single script)."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger("rtn_notebook")
sns.set_theme(style="whitegrid", font_scale=1.1)

PROJECT_ROOT = Path.cwd().resolve()
print(f"Working directory: {PROJECT_ROOT}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")

# ════════════════════════════════════════════════════════════════
# Inline configuration (replaces config/default.yaml)
# ════════════════════════════════════════════════════════════════

def _resolve_paths(cfg):
    paths = cfg.get("paths", {})
    for key in ("artifacts_dir", "results_dir", "calibration_dir", "plots_dir"):
        if key in paths and not Path(paths[key]).is_absolute():
            paths[key] = str(PROJECT_ROOT / paths[key])

cfg = {
    "base_model": "Qwen/Qwen2-1.5B",
    "calibration": {
        "dataset": "wikitext",
        "dataset_name": "wikitext-2-raw-v1",
        "split": "train",
        "num_samples": 512,
        "max_length": 2048,
        "seed": 42,
    },
    "quant_configs": [
        {"method": "gptq", "bits": 4, "group_size": 128, "desc_act": False},
        {"method": "awq",  "bits": 4, "group_size": 128, "zero_point": True, "version": "GEMM"},
        {"method": "rtn",  "bits": 8, "per_channel": True},
    ],
    "ablation_variants": ["full_quant", "attn_only_quant", "mlp_only_quant"],
    "eval": {
        "perplexity": {
            "dataset": "wikitext", "dataset_name": "wikitext-2-raw-v1",
            "split": "test", "max_length": 512, "stride": 256,
            "max_eval_tokens": 131072,
        },
        "gsm8k":          {"num_fewshot": 8, "num_samples": 300, "max_new_tokens": 256},
        "math":           {"num_samples": 500, "max_new_tokens": 1024},
        "arc_challenge":  {"num_samples": 500, "max_new_tokens": 5},
        "gpqa":           {"max_new_tokens": 5},
    },
    "accuracy_weights": {"gsm8k": 0.25, "math": 0.25, "arc_challenge": 0.25, "gpqa": 0.25},
    "deployment_benchmark": {
        "prompt_lengths": [128, 512, 1024],
        "generation_lengths": [128, 256],
        "batch_size": 1, "warmup_iters": 2, "bench_iters": 5,
    },
    "paths": {
        "artifacts_dir": "/output/ece226/artifacts3",
        "results_dir": "/output/ece226/results3",
        "calibration_dir": "/output/ece226/calibration_data3",
        "plots_dir": "/output/ece226/results/plots3",
    },
}
_resolve_paths(cfg)

METHOD = "rtn"
VARIANTS = ["full_quant", "attn_only_quant", "mlp_only_quant"]

print(f"Base model : {cfg['base_model']}")
print(f"Method     : {METHOD}")
print(f"Variants   : {VARIANTS}")

# ════════════════════════════════════════════════════════════════
# Module utilities — selective quantization via llm_int8_skip_modules
# ════════════════════════════════════════════════════════════════

ATTN_PATTERN = re.compile(r"self_attn\.(q|k|v|o)_proj")
MLP_PATTERN  = re.compile(r"mlp\.(gate|up|down)_proj")

@dataclass
class ModuleClassification:
    attn:  list[str] = field(default_factory=list)
    mlp:   list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    @property
    def all_linear(self) -> list[str]:
        return self.attn + self.mlp + self.other

def classify_linear_modules(model: nn.Module) -> ModuleClassification:
    cls = ModuleClassification()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if ATTN_PATTERN.search(name):   cls.attn.append(name)
        elif MLP_PATTERN.search(name):  cls.mlp.append(name)
        else:                           cls.other.append(name)
    return cls


def build_skip_modules(classification: ModuleClassification, variant: str) -> list[str]:
    """Build the list of module names to SKIP (keep in FP16) for bitsandbytes."""
    if variant == "full_quant":
        return []
    if variant == "attn_only_quant":
        return classification.mlp + classification.other
    if variant == "mlp_only_quant":
        return classification.attn + classification.other
    raise ValueError(f"Unknown ablation variant: {variant}")

# ════════════════════════════════════════════════════════════════
# Calibration data
# ════════════════════════════════════════════════════════════════

def get_calibration_data(cfg, tokenizer=None):
    cal_cfg = cfg["calibration"]
    cache_dir = Path(cfg["paths"]["calibration_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"calibration_{cal_cfg['num_samples']}.pt"
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    ds = load_dataset(cal_cfg["dataset"], cal_cfg["dataset_name"], split=cal_cfg["split"])
    all_text = "\n\n".join(t for t in ds["text"] if t.strip())
    rng = torch.Generator().manual_seed(cal_cfg["seed"])
    max_len = cal_cfg["max_length"]
    encoded = tokenizer(all_text, return_tensors="pt")
    total_tokens = encoded.input_ids.shape[1]
    samples = []
    starts = torch.randint(0, total_tokens - max_len, (cal_cfg["num_samples"],), generator=rng)
    for start in starts:
        end = start + max_len
        input_ids = encoded.input_ids[:, start:end]
        samples.append({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})
    torch.save(samples, cache_path)
    return samples

# ════════════════════════════════════════════════════════════════
# Checkpoint size
# ════════════════════════════════════════════════════════════════

def bytes_per_param_from_safetensors(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    st_files = list(checkpoint_dir.glob("*.safetensors"))
    if not st_files:
        bin_files = list(checkpoint_dir.glob("*.bin")) + list(checkpoint_dir.glob("*.pt"))
        if not bin_files:
            logger.warning("No checkpoint files found in %s", checkpoint_dir); return 0.0
        total_bytes = total_params = 0
        for p in bin_files:
            sd = torch.load(str(p), map_location="cpu", weights_only=True)
            for t in sd.values():
                if hasattr(t, "numel"): total_bytes += t.numel() * t.element_size(); total_params += t.numel()
        return total_bytes / max(total_params, 1)
    try:
        from safetensors import safe_open
    except ImportError:
        return 0.0
    total_bytes = total_params = 0
    for p in st_files:
        with safe_open(str(p), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                total_bytes += t.numel() * t.element_size(); total_params += t.numel()
    if total_params == 0: return 0.0
    bpp = total_bytes / total_params
    logger.info("Checkpoint %s: %.2f GB, %d params, %.3f bytes/param", checkpoint_dir.name, total_bytes/1e9, total_params, bpp)
    return bpp

# ════════════════════════════════════════════════════════════════
# RTN quantization via bitsandbytes INT8
# ════════════════════════════════════════════════════════════════

def quantize_rtn(cfg, variant="full_quant"):
    """Quantize the base model to INT8 via bitsandbytes and save the artifact."""
    output_dir = Path(cfg["paths"]["artifacts_dir"]) / "rtn_w8_perchannel" / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    if (output_dir / "config.json").exists():
        logger.info("RTN artifact already exists at %s — skipping.", output_dir)
        return output_dir

    classification_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.float16, trust_remote_code=True, device_map="cpu",
    )
    classification = classify_linear_modules(classification_model)
    skip_list = build_skip_modules(classification, variant)
    del classification_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=skip_list if skip_list else None,
    )

    logger.info("Loading model in 8-bit (%s), skipping %d modules...", variant, len(skip_list))
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)

    logger.info("Saving RTN checkpoint to %s", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir

# ════════════════════════════════════════════════════════════════
# Evaluation functions  (all HuggingFace-native, no vLLM dependency)
# ════════════════════════════════════════════════════════════════

def _load_eval_model(model_path):
    """Load a model for evaluation — works for both FP16 and RTN checkpoints."""
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )


def evaluate_perplexity(model_path, cfg, *, device="cuda", **_kw):
    ppl_cfg = cfg["eval"]["perplexity"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = _load_eval_model(model_path)
    model.eval()
    ds = load_dataset(ppl_cfg["dataset"], ppl_cfg["dataset_name"], split=ppl_cfg["split"])
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    encodings = tokenizer(text, return_tensors="pt")
    max_len = ppl_cfg["max_length"]
    stride = ppl_cfg["stride"]
    max_eval_tokens = ppl_cfg.get("max_eval_tokens", 131072)
    seq_len = min(encodings.input_ids.size(1), max_eval_tokens)
    nlls, prev_end = [], 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        input_ids = encodings.input_ids[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :max(0, prev_end - begin)] = -100
        with torch.no_grad():
            nlls.append(model(input_ids, labels=target_ids).loss)
        prev_end = end
        if end >= seq_len:
            break
    ppl = math.exp(torch.stack(nlls).mean().item())
    del model; torch.cuda.empty_cache()
    logger.info("Perplexity (%.0fk tokens): %.2f", seq_len / 1e3, ppl)
    return ppl


# ────────────────────────────────────────────────────────────────
# Shared generation helpers
# ────────────────────────────────────────────────────────────────

_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")

def _extract_numeric_answer(text):
    m = _ANSWER_RE.search(text)
    if m:
        return float(m.group(1).replace(",", ""))
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    return float(nums[-1].replace(",", "")) if nums else None


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

def _extract_boxed_answer(text):
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _normalize_math_answer(ans):
    if ans is None:
        return None
    ans = ans.strip()
    ans = ans.replace("\\$", "").replace(",", "").replace(" ", "")
    ans = ans.replace("\\frac", "frac").replace("\\dfrac", "frac")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\text{", "").replace("\\mathrm{", "")
    ans = ans.rstrip("}")
    try:
        return str(float(ans))
    except ValueError:
        return ans.lower()


# ────────────────────────────────────────────────────────────────
# Batched generation helper
# ────────────────────────────────────────────────────────────────

EVAL_BATCH_SIZE = 8  # process 8 prompts at a time

def _batched_generate(model, tokenizer, prompts, max_new_tokens, max_input_len=4096):
    """Generate for a list of prompts in batches with left-padding."""
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    all_outputs = []
    for i in range(0, len(prompts), EVAL_BATCH_SIZE):
        batch_prompts = prompts[i : i + EVAL_BATCH_SIZE]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_input_len,
        ).to(model.device)
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j in range(out_ids.shape[0]):
            gen_ids = out_ids[j][inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_outputs.append(gen_text)
    tokenizer.padding_side = orig_side
    return all_outputs


# ────────────────────────────────────────────────────────────────
# GSM8K  (8-shot, 300 samples, batched)
# ────────────────────────────────────────────────────────────────

_GSM_PROMPT = "Question: {question}\nAnswer: Let's think step by step.\n"

def evaluate_gsm8k(model_path, cfg, **_kw):
    gsm_cfg = cfg["eval"]["gsm8k"]
    n_shot = gsm_cfg.get("num_fewshot", 8)
    num_samples = gsm_cfg.get("num_samples", 300)
    max_new = gsm_cfg.get("max_new_tokens", 256)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_eval_model(model_path); model.eval()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))

    exemplar_ds = load_dataset("openai/gsm8k", "main", split="train")
    exemplars = exemplar_ds.shuffle(seed=42).select(range(n_shot))
    prefix_text = ""
    for ex in exemplars:
        prefix_text += _GSM_PROMPT.format(question=ex["question"]) + ex["answer"] + "\n\n"

    prompts = [prefix_text + _GSM_PROMPT.format(question=row["question"]) for row in ds]
    gold_answers = [_extract_numeric_answer(row["answer"]) for row in ds]

    outputs = _batched_generate(model, tokenizer, prompts, max_new, max_input_len=4096)

    correct = 0
    for gen, gold in zip(outputs, gold_answers):
        pred = _extract_numeric_answer(gen)
        if pred is not None and gold is not None and abs(pred - gold) < 1e-3:
            correct += 1
    del model; torch.cuda.empty_cache()
    acc = correct / max(len(outputs), 1)
    logger.info("GSM8K accuracy: %.3f (%d/%d)", acc, correct, len(outputs))
    return acc

# ────────────────────────────────────────────────────────────────
# MATH  (0-shot, 500 Level-5 problems, extract \boxed{} answer, batched)
# ────────────────────────────────────────────────────────────────

_MATH_PROMPT = (
    "Solve the following math problem. "
    "Show your work and put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}\n\nSolution:"
)

def evaluate_math(model_path, cfg, **_kw):
    math_cfg = cfg["eval"]["math"]
    num_samples = math_cfg.get("num_samples", 500)
    max_new = math_cfg.get("max_new_tokens", 1024)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_eval_model(model_path); model.eval()
    ds = load_dataset("lighteval/MATH-Hard", split="test")
    ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))

    prompts = [_MATH_PROMPT.format(problem=row["problem"]) for row in ds]
    gold_answers = [_normalize_math_answer(_extract_boxed_answer(row["solution"])) for row in ds]

    outputs = _batched_generate(model, tokenizer, prompts, max_new, max_input_len=2048)

    correct = 0
    for gen, gold in zip(outputs, gold_answers):
        pred = _normalize_math_answer(_extract_boxed_answer(gen))
        if pred is not None and gold is not None and pred == gold:
            correct += 1
    del model; torch.cuda.empty_cache()
    acc = correct / max(len(outputs), 1)
    logger.info("MATH accuracy: %.3f (%d/%d)", acc, correct, len(outputs))
    return acc


# ────────────────────────────────────────────────────────────────
# ARC-Challenge  (0-shot MC, 500 samples, batched)
# ────────────────────────────────────────────────────────────────

_ARC_CHOICES = ["A", "B", "C", "D", "E"]

def _format_arc_prompt(row):
    labels = row["choices"]["label"]
    texts  = row["choices"]["text"]
    prompt = row["question"] + "\n"
    for lbl, txt in zip(labels, texts):
        prompt += f"{lbl}. {txt}\n"
    prompt += "Answer:"
    return prompt, labels

def evaluate_arc_challenge(model_path, cfg, **_kw):
    arc_cfg = cfg["eval"]["arc_challenge"]
    num_samples = arc_cfg.get("num_samples", 500)
    max_new = arc_cfg.get("max_new_tokens", 5)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_eval_model(model_path); model.eval()
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))

    prompts = []
    gold_answers = []
    for row in ds:
        prompt, _labels = _format_arc_prompt(row)
        prompts.append(prompt)
        gold_answers.append(row["answerKey"])

    outputs = _batched_generate(model, tokenizer, prompts, max_new, max_input_len=2048)

    correct = 0
    for gen, gold in zip(outputs, gold_answers):
        gen = gen.strip()
        pred_letter = gen.split()[0].strip().rstrip(".") if gen.split() else ""
        if pred_letter.upper() == gold.upper():
            correct += 1
    del model; torch.cuda.empty_cache()
    acc = correct / max(len(outputs), 1)
    logger.info("ARC-Challenge accuracy: %.3f (%d/%d)", acc, correct, len(outputs))
    return acc


# ────────────────────────────────────────────────────────────────
# GPQA  (0-shot MC, full diamond split ~198 Qs, shuffled choices, batched)
# ────────────────────────────────────────────────────────────────

def _format_gpqa_prompt(row, rng):
    correct_ans = row["Correct Answer"]
    wrong = [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    options = [correct_ans] + wrong
    rng.shuffle(options)
    correct_idx = options.index(correct_ans)
    letters = ["A", "B", "C", "D"]
    prompt = row["Question"] + "\n"
    for lbl, opt in zip(letters, options):
        prompt += f"{lbl}. {opt}\n"
    prompt += "Answer:"
    return prompt, letters[correct_idx]

def evaluate_gpqa(model_path, cfg, **_kw):
    gpqa_cfg = cfg["eval"]["gpqa"]
    max_new = gpqa_cfg.get("max_new_tokens", 5)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_eval_model(model_path); model.eval()
    hf_token = os.environ.get("HF_TOKEN") or None
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=hf_token)

    import random
    rng = random.Random(42)

    prompts = []
    gold_answers = []
    for row in ds:
        prompt, gold_letter = _format_gpqa_prompt(row, rng)
        prompts.append(prompt)
        gold_answers.append(gold_letter)

    outputs = _batched_generate(model, tokenizer, prompts, max_new, max_input_len=4096)

    correct = 0
    for gen, gold in zip(outputs, gold_answers):
        gen = gen.strip()
        pred_letter = gen.split()[0].strip().rstrip(".") if gen.split() else ""
        if pred_letter.upper() == gold.upper():
            correct += 1
    del model; torch.cuda.empty_cache()
    acc = correct / max(len(outputs), 1)
    logger.info("GPQA-Diamond accuracy: %.3f (%d/%d)", acc, correct, len(outputs))
    return acc


# ────────────────────────────────────────────────────────────────
# Deployment benchmark  (multi-prompt sweep)
# ────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    tokens_per_sec: float
    ms_per_token: float
    peak_vram_gb: float
    prompt_length: int
    generation_length: int


def benchmark_throughput(model_path, cfg, **_kw):
    bc = cfg["deployment_benchmark"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_eval_model(model_path); model.eval()

    prompt_lengths = bc.get("prompt_lengths", [128, 512, 1024])
    gen_lengths = bc.get("generation_lengths", [128, 256])
    results = []

    for plen in prompt_lengths:
        for glen in gen_lengths:
            dummy = "Hello " * (plen // 2)
            inputs = tokenizer(dummy, return_tensors="pt", truncation=True, max_length=plen).to(model.device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            for _ in range(bc["warmup_iters"]):
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=glen, do_sample=False)
            total_tokens = 0
            start = time.perf_counter()
            for _ in range(bc["bench_iters"]):
                with torch.no_grad():
                    out_ids = model.generate(**inputs, max_new_tokens=glen, do_sample=False)
                total_tokens += out_ids.shape[1] - inputs["input_ids"].shape[1]
            elapsed = time.perf_counter() - start
            tok_s = total_tokens / elapsed
            peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
            r = BenchmarkResult(
                tokens_per_sec=tok_s,
                ms_per_token=(elapsed / total_tokens) * 1000,
                peak_vram_gb=peak,
                prompt_length=plen,
                generation_length=glen,
            )
            results.append(r)
            logger.info(
                "Bench prompt=%d gen=%d: %.1f tok/s, %.2f ms/tok, %.2f GB",
                plen, glen, tok_s, r.ms_per_token, peak,
            )

    del model; torch.cuda.empty_cache()
    best = max(results, key=lambda r: r.tokens_per_sec)
    return best

# ════════════════════════════════════════════════════════════════
# Layer diagnostics
# ════════════════════════════════════════════════════════════════

@dataclass
class LayerStats:
    method: str; variant: str; layer_idx: int; module_family: str
    module_name: str; mse: float; p50: float; p90: float; p99: float; p99_9: float; act_max: float

@dataclass
class LogitDiagnostics:
    method: str; variant: str; kl_div: float; cosine_sim: float
    top1_agreement: float; top5_agreement: float

def _module_family(name):
    if ATTN_PATTERN.search(name): return "attn"
    if MLP_PATTERN.search(name): return "mlp"
    return "other"

def _layer_index(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1

def _activation_percentiles(tensor):
    flat = tensor.float().abs().flatten()
    if flat.numel() == 0: return {"p50": 0, "p90": 0, "p99": 0, "p99_9": 0, "act_max": 0}
    q = torch.quantile(flat, torch.tensor([0.50, 0.90, 0.99, 0.999], device=flat.device))
    return {"p50": q[0].item(), "p90": q[1].item(), "p99": q[2].item(), "p99_9": q[3].item(), "act_max": flat.max().item()}

class _HookCollector:
    def __init__(self): self.data = {}; self._handles = []
    def register(self, model):
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and _module_family(name) != "other":
                self._handles.append(mod.register_forward_hook(self._make_hook(name)))
    def _make_hook(self, name):
        def hook(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            self.data[name] = {"input": x.detach().cpu(), "output": out.detach().cpu()}
        return hook
    def remove(self):
        for h in self._handles: h.remove()
        self._handles.clear()

def _load_model(path, **_kw):
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )

def _compute_logit_diagnostics(fp16_logits, q_logits, method, variant):
    fp = fp16_logits.float().flatten(0, 1); qp = q_logits.float().flatten(0, 1)
    log_p = F.log_softmax(fp, dim=-1); p = log_p.exp()
    log_q = F.log_softmax(qp, dim=-1)
    kl = F.kl_div(log_q, p, reduction="batchmean", log_target=False).item()
    cos = F.cosine_similarity(fp, qp, dim=-1).mean().item()
    top1_agree = (fp.argmax(-1) == qp.argmax(-1)).float().mean().item()
    t5f, t5q = fp.topk(5, dim=-1).indices, qp.topk(5, dim=-1).indices
    top5 = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(t5f, t5q)) / (t5f.size(0) * 5)
    return LogitDiagnostics(method=method, variant=variant, kl_div=kl, cosine_sim=cos, top1_agreement=top1_agree, top5_agreement=top5)

def run_layer_diagnostics(fp16_path, quant_path, cfg, method, variant, *, probe_seqs=4, probe_len=128, device="cuda", **_kw):
    tokenizer = AutoTokenizer.from_pretrained(fp16_path, trust_remote_code=True)
    cal = get_calibration_data(cfg, tokenizer)
    input_ids = torch.cat([s["input_ids"][:, :probe_len] for s in cal[:probe_seqs]], dim=0).to(device)

    fp16_model = _load_model(fp16_path); fp16_model.eval()
    fp16_hooks = _HookCollector(); fp16_hooks.register(fp16_model)
    with torch.no_grad():
        fp16_out = fp16_model(input_ids)
    fp16_logits = fp16_out.logits.detach().cpu()
    fp16_data = dict(fp16_hooks.data); fp16_hooks.remove()
    del fp16_model; torch.cuda.empty_cache()

    q_model = _load_model(quant_path); q_model.eval()
    q_hooks = _HookCollector(); q_hooks.register(q_model)
    with torch.no_grad():
        q_out = q_model(input_ids)
    q_logits = q_out.logits.detach().cpu()
    q_data = dict(q_hooks.data); q_hooks.remove()
    del q_model; torch.cuda.empty_cache()

    stats = []
    for name in sorted(set(fp16_data) & set(q_data)):
        mse = F.mse_loss(q_data[name]["output"].float(), fp16_data[name]["output"].float()).item()
        act = _activation_percentiles(fp16_data[name]["input"])
        stats.append(LayerStats(method=method, variant=variant, layer_idx=_layer_index(name),
                                module_family=_module_family(name), module_name=name, mse=mse, **act))
    logit_diag = _compute_logit_diagnostics(fp16_logits, q_logits, method, variant)
    return stats, logit_diag

# ════════════════════════════════════════════════════════════════
# Visualization
# ════════════════════════════════════════════════════════════════

def plot_accuracy_vs_bandwidth(eval_df, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = eval_df.dropna(subset=["score"])
    if df.empty: logger.warning("No valid scores for accuracy-vs-bandwidth plot."); return
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in df["method"].unique():
        sub = df[df["method"] == m]
        ax.scatter(sub["bytes_per_param_actual"], sub["score"], label=m.upper(), s=80, edgecolors="k", linewidths=0.5)
        for _, r in sub.iterrows():
            ax.annotate(r["variant"], (r["bytes_per_param_actual"], r["score"]), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Effective bytes / parameter"); ax.set_ylabel("Composite accuracy score")
    ax.set_title("Accuracy vs. Bandwidth (Sweet-Spot Plot)"); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "accuracy_vs_bandwidth.png", dpi=200); plt.close(fig)

def plot_accuracy_vs_deployment(eval_df, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = eval_df.dropna(subset=["score"])
    if df.empty: logger.warning("No valid scores for accuracy-vs-deployment plot."); return
    metrics = [("tok_s", "Tokens / sec"), ("ms_per_token", "ms / token"), ("peak_vram_gb", "Peak VRAM (GB)")]
    available = [m for m in metrics if m[0] in df.columns]
    if not available: return
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1: axes = [axes]
    for ax, (col, label) in zip(axes, available):
        for m in df["method"].unique():
            sub = df[df["method"] == m]
            ax.scatter(sub[col], sub["score"], label=m.upper(), s=80, edgecolors="k", linewidths=0.5)
        ax.set_xlabel(label); ax.set_ylabel("Composite score"); ax.legend(fontsize=8)
    fig.suptitle("Accuracy vs. Deployment Metrics", fontsize=13)
    fig.tight_layout(); fig.savefig(output_dir / "accuracy_vs_deployment.png", dpi=200); plt.close(fig)

def plot_ablation_sensitivity(eval_df, output_dir, task_col="score"):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in eval_df["method"].unique():
        sub = eval_df[eval_df["method"] == m]
        full = sub.loc[sub["variant"] == "full_quant", task_col]
        attn = sub.loc[sub["variant"] == "attn_only_quant", task_col]
        mlp  = sub.loc[sub["variant"] == "mlp_only_quant", task_col]
        if full.empty: continue
        fv = full.values[0]
        if fv is None: continue
        if not attn.empty and attn.values[0] is not None:
            rows.append({"method": m.upper(), "family": "Attn projections", "delta": fv - attn.values[0]})
        if not mlp.empty and mlp.values[0] is not None:
            rows.append({"method": m.upper(), "family": "MLP projections",  "delta": fv - mlp.values[0]})
    if not rows: logger.warning("Not enough ablation data for sensitivity plot."); return
    bar_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=bar_df, x="method", y="delta", hue="family", ax=ax)
    ax.set_ylabel("Score drop (full_quant - <family>_only)"); ax.set_xlabel("Quantization method")
    ax.set_title("Module-Family Sensitivity: MLP vs Attention")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    fig.tight_layout(); fig.savefig(output_dir / "ablation_sensitivity.png", dpi=200); plt.close(fig)

def plot_layer_mse_heatmap(layer_stats_df, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = layer_stats_df if isinstance(layer_stats_df, pd.DataFrame) else pd.read_parquet(str(layer_stats_df))
    for m in df["method"].unique():
        sub = df[df["method"] == m]
        pivot = sub.pivot_table(index="layer_idx", columns="module_family", values="mse", aggfunc="mean")
        if pivot.empty: continue
        fig, ax = plt.subplots(figsize=(6, max(8, len(pivot) * 0.3)))
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3, cbar_kws={"label": "MSE"})
        ax.set_title(f"Per-Layer MSE - {m.upper()}"); ax.set_ylabel("Transformer layer index"); ax.set_xlabel("Module family")
        fig.tight_layout(); fig.savefig(output_dir / f"layer_mse_{m}.png", dpi=200); plt.close(fig)

# ════════════════════════════════════════════════════════════════
# Orchestration helpers
# ════════════════════════════════════════════════════════════════

def ensure_fp16(cfg):
    fp16_dir = Path(cfg["paths"]["artifacts_dir"]) / "fp16"
    fp16_dir.mkdir(parents=True, exist_ok=True)
    if (fp16_dir / "config.json").exists():
        logger.info("FP16 reference already cached at %s", fp16_dir)
        return fp16_dir
    logger.info("Saving FP16 reference to %s ...", fp16_dir)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.float16, trust_remote_code=True)
    mdl.save_pretrained(str(fp16_dir)); tok.save_pretrained(str(fp16_dir))
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fp16_dir


def discover_artifacts(cfg, method):
    base = Path(cfg["paths"]["artifacts_dir"])
    found = []
    for md in sorted(base.iterdir()):
        if not md.is_dir() or md.name == "fp16":
            continue
        key = md.name.split("_")[0]
        if key != method:
            continue
        for vd in sorted(md.iterdir()):
            if not vd.is_dir() or not any(vd.iterdir()):
                continue
            found.append({"method": key, "variant": vd.name, "path": str(vd)})
    return found


def _compute_score(row, weights):
    """Compute weighted score from available (non-None, non-NaN) metrics."""
    valid = {k: row[k] for k in weights
             if row.get(k) is not None and not (isinstance(row.get(k), float) and math.isnan(row.get(k)))}
    if not valid:
        return None
    total_w = sum(weights[k] for k in valid)
    return sum(weights[k] * valid[k] for k in valid) / total_w


def evaluate_artifact(art, cfg, *, skip_tasks=None):
    skip = skip_tasks or set()
    weights = cfg["accuracy_weights"]
    row = {
        "method": art["method"],
        "variant": art["variant"],
        "bytes_per_param_actual": bytes_per_param_from_safetensors(art["path"]),
    }
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluating: {art['method']} / {art['variant']}", flush=True)
    print(f"{'='*60}", flush=True)
    if "perplexity" not in skip:
        try:
            print(">> Perplexity ...", flush=True)
            row["ppl"] = evaluate_perplexity(art["path"], cfg)
        except Exception:
            logger.exception("Perplexity failed"); row["ppl"] = None
    if "gsm8k" not in skip:
        try:
            print(">> GSM8K ...", flush=True)
            row["gsm8k"] = evaluate_gsm8k(art["path"], cfg)
        except Exception:
            logger.exception("GSM8K failed"); row["gsm8k"] = None
    if "math" not in skip:
        try:
            print(">> MATH ...", flush=True)
            row["math"] = evaluate_math(art["path"], cfg)
        except Exception:
            logger.exception("MATH failed"); row["math"] = None
    if "arc_challenge" not in skip:
        try:
            print(">> ARC-Challenge ...", flush=True)
            row["arc_challenge"] = evaluate_arc_challenge(art["path"], cfg)
        except Exception:
            logger.exception("ARC-Challenge failed"); row["arc_challenge"] = None
    if "gpqa" not in skip:
        try:
            print(">> GPQA ...", flush=True)
            row["gpqa"] = evaluate_gpqa(art["path"], cfg)
        except Exception:
            logger.exception("GPQA failed"); row["gpqa"] = None
    row["score"] = _compute_score(row, weights)
    if "benchmark" not in skip:
        try:
            print(">> Deployment benchmark ...", flush=True)
            b = benchmark_throughput(art["path"], cfg)
            row["tok_s"] = b.tokens_per_sec
            row["ms_per_token"] = b.ms_per_token
            row["peak_vram_gb"] = b.peak_vram_gb
        except Exception:
            logger.exception("Benchmark failed")
    print(f">> Done: {art['variant']} | score={row.get('score')}", flush=True)
    return row


def load_cached_eval(cfg, method):
    """Load previously saved eval results if they exist."""
    rd = Path(cfg["paths"]["results_dir"])
    cache_path = rd / f"{method}_eval.jsonl"
    if not cache_path.exists():
        return {}
    cached = {}
    with open(cache_path) as f:
        for line in f:
            row = json.loads(line)
            cached[row["variant"]] = row
    logger.info("Loaded %d cached eval results from %s", len(cached), cache_path)
    return cached


def _cache_is_complete(row, weights):
    """Return True only if all weighted eval metrics are present and valid."""
    for k in weights:
        v = row.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return False
    return True


def evaluate_all(artifacts, cfg, *, skip_tasks=None, method="rtn"):
    cached = load_cached_eval(cfg, method)
    weights = cfg["accuracy_weights"]
    results = []
    for i, art in enumerate(artifacts):
        variant = art["variant"]
        if variant in cached and _cache_is_complete(cached[variant], weights):
            print(f"\n[{i+1}/{len(artifacts)}] Using cached results for {variant}", flush=True)
            row = cached[variant]
            row["score"] = _compute_score(row, weights)
            results.append(row)
            continue
        elif variant in cached:
            missing = [k for k in weights if cached[variant].get(k) is None
                       or (isinstance(cached[variant].get(k), float) and math.isnan(cached[variant].get(k)))]
            print(f"\n[{i+1}/{len(artifacts)}] Re-evaluating {variant} (missing/NaN: {missing})", flush=True)
        row = evaluate_artifact(art, cfg, skip_tasks=skip_tasks)
        results.append(row)
        # Save incrementally after each eval so progress is not lost
        _save_incremental(results, cfg, method)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{i+1}/{len(artifacts)}] Memory after cleanup: "
              f"{torch.cuda.memory_allocated()/1e9:.1f}GB GPU, "
              f"{torch.cuda.memory_reserved()/1e9:.1f}GB reserved", flush=True)
    return pd.DataFrame(results)


def _save_incremental(results, cfg, method):
    """Save results after each variant so we can resume on failure."""
    rd = Path(cfg["paths"]["results_dir"]); rd.mkdir(parents=True, exist_ok=True)
    out = rd / f"{method}_eval.jsonl"
    with open(out, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    logger.info("Incremental save: %d rows to %s", len(results), out)


def run_all_diagnostics(artifacts, cfg):
    fp16_path = str(Path(cfg["paths"]["artifacts_dir"]) / "fp16")
    all_stats, all_logits = [], []
    for art in artifacts:
        logger.info("Diagnostics: %s / %s", art["method"], art["variant"])
        try:
            s, l = run_layer_diagnostics(fp16_path, art["path"], cfg, art["method"], art["variant"])
            all_stats.extend(s)
            all_logits.append(l)
        except Exception:
            logger.exception("Diagnostics failed for %s / %s", art["method"], art["variant"])
        gc.collect()
        torch.cuda.empty_cache()
    stats_df = pd.DataFrame([asdict(s) for s in all_stats]) if all_stats else pd.DataFrame()
    return stats_df, [asdict(d) for d in all_logits]


def save_eval_results(eval_df, cfg, method):
    rd = Path(cfg["paths"]["results_dir"]); rd.mkdir(parents=True, exist_ok=True)
    out = rd / f"{method}_eval.jsonl"
    with open(out, "w") as f:
        for row in eval_df.to_dict(orient="records"):
            f.write(json.dumps(row) + "\n")
    logger.info("Saved %d rows to %s", len(eval_df), out)
    return out


def save_diagnostics(layer_stats_df, logit_diags, cfg, method):
    rd = Path(cfg["paths"]["results_dir"]); rd.mkdir(parents=True, exist_ok=True)
    pq = rd / f"{method}_layer_stats.parquet"
    if not layer_stats_df.empty:
        layer_stats_df.to_parquet(str(pq), index=False)
    jp = rd / f"{method}_logit_diagnostics.json"
    with open(jp, "w") as f:
        json.dump(logit_diags, f, indent=2)
    return pq, jp


def get_plots_dir(cfg, method):
    d = Path(cfg["paths"]["plots_dir"]) / method
    d.mkdir(parents=True, exist_ok=True)
    return d


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("All functions defined. Setup complete.\n")

    # ── 2. FP16 Baseline and Calibration Data ──
    print("=" * 60)
    print("Step 2: FP16 Baseline and Calibration Data")
    print("=" * 60)
    fp16_dir = ensure_fp16(cfg)
    print(f"FP16 checkpoint: {fp16_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    get_calibration_data(cfg, tokenizer)
    print("Calibration data ready.\n")

    # ── 3. RTN Quantization ──
    print("=" * 60)
    print("Step 3: RTN Quantization")
    print("=" * 60)
    for variant in VARIANTS:
        print(f"\n{'='*60}")
        print(f"Quantizing: RTN / {variant}")
        print(f"{'='*60}")
        try:
            out = quantize_rtn(cfg, variant=variant)
            print(f"  -> Saved to {out}")
        except Exception as e:
            print(f"  !! FAILED: {e}")
    print("\nRTN quantization complete.\n")

    # ── 4. Evaluation ──
    print("=" * 60)
    print("Step 4: Evaluation")
    print("=" * 60)
    artifacts = discover_artifacts(cfg, METHOD)
    print(f"Found {len(artifacts)} RTN artifact(s):")
    for a in artifacts:
        print(f"  {a['variant']:20s}  {a['path']}")

    eval_df = evaluate_all(artifacts, cfg, method=METHOD)
    save_eval_results(eval_df, cfg, METHOD)
    print("\nEvaluation results:")
    print(eval_df.to_string())
    print()

    # ── 5. Ablation Study ──
    print("=" * 60)
    print("Step 5: Ablation Study")
    print("=" * 60)
    plots_dir = get_plots_dir(cfg, METHOD)
    plot_ablation_sensitivity(eval_df, plots_dir)
    ablation_png = plots_dir / "ablation_sensitivity.png"
    if ablation_png.exists():
        print(f"Ablation sensitivity plot saved to {ablation_png}")
    else:
        print("Not enough ablation data to generate the sensitivity plot.")
    print()

    # ── 6. Layer Diagnostics ──
    print("=" * 60)
    print("Step 6: Layer Diagnostics")
    print("=" * 60)
    layer_stats_df, logit_diags = run_all_diagnostics(artifacts, cfg)
    save_diagnostics(layer_stats_df, logit_diags, cfg, METHOD)
    print(f"Layer stats shape: {layer_stats_df.shape}")
    print(f"Logit diagnostics: {len(logit_diags)} entries")
    if logit_diags:
        print(pd.DataFrame(logit_diags).to_string())
    print()

    # ── 7. Accuracy vs Bandwidth ──
    print("=" * 60)
    print("Step 7: Accuracy vs Bandwidth")
    print("=" * 60)
    plot_accuracy_vs_bandwidth(eval_df, plots_dir)
    bw_png = plots_dir / "accuracy_vs_bandwidth.png"
    if bw_png.exists():
        print(f"Accuracy vs bandwidth plot saved to {bw_png}")
    print()

    # ── 8. Deployment Metrics ──
    print("=" * 60)
    print("Step 8: Deployment Metrics")
    print("=" * 60)
    plot_accuracy_vs_deployment(eval_df, plots_dir)
    dep_png = plots_dir / "accuracy_vs_deployment.png"
    if dep_png.exists():
        print(f"Deployment metrics plot saved to {dep_png}")
    print()

    # ── 9. Layer MSE Heatmap ──
    print("=" * 60)
    print("Step 9: Layer MSE Heatmap")
    print("=" * 60)
    if not layer_stats_df.empty:
        plot_layer_mse_heatmap(layer_stats_df, plots_dir)
        mse_png = plots_dir / f"layer_mse_{METHOD}.png"
        if mse_png.exists():
            print(f"Layer MSE heatmap saved to {mse_png}")
    else:
        print("No layer stats data available for heatmap.")
    print()

    # ── 10. Summary ──
    print("=" * 60)
    print("Step 10: Summary")
    print("=" * 60)
    summary_cols = [
        "variant", "bytes_per_param_actual", "ppl",
        "gsm8k", "math", "arc_challenge", "gpqa", "score",
        "tok_s", "ms_per_token", "peak_vram_gb",
    ]
    available = [c for c in summary_cols if c in eval_df.columns]
    print(eval_df[available].round(3).to_string())
    print()

    print(f"All results saved to: {Path(cfg['paths']['results_dir'])}")
    print(f"All plots  saved to: {plots_dir}")
    print("\nRTN notebook complete.")


if __name__ == "__main__":
    main()
