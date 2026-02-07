"""VLM evaluation of generated images for seed mining."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from seed_mining.io_utils import append_metadata_jsonl, image_path, load_existing_keys
from seed_mining.logging_utils import ThroughputTracker, setup_logging
from seed_mining.prompts import Prompt, build_all_prompts

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from seed_mining.eval_config import EvalConfig

logger = logging.getLogger("seed_mining.evaluator")

# ---------------------------------------------------------------------------
# VLM prompts (matching paper)
# ---------------------------------------------------------------------------

NUMERACY_TEMPLATE = "Answer in one sentence: How many {objects} are in this image?"
SPATIAL_DESCRIBE = "Describe the positions of the objects in the image in one sentence"
SPATIAL_VERIFY = (
    "Answer with yes or no: Is there a {obj_a} positioned {relation} a {obj_b} in the image?"
)

# Pattern to extract obj_a, relation, obj_b from spatial prompts like
# "A pineapple on the left of a monkey"
_SPATIAL_RE = re.compile(
    r"^[Aa]n?\s+(.+?)\s+(on the (?:left|right|top) of|on top of|under)\s+[Aa]n?\s+(.+)$"
)


def _parse_spatial_prompt(text: str) -> tuple[str, str, str] | None:
    """Extract (obj_a, relation, obj_b) from a spatial prompt string."""
    m = _SPATIAL_RE.match(text)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_vlm(
    model_id: str, quantize: str | None = "4bit", device: str = "cuda"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a CogVLM2 model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()  # type: ignore[union-attr]
    return model, tokenizer  # type: ignore[return-value]


def _query_vlm(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    image: Image.Image,
    question: str,
) -> str:
    """Send a single image+question to the VLM and return the response text."""
    input_by_model = model.build_conversation_input_ids(  # type: ignore[union-attr]
        tokenizer,
        query=question,
        images=[image],
    )
    inputs = {
        "input_ids": input_by_model["input_ids"].unsqueeze(0).to(model.device),  # type: ignore[union-attr]
        "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(model.device),  # type: ignore[union-attr]
        "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(model.device),  # type: ignore[union-attr]
        "images": [[input_by_model["images"][0].to(model.device, dtype=torch.bfloat16)]],  # type: ignore[union-attr]
    }

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)  # type: ignore[union-attr]

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)  # type: ignore[union-attr]
    return response.strip()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def _evaluate_numeracy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    img: Image.Image,
    prompt: Prompt,
    seed: int,
    rel_path: str,
) -> dict[str, Any]:
    """Evaluate one numeracy image."""
    objects = prompt.metadata["object_plural"]
    question = NUMERACY_TEMPLATE.format(objects=objects)
    response = _query_vlm(model, tokenizer, img, question)

    return {
        "seed": seed,
        "prompt_id": prompt.prompt_id,
        "category": "numeracy",
        "prompt_text": prompt.text,
        "count_target": prompt.metadata["count_target"],
        "object_plural": objects,
        "vlm_question": question,
        "vlm_response": response,
        "image_path": rel_path,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def _evaluate_spatial(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    img: Image.Image,
    prompt: Prompt,
    seed: int,
    rel_path: str,
) -> dict[str, Any]:
    """Evaluate one spatial image with two-step prompting."""
    raw = prompt.metadata["spatial_prompt_raw"]
    parsed = _parse_spatial_prompt(raw)

    record: dict[str, Any] = {
        "seed": seed,
        "prompt_id": prompt.prompt_id,
        "category": "spatial",
        "prompt_text": prompt.text,
        "spatial_prompt_raw": raw,
        "image_path": rel_path,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }

    if parsed is None:
        logger.warning("Could not parse spatial prompt: %r", raw)
        record["vlm_response_1"] = ""
        record["vlm_response_2"] = ""
        record["parse_error"] = True
        return record

    obj_a, relation, obj_b = parsed
    record["obj_a"] = obj_a
    record["obj_b"] = obj_b
    record["expected_relation"] = relation

    # Step 1: describe positions
    response_1 = _query_vlm(model, tokenizer, img, SPATIAL_DESCRIBE)
    record["vlm_question_1"] = SPATIAL_DESCRIBE
    record["vlm_response_1"] = response_1

    # Step 2: verify relation
    question_2 = SPATIAL_VERIFY.format(obj_a=obj_a, relation=relation, obj_b=obj_b)
    response_2 = _query_vlm(model, tokenizer, img, question_2)
    record["vlm_question_2"] = question_2
    record["vlm_response_2"] = response_2

    return record


def run_evaluation(config: EvalConfig) -> None:
    """Run VLM evaluation on generated images."""
    log_dir = config.eval_out_dir / "logs"
    setup_logging(log_dir, rank=0)

    logger.info("Building prompts (split=%s) ...", config.split)
    numeracy, spatial = build_all_prompts(
        config.prompt_dataset_dir,
        config.split,
        num_objects=config.num_objects,
        num_settings=config.num_settings,
        append_background_to_spatial=config.append_background_to_spatial,
    )

    seeds = config.seeds
    categories: list[tuple[str, list[Prompt]]] = []
    if config.category in ("numeracy", "all"):
        categories.append(("numeracy", numeracy))
    if config.category in ("spatial", "all"):
        categories.append(("spatial", spatial))

    # Count total work
    total = sum(len(seeds) * len(prompts) for _, prompts in categories)
    tracker = ThroughputTracker(total=total)

    # Load VLM
    logger.info("Loading VLM: %s (quantize=%s) ...", config.vlm_model_id, config.quantize)
    model, tokenizer = load_vlm(config.vlm_model_id, config.quantize)
    logger.info("VLM loaded")

    for category, prompts in categories:
        responses_path = config.eval_out_dir / "responses" / f"{category}_responses.jsonl"

        # Resume: find already-evaluated (seed, prompt_id) pairs
        existing = load_existing_keys(responses_path)
        logger.info("[%s] %d already evaluated, %d total", category, len(existing), len(seeds) * len(prompts))

        for seed in seeds:
            todo = [p for p in prompts if (seed, p.prompt_id) not in existing]
            skipped = len(prompts) - len(todo)

            if not todo:
                tracker.update(len(prompts))
                continue

            tracker.update(skipped)
            logger.info("Seed %d [%s]: evaluating %d images (%d skipped)", seed, category, len(todo), skipped)

            records: list[dict[str, Any]] = []
            for prompt in todo:
                img_path = image_path(
                    config.images_dir, category, seed, prompt.prompt_id, config.image_format
                )
                if not img_path.exists():
                    logger.warning("Image not found: %s", img_path)
                    tracker.update(1)
                    continue

                img = Image.open(img_path).convert("RGB")
                rel = str(img_path.relative_to(config.images_dir))

                if category == "numeracy":
                    rec = _evaluate_numeracy(model, tokenizer, img, prompt, seed, rel)
                else:
                    rec = _evaluate_spatial(model, tokenizer, img, prompt, seed, rel)

                records.append(rec)
                tracker.update(1)

            if records:
                append_metadata_jsonl(responses_path, records)

            logger.info("Seed %d [%s] done | %s", seed, category, tracker.summary_line())

    logger.info("Evaluation complete | %s", tracker.summary_line())
