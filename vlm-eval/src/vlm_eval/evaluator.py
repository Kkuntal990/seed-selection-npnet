"""VLM evaluation of generated images for seed mining."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from tqdm.auto import tqdm

from seed_mining.io_utils import append_metadata_jsonl, image_path, load_existing_keys
from seed_mining.logging_utils import ThroughputTracker, setup_logging
from seed_mining.prompts import Prompt, build_all_prompts

if TYPE_CHECKING:
    from vlm_eval.eval_config import EvalConfig

logger = logging.getLogger("vlm_eval.evaluator")

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
    model_id: str,
    quantize: str | None = "4bit",
) -> tuple[Any, Any]:
    """Load a Qwen2.5-VL model and processor."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, processor


def _query_vlm(
    model: Any,
    processor: Any,
    image: Image.Image,
    question: str,
) -> str:
    """Send a single image+question to the VLM and return the response text."""
    from qwen_vl_utils import process_vision_info

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    # Trim prompt tokens — keep only generated tokens
    generated = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids, strict=True)]
    response = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return str(response[0]).strip()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def _evaluate_numeracy(
    model: Any,
    processor: Any,
    img: Image.Image,
    prompt: Prompt,
    seed: int,
    rel_path: str,
) -> dict[str, Any]:
    """Evaluate one numeracy image."""
    objects = prompt.metadata["object_plural"]
    question = NUMERACY_TEMPLATE.format(objects=objects)
    response = _query_vlm(model, processor, img, question)

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
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _evaluate_spatial(
    model: Any,
    processor: Any,
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
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
    response_1 = _query_vlm(model, processor, img, SPATIAL_DESCRIBE)
    record["vlm_question_1"] = SPATIAL_DESCRIBE
    record["vlm_response_1"] = response_1

    # Step 2: verify relation
    question_2 = SPATIAL_VERIFY.format(obj_a=obj_a, relation=relation, obj_b=obj_b)
    response_2 = _query_vlm(model, processor, img, question_2)
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
    model, processor = load_vlm(config.vlm_model_id, config.quantize)
    logger.info("VLM loaded")

    pbar = tqdm(total=total, desc="VLM eval", unit="img")

    for category, prompts in categories:
        responses_path = config.eval_out_dir / "responses" / f"{category}_responses.jsonl"

        # Resume: find already-evaluated (seed, prompt_id) pairs
        existing = load_existing_keys(responses_path)
        logger.info(
            "[%s] %d already evaluated, %d total",
            category,
            len(existing),
            len(seeds) * len(prompts),
        )

        for seed in seeds:
            todo = [p for p in prompts if (seed, p.prompt_id) not in existing]
            skipped = len(prompts) - len(todo)

            if not todo:
                tracker.update(len(prompts))
                pbar.update(len(prompts))
                continue

            tracker.update(skipped)
            pbar.update(skipped)
            pbar.set_postfix(category=category, seed=seed)

            records: list[dict[str, Any]] = []
            for prompt in todo:
                img_path = image_path(
                    config.images_dir, category, seed, prompt.prompt_id, config.image_format
                )
                if not img_path.exists():
                    logger.warning("Image not found: %s", img_path)
                    tracker.update(1)
                    pbar.update(1)
                    continue

                img = Image.open(img_path).convert("RGB")
                rel = str(img_path.relative_to(config.images_dir))

                if category == "numeracy":
                    rec = _evaluate_numeracy(model, processor, img, prompt, seed, rel)
                else:
                    rec = _evaluate_spatial(model, processor, img, prompt, seed, rel)

                records.append(rec)
                tracker.update(1)
                pbar.update(1)

            if records:
                append_metadata_jsonl(responses_path, records)

            logger.info("Seed %d [%s] done | %s", seed, category, tracker.summary_line())

    pbar.close()
    logger.info("Evaluation complete | %s", tracker.summary_line())
