import os
import json
import base64
import argparse
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt


REFERENCE_ID_SYSTEM_PROMPT = r'''You are a visual identity extraction engine for identity-preserving text-to-video generation.

Your task is to analyze a single reference image of one person, usually a face crop, and extract a compact, conservative, machine-readable summary of the person's stable facial identity attributes.

This summary will later be used for prompt conflict scoring and prompt augmentation. Therefore, your output must prioritize stable, visually grounded identity cues that are safe to inject into a text prompt.

Core objective:
Extract only stable facial identity attributes that are clearly supported by the image and are useful for preserving identity during generation.

Rules:
1. Only use information that is visually grounded in the image.
2. Do NOT guess attributes that are unclear, ambiguous, hidden, or not visible.
3. If an attribute is not reliable enough for prompt injection, mark it as "unknown".
4. Do NOT infer clothing, background, body pose, personality, occupation, ethnicity, nationality, or other non-facial attributes.
5. Use neutral visual language such as "masculine-presenting", "feminine-presenting", "young adult", "middle-aged adult", etc.
6. Focus only on stable facial identity cues that are likely to remain valid across different scenes and prompts.
7. Distinguish between:
   - high-confidence stable attributes that are safe to inject into prompts;
   - uncertain or weak attributes that should not be injected.
8. Return JSON only. No markdown, no explanations, no extra text.

Output schema:
{
  "stable_id_summary": {
    "compact_summary": "A short compact summary of high-confidence stable facial identity attributes only.",
    "safe_injection_attributes": {
      "gender_presentation": "string or unknown",
      "age_impression": "string or unknown",
      "hair_color": "string or unknown",
      "hair_style": "string or unknown",
      "facial_hair": "string or unknown",
      "glasses": "yes / no / unknown",
      "other_high_confidence_facial_traits": [
        "string"
      ]
    },
    "do_not_inject_attributes": [
      {
        "attribute": "string",
        "reason": "unclear / weak / temporary / not visible"
      }
    ]
  },
  "reference_image_assessment": {
    "frontality": "frontal / near-frontal / side / unclear",
    "face_visibility": "clear / partial / occluded / unclear",
    "image_quality": "high / medium / low",
    "identity_summary_reliability": "high / medium / low"
  }
}

Additional constraints:
- "compact_summary" must be under 30 words.
- "other_high_confidence_facial_traits" should be short and sparse; include only highly reliable traits.
- Prefer "unknown" over guessing.
- Do not include low-confidence details just to make the summary richer.
- The output should be optimized for downstream prompt augmentation under the goal of preserving text consistency while improving identity consistency.
'''


PROMPT_AUG_SYSTEM_PROMPT = r'''You are a prompt adaptation engine for identity-preserving text-to-video generation.

Your input will contain:
1. a raw user prompt, and
2. a compact stable facial identity summary extracted from a reference image.

Your task is to decide whether prompt augmentation is necessary, how strong it should be, and then output a single processed prompt.

Primary goal:
Improve identity consistency while preserving the original prompt semantics and text consistency as much as possible.

Core principles:
1. The raw prompt is the primary carrier of user intent and should be preserved as much as possible.
2. The reference identity summary is only a supporting constraint for identity preservation.
3. Only make the minimum necessary changes.
4. Never add unnecessary scene, action, camera, or style details.
5. Only inject high-confidence stable facial identity attributes from the reference summary.
6. If the prompt contains facial identity attributes that clearly conflict with the reference summary, resolve them conservatively.
7. If the prompt contains indirect interference factors that may reduce identity consistency (such as helmet, mask, sunglasses, tiny face, far shot, full body, side view, back view, heavy motion, strong occlusion, crowding), weaken them only when necessary.
8. Do not remove core prompt semantics unless they strongly threaten identity preservation.
9. Return JSON only. No markdown, no explanation, no extra text.

You must internally consider three factors:
- identity attribute conflict
- indirect interference risk
- text fragility

But do NOT output these three factors separately.

Instead, output one final integrated score:

prompt_aug_score:
- 0 = none
- 1 = light
- 2 = medium
- 3 = safe

Meaning of each level:
- 0 (none): the raw prompt should remain unchanged because augmentation is unnecessary or not worth the risk.
- 1 (light): only minimal identity-supportive adjustment is needed; wording should stay extremely close to the raw prompt.
- 2 (medium): moderate prompt augmentation is needed, including conservative resolution of identity conflicts and light protective wording.
- 3 (safe): strong but still conservative augmentation is needed; resolve clear identity conflicts and cautiously weaken only the highest-risk interference factors while preserving core semantics.

Scoring guidance:
- Increase the score when the prompt clearly conflicts with the reference identity summary.
- Increase the score when the prompt contains strong indirect interference factors that may hurt identity preservation.
- Decrease the score when the prompt is text-fragile and likely to lose important semantics under aggressive rewriting.
- Prefer lower scores when the benefit of rewriting is uncertain.
- When in doubt, preserve the raw prompt more strongly.

Routing rule:
- If prompt_aug_score = 0, processed_prompt must be exactly the raw prompt.
- If prompt_aug_score = 1, processed_prompt must be only minimally edited.
- If prompt_aug_score = 2, processed_prompt may resolve clear conflicts and add light identity-preserving support.
- If prompt_aug_score = 3, processed_prompt may more conservatively protect identity, but must still preserve the raw prompt's main subject, main action, main scene, and main intent.

Output schema:
{
  "prompt_aug_score": 0,
  "prompt_aug_strength": "none / light / medium / safe",
  "processed_prompt": "string"
}

Additional constraints:
- "prompt_aug_strength" must exactly match the score:
  - 0 -> "none"
  - 1 -> "light"
  - 2 -> "medium"
  - 3 -> "safe"
- Keep the processed prompt concise.
- Do not produce multiple prompt variants.
- Do not output any extra analysis.
- The output should be optimized for downstream inference-time prompt enhancement under the objective:
  maximize identity consistency without significantly harming text consistency.
'''


def _image_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext)
    if mime is None:
        raise ValueError(f"Unsupported image extension: {ext}")

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    return text


def _parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = _clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON. Raw output:\n{cleaned}") from e


@retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(5))
def _chat_completion_json(
    messages,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response content from model.")
    return _parse_json_response(content)


def extract_reference_stable_id_summary(
    image_path: str,
    api_key: Optional[str] = None,
    model_name: str = "gpt-5.4",
    base_url: Optional[str] = None,
    extra_user_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    image_data_url = _image_to_data_url(image_path)

    user_parts = [
        {"type": "text", "text": "Analyze this reference image and return the JSON object only."},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]
    if extra_user_instruction:
        user_parts.append({"type": "text", "text": extra_user_instruction})

    messages = [
        {"role": "system", "content": REFERENCE_ID_SYSTEM_PROMPT},
        {"role": "user", "content": user_parts},
    ]
    return _chat_completion_json(
        messages=messages,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )


def adapt_prompt_with_reference_id(
    raw_prompt: str,
    stable_id_summary: Dict[str, Any],
    api_key: Optional[str] = None,
    model_name: str = "gpt-5.4",
    base_url: Optional[str] = None,
    extra_user_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    user_text = (
        f"Raw prompt:\n{raw_prompt}\n\n"
        f"Reference stable ID summary:\n{json.dumps(stable_id_summary, ensure_ascii=False, indent=2)}\n"
    )
    if extra_user_instruction:
        user_text += f"\nAdditional instruction:\n{extra_user_instruction}\n"

    messages = [
        {"role": "system", "content": PROMPT_AUG_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return _chat_completion_json(
        messages=messages,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _write_text_file(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _find_prompt_txt_files(input_dir: str) -> List[str]:
    txt_files: List[str] = []
    for name in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, name)
        if not os.path.isfile(full_path):
            continue
        if not name.lower().endswith(".txt"):
            continue
        if name.lower().endswith("_processed.txt"):
            continue
        txt_files.append(full_path)
    return txt_files


def process_directory(
    input_dir: str,
    api_key: Optional[str] = None,
    model_name: str = "gpt-5.4",
    base_url: Optional[str] = None,
    extract_instruction: Optional[str] = None,
    adapt_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    image_path = os.path.join(input_dir, "image.png")
    if not os.path.isfile(image_path):
        raise ValueError(f"Expected image file not found: {image_path}")

    prompt_txt_files = _find_prompt_txt_files(input_dir)
    if not prompt_txt_files:
        raise ValueError(
            f"No prompt .txt files found in directory: {input_dir} "
            f"(files ending with _processed.txt are ignored)"
        )

    stable_id_summary = extract_reference_stable_id_summary(
        image_path=image_path,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        extra_user_instruction=extract_instruction,
    )
    stable_id_json_path = os.path.join(input_dir, "stable_id_summary.json")
    save_json(stable_id_summary, stable_id_json_path)

    results: List[Dict[str, Any]] = []
    for txt_path in prompt_txt_files:
        raw_prompt = _read_text_file(txt_path)
        adapted = adapt_prompt_with_reference_id(
            raw_prompt=raw_prompt,
            stable_id_summary=stable_id_summary,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            extra_user_instruction=adapt_instruction,
        )

        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        prompt_aug_json_path = os.path.join(input_dir, f"{base_name}_prompt_aug.json")
        processed_txt_path = os.path.join(input_dir, f"{base_name}_processed.txt")

        save_json(adapted, prompt_aug_json_path)
        processed_prompt = str(adapted.get("processed_prompt", "")).strip()
        _write_text_file(processed_txt_path, processed_prompt)

        results.append(
            {
                "input_txt": txt_path,
                "prompt_aug_json": prompt_aug_json_path,
                "processed_prompt_txt": processed_txt_path,
            }
        )

    return {
        "input_dir": input_dir,
        "image_path": image_path,
        "stable_id_json": stable_id_json_path,
        "num_prompts": len(prompt_txt_files),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference ID extraction + prompt augmentation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_dir = subparsers.add_parser(
        "process_dir",
        help="Process one directory containing image.png and one or more prompt .txt files",
    )
    p_dir.add_argument("--input_dir", type=str, required=True)
    p_dir.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"))
    p_dir.add_argument("--model_name", type=str, default="gpt-5.4")
    p_dir.add_argument("--base_url", type=str, default='https://api.zhizengzeng.com/v1')
    p_dir.add_argument("--extract_instruction", type=str, default=None)
    p_dir.add_argument("--adapt_instruction", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("API key is required. Pass --api_key or set OPENAI_API_KEY.")

    result = process_directory(
        input_dir=args.input_dir,
        api_key=args.api_key,
        model_name=args.model_name,
        base_url=args.base_url,
        extract_instruction=args.extract_instruction,
        adapt_instruction=args.adapt_instruction,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
