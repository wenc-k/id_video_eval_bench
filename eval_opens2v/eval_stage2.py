import os
import re
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    ensure_dir(dst.parent)
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def mean_valid(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def make_case_key(idx: int, case: Dict[str, Any]) -> str:
    stem = Path(case["video_path"]).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return f"case_{idx:04d}_{stem}"


def normalize_human_domain_case(case: Dict[str, Any]) -> Dict[str, Optional[float]]:
    aes = safe_float(case.get("aes_score"))
    motion_smoothness = safe_float(case.get("motion_smoothness"))
    motion_amplitude = safe_float(case.get("motion_fb"))
    facesim_cur = safe_float(case.get("cur_score"))
    gme_score = safe_float(case.get("gme_score"))

    natural_score = safe_float(case.get("natural_score"))
    if natural_score is None:
        natural_score = safe_float(case.get("naturalscore"))
    if natural_score is None:
        natural_score = mean_valid([safe_float(case.get(f"naturalscore_{i}")) for i in (1, 2, 3)])

    if aes is not None:
        aes = max(min(aes, 7.0), 4.0)
        aes = (aes - 4.0) / 3.0
    if motion_smoothness is not None:
        motion_smoothness = min(abs(motion_smoothness), 1.0)
    if motion_amplitude is not None:
        motion_amplitude = min(abs(motion_amplitude), 1.0)
    if facesim_cur is not None:
        facesim_cur = min(facesim_cur, 1.0)
    if gme_score is not None:
        gme_score = min(gme_score, 1.0)
    if natural_score is not None:
        natural_score = max(min(natural_score, 5.0), 1.0)
        natural_score = (natural_score - 1.0) / 4.0

    return {
        "aes_score_norm": aes,
        "motion_smoothness_norm": motion_smoothness,
        "motion_amplitude_norm": motion_amplitude,
        "facesim_cur_norm": facesim_cur,
        "gme_score_norm": gme_score,
        "natural_score_norm": natural_score,
    }


def human_domain_total_from_norm(norm: Dict[str, Optional[float]]) -> Optional[float]:
    keys = [
        "aes_score_norm",
        "motion_smoothness_norm",
        "motion_amplitude_norm",
        "facesim_cur_norm",
        "gme_score_norm",
        "natural_score_norm",
    ]
    if any(norm.get(k) is None for k in keys):
        return None
    return (
        0.18 * norm["aes_score_norm"]
        + 0.09 * norm["motion_smoothness_norm"]
        + 0.03 * norm["motion_amplitude_norm"]
        + 0.25 * norm["facesim_cur_norm"]
        + 0.15 * norm["gme_score_norm"]
        + 0.30 * norm["natural_score_norm"]
    )


def aggregate_by_id(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for case in case_results:
        grouped[str(case.get("id_index", "unknown"))].append(case)

    out = {}
    for id_key, cases in sorted(grouped.items(), key=lambda x: str(x[0])):
        natural_vals = []
        for c in cases:
            val = safe_float(c.get("natural_score"))
            if val is None:
                val = safe_float(c.get("naturalscore"))
            if val is None:
                val = mean_valid([safe_float(c.get(f"naturalscore_{i}")) for i in (1, 2, 3)])
            natural_vals.append(val)

        row = {
            "id_index": id_key,
            "num_cases": len(cases),
            "ref_img_path": sorted({c.get("ref_img_path", "") for c in cases}),
            "aes_score": mean_valid([safe_float(c.get("aes_score")) for c in cases]),
            "motion_smoothness": mean_valid([safe_float(c.get("motion_smoothness")) for c in cases]),
            "motion_fb": mean_valid([safe_float(c.get("motion_fb")) for c in cases]),
            "cur_score": mean_valid([safe_float(c.get("cur_score")) for c in cases]),
            "arc_score": mean_valid([safe_float(c.get("arc_score")) for c in cases]),
            "gme_score": mean_valid([safe_float(c.get("gme_score")) for c in cases]),
            "natural_score": mean_valid(natural_vals),
        }
        norm = normalize_human_domain_case(row)
        row.update(norm)
        row["total_score"] = human_domain_total_from_norm(norm)
        out[id_key] = row
    return out


def aggregate_summary(case_results: List[Dict[str, Any]], id_results: Dict[str, Any]) -> Dict[str, Any]:
    natural_vals = []
    for c in case_results:
        val = safe_float(c.get("natural_score"))
        if val is None:
            val = safe_float(c.get("naturalscore"))
        if val is None:
            val = mean_valid([safe_float(c.get(f"naturalscore_{i}")) for i in (1, 2, 3)])
        natural_vals.append(val)

    avg_case = {
        "aes_score": mean_valid([safe_float(c.get("aes_score")) for c in case_results]),
        "motion_smoothness": mean_valid([safe_float(c.get("motion_smoothness")) for c in case_results]),
        "motion_fb": mean_valid([safe_float(c.get("motion_fb")) for c in case_results]),
        "cur_score": mean_valid([safe_float(c.get("cur_score")) for c in case_results]),
        "arc_score": mean_valid([safe_float(c.get("arc_score")) for c in case_results]),
        "gme_score": mean_valid([safe_float(c.get("gme_score")) for c in case_results]),
        "natural_score": mean_valid(natural_vals),
    }
    norm = normalize_human_domain_case(avg_case)
    return {
        "num_cases": len(case_results),
        "num_ids": len(id_results),
        "human_domain_weights": {
            "aes_score": 0.18,
            "motion_smoothness": 0.09,
            "motion_amplitude": 0.03,
            "facesim_cur": 0.25,
            "gme_score": 0.15,
            "natural_score": 0.30,
        },
        "case_average_raw": avg_case,
        "case_average_norm": norm,
        "total_score": human_domain_total_from_norm(norm),
    }


class MotionSmoothnessMerger:
    def __init__(self, args):
        self.args = args
        self.work_dir = Path(args.output_dir) / "_merge_ms_work"
        self.video_dir = (
            Path(args.reuse_input_video_dir)
            if args.reuse_input_video_dir is not None
            else (self.work_dir / "input_videos")
        )
        self.ms_out_dir = self.work_dir / "motion_smoothness"

    def prepare_videos(self, input_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        meta = []

        # 复用已有 input_videos：不再重复拷贝，只校验文件是否存在
        if self.args.reuse_input_video_dir is not None:
            self.video_dir = Path(self.args.reuse_input_video_dir)
            if not self.video_dir.exists():
                raise FileNotFoundError(f"reuse_input_video_dir not found: {self.video_dir}")

            for idx, case in enumerate(input_cases):
                for req in ("id_index", "ref_img_path", "video_path", "prompt"):
                    if req not in case:
                        raise KeyError(f"Missing key `{req}` in case index {idx}")

                case_key = make_case_key(idx, case)
                expected_video = self.video_dir / f"{case_key}.mp4"
                if not expected_video.exists():
                    raise FileNotFoundError(f"Missing reused video: {expected_video}")

                meta.append({"index": idx, "case_key": case_key})

            return meta

        # 默认逻辑：重新 link/copy 一份 input_videos
        ensure_dir(self.video_dir)
        for idx, case in enumerate(input_cases):
            for req in ("id_index", "ref_img_path", "video_path", "prompt"):
                if req not in case:
                    raise KeyError(f"Missing key `{req}` in case index {idx}")

            video_src = Path(case["video_path"])
            if not video_src.exists():
                raise FileNotFoundError(f"Video not found: {video_src}")

            case_key = make_case_key(idx, case)
            video_dst = self.video_dir / f"{case_key}.mp4"
            safe_link_or_copy(video_src, video_dst)
            meta.append({"index": idx, "case_key": case_key})

        return meta

    def run_motion_smoothness(self) -> Path:
        ensure_dir(self.ms_out_dir)
        cmd = [
            sys.executable,
            str(Path(self.args.motion_smoothness_script)),
            "--input_video_folder", str(self.video_dir),
            "--output_json_folder", str(self.ms_out_dir),
            "--model_path", self.args.motion_smoothness_model_path,
            "--window_size", str(self.args.motion_smoothness_window_size),
        ]
        print(f"[MotionSmoothnessMerger] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return self.ms_out_dir / "motion_smoothness.json"

    def run(self):
        input_cases = load_json(Path(self.args.input_json))
        existing_cases = load_json(Path(self.args.existing_case_json))
        if len(input_cases) != len(existing_cases):
            raise ValueError("input_json and existing_case_json must have the same number of cases")

        meta = self.prepare_videos(input_cases)
        ms_json_path = self.run_motion_smoothness()
        ms_map = load_json(ms_json_path)

        merged_cases = []
        for case, m in zip(existing_cases, meta):
            row = dict(case)
            ms_row = ms_map.get(m["case_key"], {})
            if isinstance(ms_row, dict):
                row.update(ms_row)
            norm = normalize_human_domain_case(row)
            row.update(norm)
            row["total_score"] = human_domain_total_from_norm(norm)
            merged_cases.append(row)

        id_results = aggregate_by_id(merged_cases)
        summary = aggregate_summary(merged_cases, id_results)

        case_out = Path(self.args.output_dir) / "case_results_with_qalign_ms.json"
        id_out = Path(self.args.output_dir) / "id_results_with_qalign_ms.json"
        summary_out = Path(self.args.output_dir) / "summary_human_domain_total.json"

        dump_json(merged_cases, case_out)
        dump_json(id_results, id_out)
        dump_json(summary, summary_out)

        print(json.dumps({
            "motion_smoothness_json": str(ms_json_path),
            "case_results": str(case_out),
            "id_results": str(id_out),
            "summary_results": str(summary_out),
        }, indent=2, ensure_ascii=False))


def build_parser():
    p = argparse.ArgumentParser(description="Run qalign motion_smoothness then merge with existing metrics using Human-Domain weights.")
    p.add_argument("--input_json", type=str, required=True, help="Original unified input case json.")
    p.add_argument("--existing_case_json", type=str, required=True, help="Existing case_results_v2.json containing other metrics.")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--motion_smoothness_script", type=str, default=str(SCRIPT_DIR / "get_motion_smoothness.py"))
    p.add_argument("--motion_smoothness_model_path", type=str, default="q-future/one-align")
    p.add_argument("--motion_smoothness_window_size", type=int, default=3)

    p.add_argument(
        "--reuse_input_video_dir",
        type=str,
        default=None,
        help="Reuse an existing _v2_work/input_videos directory instead of copying videos again.",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    merger = MotionSmoothnessMerger(args)
    merger.run()


if __name__ == "__main__":
    main()
