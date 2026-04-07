# ID Video Evaluation System

A unified evaluation toolkit for **ID-preserving video generation**.  
The input of each test case is a **reference image (or reference image folder)**, a **text prompt**, and a **generated video**. The toolkit evaluates three high-level dimensions:

- **ID Consistency**
- **Text Consistency**
- **Video Quality**

The evaluation script reads a JSON file of test cases and produces three output JSON files:

1. **case-level results** for all samples
2. **id-level results** aggregated over all cases belonging to the same identity
3. **overall summary results** aggregated over the full benchmark

---

## 1. Environment Installation

### 1.1 Create environment

```bash
conda create -n id_video_eval python=3.10 -y
conda activate id_video_eval
```

### 1.2 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Download Model Weights from Hugging Face

All model weights used by this project are hosted on Hugging Face.

wenc-k/id_video_eval_bench
https://huggingface.co/datasets/wenc-k/id_video_eval_bench


### Expected local paths

After downloading, the following paths should be valid:

- `dino-vits16`
- `raft_model/models/raft-things.pth`
- `amt_model/AMT-S.yaml`
- `amt_model/amt-s.pth`
- `ip`

---

## 3. Input Format

The script expects a JSON file containing a **list of test cases**.

- `id_index`: identity index
- `ref_img_path`: path to a single reference image or a folder of reference images
- `video_path`: path to a generated video file or a folder of frames
- `prompt`: text prompt used to generate the video

---

## 4. Output Files

The script writes **three JSON files** to `output_dir`:

- `case_results.json`
- `id_results.json`
- `summary_results.json`

### 4.1 `case_results.json`

Contains one entry per input case. The original fields are preserved and metric values are appended.

### 4.2 `id_results.json`

Contains one entry per identity. All valid cases sharing the same `id_index` are averaged metric-wise.

### 4.3 `summary_results.json`

Contains the benchmark-wide statistics.

---