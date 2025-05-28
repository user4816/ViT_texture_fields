## What's New
Based on the original Stable Flow code(https://github.com/snap-research/stable-flow), we introduce an adjustable attention injection strength parameter. Unlike the original method—which uniformly injects attention at full intensity (100%) across all Vital Layers—our approach enables users to precisely control the attention strength. This modification significantly improves editing quality, resulting in more natural and consistent outputs for diverse tasks such as background replacement, time-of-day transitions, and style transformations.

## Installation
conda env create -f environment.yml
conda activate stable-flow

## Reproduction
python run_stable_flow.py \
  --hf_token YOUR_PERSONAL_HUGGINGFACE_TOKEN \
  --input_img_path inputs/input_5.jpg \
  --prompts "A studio portrait photo of a woman" "A pencil sketch portrait of the same woman" \
  --attention_alpha 0.2 \
  --cpu_offload \
  --seed 42

## Style Editing (Portrait -> Pencil Sketch)
![tasK_1](https://github.com/user-attachments/assets/fab7d81e-6c82-4ff3-a8d9-3462b3bb0e62)



## Time-of-Day Transition (Daytime City Street -> Night)
![task_2](https://github.com/user-attachments/assets/a1cd14d3-759f-4783-af0b-a59f3c6de958)
