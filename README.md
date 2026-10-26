# ComfyUI-Evaluation-Pack

[English](README.md) | [한국어](README_ko.md)

Image evaluation nodes that score a generated image with fixed, resolution-independent rules and report every
measurement behind the score.

## Usage

### Anatomy Evaluator

Finds faces, eyes, hands and feet with SAM3 text prompts, counts them against the faces found, counts the fingers of
each hand against the fingertips RTMW whole-body keypoints see, and measures how sharp each part is. Nothing is read
from the generation prompt: what should be in view comes from the pose.
The score is the lowest row's `final`, where a row with a wrong count scores 0 and otherwise its detail; the report is a fixed-width table with a row per part kind and an indented row per
instance, each with the measured value and the score, so a low score always says why.

| Widget | What it does |
|--------|--------------|
| `image` | Image or batch to score. A batch returns the mean score and one report per image |
| `threshold` | SAM3 confidence below which a detection is dropped. Default 0.3 |

| Output | Description |
|--------|-------------|
| `score` | 0 to 1 |
| `report` | Person count, then a table. `anatomy`: `detected` and `target`, on a kind's row the instances found and how many the pose sees in view (hands only count with a fingertip in view), on a hand's row the fingers found and the fingertips the pose sees (5 when the pose does not see the hand). `detail`: `clarity`, the part's sharpness over the median of its kind. `score`: `anatomy`, `detail` and `final`, their product. `note` says why a score fell |

**Anatomy.** N is the number of faces found. Each kind's `target` is what the pose sees in view: faces by the nose,
eyes by the eye keypoints, feet by the ankles, hands by wrists with at least one fingertip in view (a hand in a pocket
shows none), plus hands the pose missed but SAM3 found fingers on. A keypoint is in view when it scores at least half
of its person's best keypoint, since RTMW scores are not probabilities. A kind whose `detected` differs from its
`target` scores 0; a hand the pose sees but SAM3 does not find as a hand is such a difference. A face under half the
size of the largest face, and hands and feet shorter than 1/32 of the image, belong to background figures and are not
counted, and the pose drops the same people by face size. Each hand is cropped with a 30% margin, resized to 512px
and searched for fingers; a finger piece mostly inside another counts once, and only fingers lying mostly on the
hand's own mask count, so a neighbouring hand in the margin does not add to the tally. A hand's `target` is the
fingertips the pose sees in view (5 when it does not see the hand), and a hand whose finger count differs scores 0:
three fingers on a hand the pose sees three fingertips of is a hidden hand, not a wrong one. A hand with no finger
found, or shorter than 1/16 of the image, is not judged.

**Detail.** Each part is cropped from the 1024px image and its Laplacian variance measured. A part at least half as
sharp as the median of its kind (the other eye, hand or foot) scores 1; below that the score falls in proportion, and
a kind with a single instance is not judged. A row with the right count takes its detail as its `final`, and the image's `result` is the lowest `final` of any row. This catches one melted hand
next to a sound one, not two hands that are equally soft or an image that is blurry all over.

**Resolution.** Detection runs on the image resized to 1024px on the long side, and every crop is resized to a fixed
size before it is measured, so the same picture scores the same at 512px and 2048px.

## Installation

Search for **ComfyUI-Evaluation-Pack** in ComfyUI Manager, or:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alchemine/comfyui-evaluation-pack
```

The Anatomy Evaluator runs the SAM3 code and weights that [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG)
installs: install that pack and run one of its SAM3 nodes once so `models/sam3/sam3.pt` is downloaded. The pose comes
from [rtmlib](https://github.com/Tau-J/rtmlib) (`pip install -r requirements.txt`), which downloads its YOLOX and RTMW
ONNX models on first use and runs them with onnxruntime on the GPU when ComfyUI does.

## Nodes

`EvaluationPack/Anatomy`

| Node | Description |
|------|-------------|
| **Anatomy Evaluator** | Score faces, eyes, hands, fingers and feet for count and sharpness |
