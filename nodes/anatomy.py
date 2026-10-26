"""Nodes in EvaluationPack/Anatomy."""

import statistics

import torch
import torch.nn.functional as F

from .detector import detect as sam3_detect
from .pose import (
    ANKLES,
    EYES,
    FACE,
    HANDS,
    NOSE,
    TIPS,
    WRISTS,
    keypoints as rtmw_keypoints,
)

# Every size below is fixed so the score does not move with the input resolution.
DETECT_SIZE = 1024  # long side the image is resized to before detection
CROP_SIZE = 512  # long side a hand crop is resized to before its fingers are detected
CROP_PAD = 0.3  # margin around a hand box, as a fraction of the box's long side
MIN_PART = (
    1 / 32
)  # hands and feet shorter than this fraction of the image: a background figure's
BACKGROUND_FACE = 0.5  # a face smaller than this fraction of the largest face is a background figure's
MIN_HAND = 1 / 16  # hands shorter than this fraction of the image are not judged
MERGE_OVERLAP = 0.5  # a mask mostly inside another is a piece of it (a finger joint)
OWN_HAND = 0.5  # a finger mostly on the hand's mask belongs to that hand
DETAIL_FLOOR = 0.5  # at least this fraction as sharp as the median of its kind is fine
IN_VIEW = (
    0.5  # a keypoint scoring at least this fraction of its person's best is in view
)
OWN_WINDOW = 0.5  # a pose hand with this many keypoints in a hand's window is that hand
PARTS = ("face", "eye", "hand", "foot")
LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
LUMINANCE = torch.tensor([0.299, 0.587, 0.114])


#################################################################
# Helpers
#################################################################
def _resize(image, long_side):
    """image: (H, W, 3); resized so its longer side is `long_side`."""
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    size = (max(1, round(h * scale)), max(1, round(w * scale)))
    x = image.permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(
        x, size=size, mode="bilinear", antialias=True, align_corners=False
    )[0].permute(1, 2, 0)


def _box(mask):
    ys, xs = mask.nonzero(as_tuple=True)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def merge_overlapping(masks, overlap=MERGE_OVERLAP):
    """Unions masks whose overlap covers at least `overlap` of the smaller one, largest
    first. masks: (K, H, W) bool."""
    kept = []
    for mask in sorted(masks, key=lambda m: -int(m.sum())):
        for i, other in enumerate(kept):
            if (mask & other).sum() / mask.sum() >= overlap:
                kept[i] = other | mask
                break
        else:
            kept.append(mask)
    return torch.stack(kept) if kept else masks[:0]


def _size(mask):
    x0, y0, x1, y1 = _box(mask)
    return max(x1 - x0, y1 - y0)


def _large_enough(masks, least):
    """Drops masks whose long side is under `least` pixels: background figures."""
    kept = [mask for mask in masks if _size(mask) >= least]
    return torch.stack(kept) if kept else masks[:0]


def sharpness(crop):
    """Variance of the Laplacian of `crop` (H, W, 3), a crop of the DETECT_SIZE image."""
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return 0.0
    gray = (crop * LUMINANCE.to(crop)).sum(-1)
    edges = F.conv2d(gray[None, None], LAPLACIAN.to(crop)[None, None])
    return float(edges.var())


def _face_size(xy, in_view):
    face = xy[FACE][in_view[FACE]]
    return float((face.max(0) - face.min(0)).max()) if len(face) > 1 else 0.0


def pose_people(pose):
    """The foreground people the pose sees as (xy (133, 2), in_view (133,) bool):
    a keypoint is in view when it scores at least IN_VIEW of its person's best, and a
    person whose face is under BACKGROUND_FACE of the largest face is background."""
    xy, scores = pose
    people = [
        (person, score >= IN_VIEW * score.max())
        for person, score in zip(xy, scores)
        if score.max() > 0
    ]
    largest = max((_face_size(*person) for person in people), default=0.0)
    return [p for p in people if _face_size(*p) >= BACKGROUND_FACE * largest]


def pose_hands(people):
    """(xy (21, 2), in_view (21,) bool) per hand whose wrist and at least one fingertip
    are in view: a hand in a pocket shows its wrist but no fingertip."""
    hands = []
    for xy, in_view in people:
        for wrist, hand in zip(WRISTS, HANDS):
            if in_view[wrist] and in_view[hand][list(TIPS)].any():
                hands.append((xy[hand], in_view[hand]))
    return hands


def pose_targets(people, hands):
    """How many of each part the pose sees in view across the foreground people."""
    return {
        "face": sum(int(in_view[NOSE]) for _, in_view in people),
        "eye": sum(int(in_view[list(EYES)].sum()) for _, in_view in people),
        "hand": len(hands),
        "foot": sum(int(in_view[list(ANKLES)].sum()) for _, in_view in people),
    }


def _inside(points, window):
    ys, xs = window
    x, y = points[:, 0], points[:, 1]
    return (x >= xs.start) & (x < xs.stop) & (y >= ys.start) & (y < ys.stop)


#################################################################
# Evaluation
#################################################################
GROUPS = (
    ("", 1),
    ("anatomy", 2),
    ("detail", 1),
    ("score", 3),
    ("", 1),
)  # upper header: title, columns
HEADER = (
    "region",
    "detected",
    "target",
    "clarity",
    "anatomy",
    "detail",
    "final",
    "note",
)
EMPTY = "-"
SEPARATOR = "\n\n" + "=" * 72 + "\n\n"  # between the reports of a batch


def _table(rows, result):
    """rows: HEADER-shaped tuples, the first the person row, set off by a rule;
    result: the final score, on its own row under a rule. Columns are as wide as
    their widest cell, the region and note left-aligned, the number cells
    right-aligned, empty cells `-`."""
    rows += [("result", "", "", "", "", "", f"{result:.2f}", "")]
    rows = [HEADER] + [tuple(cell or EMPTY for cell in row) for row in rows]
    widths = [max(len(row[i]) for row in rows) for i in range(len(HEADER))]
    lines = []
    for cells in rows:
        line = f"{cells[0]:<{widths[0]}}"
        for cell, width in zip(cells[1:-1], widths[1:-1]):
            line += f" | {cell:>{width}}"
        lines.append(f"{line} | {cells[-1]}")
    groups, start = [], 0
    for title, count in GROUPS:
        width = sum(widths[start : start + count]) + 3 * (count - 1)
        groups.append(f"{title:<{width}}")
        start += count
    rule = "-" * len(lines[0])
    return (
        [" | ".join(groups).rstrip(), lines[0], rule, lines[1], rule]
        + lines[2:-1]
        + [rule, lines[-1]]
    )


def evaluate(image, detect, pose, threshold=0.3):
    """Scores one image (H, W, 3) in [0, 1].

    `detect(image, prompt, threshold)` returns (K, H, W) bool masks and
    `pose(image)` whole-body keypoints as `pose.keypoints` does. Returns
    (score, report): the score is anatomy × detail, the report a fixed-width table
    with a row per part kind and an indented row per instance.
    """
    work = _resize(image, DETECT_SIZE)
    long_side = max(work.shape[:2])
    found, ignored = {}, {}
    for part in PARTS:
        masks = merge_overlapping(detect(work, part, threshold))
        # Eyes are small by nature, and a background figure's are too small to be found at all.
        if part == "face":
            largest = max((_size(mask) for mask in masks), default=0)
            kept = _large_enough(masks, BACKGROUND_FACE * largest)
        elif part == "eye":
            kept = masks
        else:
            kept = _large_enough(masks, MIN_PART * long_side)
        found[part], ignored[part] = kept, len(masks) - len(kept)

    persons = max(1, len(found["face"]))
    people = pose_people(pose(work))
    unmatched = pose_hands(people)
    targets = pose_targets(people, unmatched)

    # Each found hand takes the pose hand with most keypoints in its window; a pose
    # hand left over is a hand that was drawn but could not be found as one.
    windows = []
    for hand in found["hand"]:
        x0, y0, x1, y1 = _box(hand)
        pad = int(CROP_PAD * max(x1 - x0, y1 - y0))
        windows.append(
            (slice(max(y0 - pad, 0), y1 + pad), slice(max(x0 - pad, 0), x1 + pad))
        )
    seen = {}
    for i, window in enumerate(windows):
        hits = [_inside(xy, window).mean() for xy, _ in unmatched]
        if hits and max(hits) >= OWN_WINDOW:
            seen[i] = unmatched.pop(hits.index(max(hits)))

    fingers = {}
    for i, (hand, window) in enumerate(zip(found["hand"], windows)):
        x0, y0, x1, y1 = _box(hand)
        size = max(x1 - x0, y1 - y0)
        if size < MIN_HAND * long_side:
            fingers[i] = ("-", "-", None, f"hand too small ({size}px), not judged")
            continue
        crop = _resize(work[window], CROP_SIZE)
        own = (
            _resize(hand[window].float().unsqueeze(-1).expand(-1, -1, 3), CROP_SIZE)[
                ..., 0
            ]
            > 0.5
        )
        pieces = merge_overlapping(detect(crop, "finger", threshold))
        # The margin lets another hand's fingers into the crop; a finger of this hand lies mostly on its mask.
        k = sum(1 for piece in pieces if (piece & own).sum() / piece.sum() >= OWN_HAND)
        # A hand the pose missed but SAM3 found fingers on is there all the same: it
        # joins the hand target and is held to 5 fingers.
        if i in seen:
            visible = int(seen[i][1][list(TIPS)].sum())
        else:
            visible = 5
            targets["hand"] += k > 0
        if k == 0:
            fingers[i] = (str(k), str(visible), None, "no finger found, not judged")
            continue
        fingers[i] = (str(k), str(visible), k == visible, "")

    # A row's final is 0 when its count is wrong, else its detail; the image's final
    # is the lowest row. Counts not judged leave the detail alone.
    rows = [("person", str(persons), str(len(people)), "", "", "", "", "")]
    finals = []
    for part in PARTS:
        n, target = len(found[part]), targets[part]
        ok = n == target
        notes = [f"{n} != {target}"] if not ok else []
        if ignored[part]:
            notes.append(f"{ignored[part]} too small, ignored")
        if part == "hand" and unmatched:
            notes.append(f"{len(unmatched)} seen by the pose but not found")
        finals.append(float(ok))
        rows.append(
            (
                part,
                str(n),
                str(target),
                "",
                f"{ok:.2f}",
                "",
                f"{ok:.2f}",
                ", ".join(notes),
            )
        )
        values = []
        for mask in found[part]:
            x0, y0, x1, y1 = _box(mask)
            values.append(sharpness(work[y0:y1, x0:x1]))
        median = statistics.median(values) if len(values) > 1 else None
        for i, value in enumerate(values):
            if median is None or median == 0:
                clarity, detail = "-", 1.0
            else:
                clarity = f"{value / median:.2f}"
                detail = min(1.0, value / (DETAIL_FLOOR * median))
            detected, target, ok, note = (
                fingers.get(i, ("", "", None, ""))
                if part == "hand"
                else ("", "", None, "")
            )
            final = detail if ok is None else (detail if ok else 0.0)
            finals.append(final)
            rows.append(
                (
                    f"  {part} {i + 1}",
                    detected,
                    target,
                    clarity,
                    "" if ok is None else f"{ok:.2f}",
                    f"{detail:.2f}",
                    f"{final:.2f}",
                    note,
                )
            )
    final = min(finals)

    return final, "\n".join(_table(rows, final))


#################################################################
# Nodes
#################################################################
class AnatomyEvaluator:
    """Scores how well faces, eyes, hands, fingers and feet hold up in an image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("score", "report")
    FUNCTION = "execute"
    CATEGORY = "EvaluationPack/Anatomy"

    @classmethod
    def execute(cls, image, threshold=0.3):
        results = [
            evaluate(single, sam3_detect, rtmw_keypoints, threshold) for single in image
        ]
        score = sum(s for s, _ in results) / len(results)
        return (score, SEPARATOR.join(report for _, report in results))
