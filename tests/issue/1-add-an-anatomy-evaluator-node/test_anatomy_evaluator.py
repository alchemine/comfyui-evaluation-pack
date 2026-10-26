import os

import numpy as np
import pytest
import torch


#################################################################
# Fixtures
#################################################################
class FakeDetector:
    """Stands in for SAM3: returns one box mask per entry of `parts[prompt]`, with
    the boxes given as fractions of whatever image it is handed, so the same
    fingers come back for every hand crop. Records the prompts it was asked."""

    def __init__(self, parts):
        self.parts = parts
        self.calls = []

    def __call__(self, image, prompt, threshold):
        self.calls.append(prompt)
        h, w = image.shape[:2]
        masks = torch.zeros((len(self.parts.get(prompt, [])), h, w), dtype=torch.bool)
        for mask, (x0, y0, x1, y1) in zip(masks, self.parts.get(prompt, [])):
            mask[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = True
        return masks


class FakePose:
    """Stands in for RTMW: one entry of `people` per person, each made by
    `person()`. A person's face (nose, eyes and the 68 contour points, spread over
    `face` of the image width) scores 1, each given hand puts its 21 keypoints in
    its box (fractions of the image) with the wrist scoring 1 and the fingertips
    `tips`, and the ankles score 1 when `feet` is set; everything else scores 0
    and is out of view."""

    def __init__(self, people=()):
        self.people = list(people)

    def __call__(self, image):
        h, w = image.shape[:2]
        xy = np.zeros((len(self.people), 133, 2))
        scores = np.zeros((len(self.people), 133))
        for i, p in enumerate(self.people):
            xy[i, 23:91, 0] = np.linspace(0, p["face"] * w, 68)
            scores[i, [0, 1, 2]] = 1.0
            scores[i, 23:91] = 1.0
            scores[i, [15, 16]] = 1.0 if p["feet"] else 0.0
            for hand, wrist, spec in (
                (slice(91, 112), 9, p["left"]),
                (slice(112, 133), 10, p["right"]),
            ):
                if spec is None:
                    continue
                (x0, y0, x1, y1), tips = spec
                xs = np.linspace(x0 * w, x1 * w, 21)
                ys = np.linspace(y0 * h, y1 * h, 21)
                xy[i, hand] = np.stack([xs, ys], 1)
                xy[i, wrist] = xy[i, hand][0]
                scores[i, wrist] = 1.0
                scores[i, hand][[4, 8, 12, 16, 20]] = tips
        return xy, scores


def person(left=None, right=None, feet=True, face=0.3):
    return dict(left=left, right=right, feet=feet, face=face)


IN_VIEW = (1.0,) * 5  # five fingertips in view
HIDDEN = (1.0, 1.0, 1.0, 0.0, 0.0)  # three in view, two hidden
POCKET = (0.0,) * 5  # wrist in view, no fingertip

FACE = (0.35, 0.05, 0.65, 0.35)
THIRD_HAND = (0.40, 0.50, 0.60, 0.70)
EYES = [(0.40, 0.15, 0.48, 0.20), (0.52, 0.15, 0.60, 0.20)]
HANDS = [(0.05, 0.50, 0.25, 0.70), (0.75, 0.50, 0.95, 0.70)]
FEET = [(0.30, 0.85, 0.45, 0.98), (0.55, 0.85, 0.70, 0.98)]
NO_POSE = FakePose()  # nobody seen
FULL_POSE = FakePose(
    [person((HANDS[0], IN_VIEW), (HANDS[1], IN_VIEW))]
)  # everything in view
# Five fingers as vertical strips over the upper half of the hand, which fills the
# middle 0.2..0.8 of its padded crop.
FINGERS = [(0.22 + 0.115 * i, 0.2, 0.31 + 0.115 * i, 0.5) for i in range(5)]
SIXTH_FINGER = (0.22, 0.55, 0.31, 0.78)  # on the hand, below the others
OTHER_HANDS_FINGER = (0.02, 0.2, 0.12, 0.5)  # in the crop's margin, off the hand


def whole(fingers=FINGERS, hands=HANDS, faces=(FACE,), eyes=EYES, feet=FEET):
    return {
        "face": list(faces),
        "eye": list(eyes),
        "hand": list(hands),
        "foot": list(feet),
        "finger": list(fingers),
    }


def checker(size, cells=64):
    """A sharp pattern whose cell count, not pixel size, is fixed, so it looks the
    same at any resolution once resized."""
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    pattern = ((y * cells // size + x * cells // size) % 2).float()
    return pattern.unsqueeze(-1).expand(size, size, 3).clone()


def row(report, name):
    """The report row named `name` as a dict keyed by the table's column names."""
    lines = report.splitlines()
    header = [cell.strip() for cell in lines[1].split("|")]
    for line in lines[3:]:
        if line.strip().startswith(name + " ") or line.strip() == name:
            return dict(zip(header, (cell.strip() for cell in line.split("|"))))
    raise KeyError(name)


def image_with_blurry_box(size, box):
    image = checker(size)
    x0, y0, x1, y1 = (int(v * size) for v in box)
    image[y0:y1, x0:x1] = 0.5
    return image


#################################################################
# Registration and node contract
#################################################################
def test_node_id_is_registered(mappings):
    assert "AnatomyEvaluator" in mappings


def test_clean_image_scores_one_and_report_is_text(pack, monkeypatch):
    anatomy = pack("anatomy")
    monkeypatch.setattr(anatomy, "sam3_detect", FakeDetector(whole()))
    monkeypatch.setattr(anatomy, "rtmw_keypoints", FULL_POSE)
    score, report = anatomy.AnatomyEvaluator.execute(
        checker(512).unsqueeze(0),
    )

    assert score == pytest.approx(1.0)
    assert isinstance(report, str) and "face" in report and "detected" in report


def test_batch_returns_mean_score_and_one_report_per_image(pack, monkeypatch):
    anatomy = pack("anatomy")
    monkeypatch.setattr(
        anatomy,
        "sam3_detect",
        FakeDetector(whole(fingers=FINGERS + [SIXTH_FINGER])),
    )
    monkeypatch.setattr(anatomy, "rtmw_keypoints", FULL_POSE)
    batch = torch.stack([checker(256), checker(256)])
    score, report = anatomy.AnatomyEvaluator.execute(
        batch,
    )

    assert score == pytest.approx(0.0)
    assert len(report.split(anatomy.SEPARATOR)) == 2


#################################################################
# Structure
#################################################################
def test_an_extra_hand_with_fingers_joins_the_target(pack):
    anatomy = pack("anatomy")
    with_three, report = anatomy.evaluate(
        checker(512), FakeDetector(whole(hands=HANDS + [THIRD_HAND])), FULL_POSE
    )

    assert with_three == pytest.approx(1.0)
    assert (row(report, "hand")["detected"], row(report, "hand")["target"]) == (
        "3",
        "3",
    )


def test_person_count_is_the_face_count(pack):
    anatomy = pack("anatomy")
    two_faces = whole(
        faces=(FACE, (0.0, 0.0, 0.3, 0.3)),
        eyes=EYES + [(0.05, 0.1, 0.1, 0.15), (0.15, 0.1, 0.2, 0.15)],
    )
    score, report = anatomy.evaluate(
        checker(512), FakeDetector(two_faces), FakePose([person(), person(feet=False)])
    )

    assert score == pytest.approx(1.0) and row(report, "person")["detected"] == "2"


def test_overlapping_finger_masks_count_once(pack):
    anatomy = pack("anatomy")
    duplicate = FINGERS + [(0.23, 0.2, 0.30, 0.5)]  # almost the first finger again
    score, _ = anatomy.evaluate(
        checker(512), FakeDetector(whole(fingers=duplicate)), FULL_POSE
    )

    assert score == pytest.approx(1.0)


def test_six_fingers_are_penalised(pack):
    anatomy = pack("anatomy")
    six = FINGERS + [SIXTH_FINGER]
    score, report = anatomy.evaluate(
        checker(512), FakeDetector(whole(fingers=six)), FULL_POSE
    )

    assert score == pytest.approx(0.0)
    hand = row(report, "hand 1")
    assert (hand["detected"], hand["target"], hand["anatomy"]) == ("6", "5", "0.00")


def test_four_fingers_cost_a_tenth_and_none_are_not_judged(pack):
    anatomy = pack("anatomy")
    four, _ = anatomy.evaluate(
        checker(512), FakeDetector(whole(fingers=FINGERS[:4])), FULL_POSE
    )
    none, report = anatomy.evaluate(
        checker(512), FakeDetector(whole(fingers=[])), FULL_POSE
    )

    assert four == pytest.approx(0.0)  # 4 fingers where the pose sees 5
    assert none == pytest.approx(1.0) and "not judged" in report


def test_another_hands_finger_in_the_margin_is_not_counted(pack):
    anatomy = pack("anatomy")
    score, report = anatomy.evaluate(
        checker(512),
        FakeDetector(whole(fingers=FINGERS + [OTHER_HANDS_FINGER])),
        FULL_POSE,
    )

    assert score == pytest.approx(1.0) and row(report, "hand 1")["detected"] == "5"


def test_small_hands_are_not_asked_for_fingers(pack):
    anatomy = pack("anatomy")
    tiny = [(0.10, 0.50, 0.15, 0.55), (0.80, 0.50, 0.85, 0.55)]  # 1/20 of the side
    detector = FakeDetector(whole(hands=tiny, fingers=FINGERS + [SIXTH_FINGER]))
    score, _ = anatomy.evaluate(checker(512), detector, FULL_POSE)

    assert score == pytest.approx(1.0) and "finger" not in detector.calls


#################################################################
# Pose: hidden fingers against missing ones
#################################################################
def test_fingers_the_pose_sees_hidden_do_not_cost(pack):
    anatomy = pack("anatomy")
    three = FakeDetector(whole(fingers=FINGERS[:3]))
    hidden, report = anatomy.evaluate(
        checker(512), three, FakePose([person((HANDS[0], HIDDEN), (HANDS[1], HIDDEN))])
    )
    in_view, _ = anatomy.evaluate(
        checker(512),
        three,
        FakePose([person((HANDS[0], IN_VIEW), (HANDS[1], IN_VIEW))]),
    )

    hand = row(report, "hand 1")
    assert hidden == pytest.approx(1.0) and (hand["detected"], hand["target"]) == (
        "3",
        "3",
    )
    assert in_view == pytest.approx(0.0)  # 3 fingers where the pose sees 5


def test_six_fingers_cost_whatever_the_pose_sees(pack):
    anatomy = pack("anatomy")
    score, _ = anatomy.evaluate(
        checker(512),
        FakeDetector(whole(fingers=FINGERS + [SIXTH_FINGER])),
        FakePose([person((HANDS[0], HIDDEN), (HANDS[1], HIDDEN))]),
    )

    assert score == pytest.approx(0.0)


def test_a_hand_the_pose_sees_but_the_detector_does_not_is_penalised(pack):
    anatomy = pack("anatomy")
    score, report = anatomy.evaluate(
        checker(512),
        FakeDetector(whole(hands=HANDS[:1])),
        FakePose([person((HANDS[0], IN_VIEW), (HANDS[1], IN_VIEW))]),
    )

    assert score == pytest.approx(0.0)
    assert row(report, "hand")["note"] == "1 != 2, 1 seen by the pose but not found"


def test_pose_hands_out_of_view_or_in_a_pocket_are_ignored(pack):
    anatomy = pack("anatomy")
    score, report = anatomy.evaluate(
        checker(512),
        FakeDetector(whole(hands=HANDS[:1])),
        FakePose([person((HANDS[0], IN_VIEW))]),
    )
    pocketed, pocket_report = anatomy.evaluate(
        checker(512),
        FakeDetector(whole(hands=HANDS[:1])),
        FakePose([person((HANDS[0], IN_VIEW), (HANDS[1], POCKET))]),
    )

    assert score == pytest.approx(1.0) and row(report, "hand")["target"] == "1"
    assert (
        pocketed == pytest.approx(1.0) and row(pocket_report, "hand")["target"] == "1"
    )


def test_background_figures_are_told_by_face_size(pack):
    anatomy = pack("anatomy")
    small = (0.0, 0.0, 0.1, 0.1)  # a face a third the size of FACE
    _, report = anatomy.evaluate(
        checker(512),
        FakeDetector(whole(faces=(FACE, small), hands=HANDS[:1])),
        FakePose([person((HANDS[0], IN_VIEW)), person((HANDS[1], IN_VIEW), face=0.1)]),
    )

    face = row(report, "face")
    assert (face["detected"], face["target"], face["note"]) == (
        "1",
        "1",
        "1 too small, ignored",
    )
    assert row(report, "hand")["target"] == "1"


def test_targets_come_from_the_pose(pack):
    anatomy = pack("anatomy")
    _, report = anatomy.evaluate(
        checker(512),
        FakeDetector(whole()),
        FakePose([person((HANDS[0], IN_VIEW), feet=False)]),
    )

    assert [
        row(report, kind)["target"] for kind in ("face", "eye", "hand", "foot")
    ] == ["1", "2", "2", "0"]


#################################################################
# Detail
#################################################################
def test_blurry_hand_lowers_the_detail_score_and_sharp_face_does_not(pack):
    anatomy = pack("anatomy")
    image = image_with_blurry_box(512, HANDS[0])
    score, report = anatomy.evaluate(image, FakeDetector(whole(fingers=[])), FULL_POSE)

    assert score < 1.0
    assert row(report, "face 1")["detail"] == "1.00"
    hand = row(report, "hand 1")
    assert hand["detail"] != "1.00" and float(hand["clarity"]) < 0.5


def test_score_does_not_depend_on_resolution(pack):
    anatomy = pack("anatomy")
    small, _ = anatomy.evaluate(
        image_with_blurry_box(512, HANDS[0]),
        FakeDetector(whole(fingers=[])),
        FULL_POSE,
    )
    large, _ = anatomy.evaluate(
        image_with_blurry_box(1024, HANDS[0]),
        FakeDetector(whole(fingers=[])),
        FULL_POSE,
    )

    assert small < 1.0 and small == pytest.approx(large, abs=0.05)


#################################################################
# SAM3
#################################################################
@pytest.mark.skipif(
    not (os.environ.get("ANATOMY_TEST_IMAGE") and os.environ.get("COMFYUI_DIR")),
    reason="set ANATOMY_TEST_IMAGE to a png and COMFYUI_DIR to a ComfyUI with ComfyUI-RMBG to run SAM3",
)
def test_sam3_evaluates_a_real_image(pack, monkeypatch):
    """Runs in a venv where ComfyUI-RMBG's SAM3 imports, e.g. ComfyUI's own."""
    import sys

    import numpy as np
    from PIL import Image

    comfy = os.environ["COMFYUI_DIR"]
    monkeypatch.setattr(
        sys.modules["folder_paths"], "models_dir", os.path.join(comfy, "models")
    )
    monkeypatch.setattr(
        sys.modules["folder_paths"],
        "get_folder_paths",
        lambda kind: [os.path.join(comfy, kind)],
    )
    anatomy = pack("anatomy")
    rgb = np.array(Image.open(os.environ["ANATOMY_TEST_IMAGE"]).convert("RGB"))
    score, report = anatomy.AnatomyEvaluator.execute(
        torch.from_numpy(rgb).float().div(255).unsqueeze(0),
    )

    assert 0.0 <= score <= 1.0 and "face" in report


def test_table_columns_stay_aligned_with_two_digit_instance_numbers(pack):
    anatomy = pack("anatomy")
    hands = [(0.02 + 0.08 * i, 0.5, 0.09 + 0.08 * i, 0.57) for i in range(12)]
    _, report = anatomy.evaluate(
        checker(1024),
        FakeDetector(whole(hands=hands, fingers=FINGERS)),
        NO_POSE,
    )

    lines = report.splitlines()
    rows = [line for line in lines if line.count("|") == lines[1].count("|")]
    assert "  hand 12 " in report
    assert len({tuple(i for i, c in enumerate(line) if c == "|") for line in rows}) == 1


def test_a_hand_found_with_fingers_joins_the_target_even_unseen_by_the_pose(pack):
    anatomy = pack("anatomy")
    _, report = anatomy.evaluate(checker(512), FakeDetector(whole()), NO_POSE)
    _, fingerless = anatomy.evaluate(
        checker(512), FakeDetector(whole(fingers=[])), NO_POSE
    )

    assert row(report, "hand")["target"] == "2"
    assert row(fingerless, "hand")["target"] == "0"


class FakeProcessor:
    """Stands in for Sam3Processor: counts the image passes and answers every
    prompt with no mask."""

    def __init__(self):
        self.images = 0

    def set_image(self, pil):
        self.images += 1
        return {"backbone_out": {}}

    def set_confidence_threshold(self, threshold):
        pass

    def set_text_prompt(self, prompt, state):
        return {"masks": torch.zeros(0)}


def test_the_image_backbone_runs_once_per_image(pack, monkeypatch):
    detector = pack("detector")
    processor = FakeProcessor()
    monkeypatch.setattr(detector, "_processor", processor)
    monkeypatch.setattr(detector, "_last_image", None)
    image = torch.rand(8, 8, 3)
    for prompt in ("face", "eye", "hand", "foot"):
        detector.detect(image, prompt, 0.3)
    assert processor.images == 1
    detector.detect(torch.rand(8, 8, 3), "finger", 0.3)
    assert processor.images == 2
