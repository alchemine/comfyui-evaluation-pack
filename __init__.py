"""Custom nodes mappings."""

from .nodes.anatomy import AnatomyEvaluator


NODE_CLASS_MAPPINGS = {
    # EvaluationPack/Anatomy ##########################################################
    "AnatomyEvaluator": AnatomyEvaluator,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    # EvaluationPack/Anatomy ##########################################################
    "AnatomyEvaluator": "Anatomy Evaluator",
}
