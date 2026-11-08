"""Metrics for custom JSON tasks; native benchmark tasks keep their native metrics."""
import re


def score_answer(example, text):
    # The shared backend normalizes thinking output before task scoring.
    answer = text.strip()
    scorer = example.get("scorer", "exact")
    if scorer == "exact":
        def normalize(s):
            return re.sub(r"\s+", " ", str(s).strip()).casefold()
        return float(normalize(answer) == normalize(example["answer"]))
    if scorer == "choice":
        boxed = re.findall(r"\\boxed\{([A-E1-5])\}", answer)
        matches = boxed or re.findall(r"(?:^|\n)\s*([A-E1-5])[.)]?\s*$", answer)
        return float(bool(matches) and matches[-1] == str(example["answer"]))
    if scorer == "math":
        from math_verify import parse, verify
        from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
        gold = parse(str(example["answer"]), extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        pred = parse(answer, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        return float(bool(gold) and bool(pred) and verify(gold, pred))
    if scorer == "ifeval":
        from lm_eval.tasks.ifeval.utils import process_results
        return float(process_results({**example, 'key': example.get('id', 'custom')}, [answer])['prompt_level_strict_acc'])
    raise ValueError("unknown_scorer")
