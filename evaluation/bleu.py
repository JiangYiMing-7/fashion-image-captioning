import nltk.translate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from evaluation.base_metric import BaseMetric


class BLEUMetric(BaseMetric):
    """
    BLEU: 支持 BLEU-1 ~ BLEU-4 的快速评估器
    """

    def __init__(self):
        self.smooth = SmoothingFunction().method1

    def compute(self, references, hypotheses):
        """
        references: List[str]
        hypotheses: List[str]
        """

        bleu_1, bleu_2, bleu_3, bleu_4 = 0, 0, 0, 0

        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.lower().split()
            hyp_tokens = hyp.lower().split()

            bleu_1 += sentence_bleu([ref_tokens], hyp_tokens,
                                    weights=(1, 0, 0, 0),
                                    smoothing_function=self.smooth)

            bleu_2 += sentence_bleu([ref_tokens], hyp_tokens,
                                    weights=(0.5, 0.5, 0, 0),
                                    smoothing_function=self.smooth)

            bleu_3 += sentence_bleu([ref_tokens], hyp_tokens,
                                    weights=(1/3, 1/3, 1/3, 0),
                                    smoothing_function=self.smooth)

            bleu_4 += sentence_bleu([ref_tokens], hyp_tokens,
                                    weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=self.smooth)

        n = len(references)
        return {
            "BLEU-1": bleu_1 / n,
            "BLEU-2": bleu_2 / n,
            "BLEU-3": bleu_3 / n,
            "BLEU-4": bleu_4 / n,
        }
