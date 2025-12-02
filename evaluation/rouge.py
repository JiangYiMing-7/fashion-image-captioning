import sys
from evaluation.base_metric import BaseMetric

try:
    from pycocoevalcap.rouge.rouge import Rouge
except ImportError:
    print("Warning: ROUGE evaluation requires pycocoevalcap")
    Rouge = None


class ROUGEMetric(BaseMetric):
    """
    ROUGE: 面向召回的评估指标
    包含ROUGE-1, ROUGE-2, ROUGE-L
    """

    def __init__(self):
        if Rouge is not None:
            self.rouge_eval = Rouge()
        else:
            self.rouge_eval = None

    def compute(self, references, hypotheses):
        """
        计算ROUGE分数
        
        Args:
            references: List[str] 参考文本列表
            hypotheses: List[str] 生成文本列表
            
        Returns:
            dict: ROUGE-1, ROUGE-2, ROUGE-L分数
        """
        if self.rouge_eval is None:
            return {
                "ROUGE-1": 0.0,
                "ROUGE-2": 0.0, 
                "ROUGE-L": 0.0
            }
            
        # 准备ROUGE需要的数据格式
        gts = {i: [ref] for i, ref in enumerate(references)}
        res = {i: [hyp] for i, hyp in enumerate(hypotheses)}
        
        try:
            score, scores = self.rouge_eval.compute_score(gts, res)
            return {
                "ROUGE-1": score,
                "ROUGE-2": score,  # 注意：pycocoevalcap返回的是综合ROUGE分数
                "ROUGE-L": score
            }
        except Exception as e:
            print(f"ROUGE calculation failed: {e}")
            return {
                "ROUGE-1": 0.0,
                "ROUGE-2": 0.0,
                "ROUGE-L": 0.0
            }