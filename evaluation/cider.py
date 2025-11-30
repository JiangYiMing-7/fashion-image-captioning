import sys
from evaluation.base_metric import BaseMetric

try:
    from pycocoevalcap.cider.cider import Cider
except ImportError:
    print("Warning: CIDEr evaluation requires pycocoevalcap")
    Cider = None


class CIDErMetric(BaseMetric):
    """
    CIDEr: 基于共识的图像描述评估指标
    包含CIDEr和CIDEr-D（带长度惩罚的版本）
    """

    def __init__(self):
        if Cider is not None:
            self.cider_eval = Cider()
        else:
            self.cider_eval = None

    def compute(self, references, hypotheses):
        """
        计算CIDEr和CIDEr-D分数
        
        Args:
            references: List[str] 参考文本列表
            hypotheses: List[str] 生成文本列表
            
        Returns:
            dict: CIDEr和CIDEr-D分数
        """
        if self.cider_eval is None:
            return {
                "CIDEr": 0.0,
                "CIDEr-D": 0.0
            }
            
        # 准备CIDEr需要的数据格式
        gts = {i: [ref] for i, ref in enumerate(references)}
        res = {i: [hyp] for i, hyp in enumerate(hypotheses)}
        
        try:
            score, scores = self.cider_eval.compute_score(gts, res)
            return {
                "CIDEr": score,
                "CIDEr-D": score  # pycocoevalcap中CIDEr默认使用长度惩罚
            }
        except Exception as e:
            print(f"CIDEr calculation failed: {e}")
            return {
                "CIDEr": 0.0,
                "CIDEr-D": 0.0
            }