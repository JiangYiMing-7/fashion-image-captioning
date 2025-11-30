import sys
from evaluation.base_metric import BaseMetric

try:
    from pycocoevalcap.meteor.meteor import Meteor
except ImportError:
    print("Warning: METEOR evaluation requires Java and pycocoevalcap setup")
    Meteor = None


class METEORMetric(BaseMetric):
    """
    METEOR: 基于对齐的机器翻译评估指标
    考虑同义词、词干和词序
    """

    def __init__(self):
        if Meteor is not None:
            self.meteor_eval = Meteor()
        else:
            self.meteor_eval = None

    def compute(self, references, hypotheses):
        """
        计算METEOR分数
        
        Args:
            references: List[str] 参考文本列表
            hypotheses: List[str] 生成文本列表
            
        Returns:
            dict: METEOR分数
        """
        if self.meteor_eval is None:
            return {"METEOR": 0.0}
            
        # 准备METEOR需要的数据格式
        gts = {i: [ref] for i, ref in enumerate(references)}
        res = {i: [hyp] for i, hyp in enumerate(hypotheses)}
        
        try:
            score, scores = self.meteor_eval.compute_score(gts, res)
            return {"METEOR": score}
        except Exception as e:
            print(f"METEOR calculation failed: {e}")
            return {"METEOR": 0.0}