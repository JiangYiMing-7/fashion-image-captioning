import sys
from evaluation.base_metric import BaseMetric

try:
    from pycocoevalcap.spice.spice import Spice
except ImportError:
    print("Warning: SPICE evaluation requires Java and pycocoevalcap setup")
    Spice = None


class SPICEMetric(BaseMetric):
    """
    SPICE: 语义命题评估指标
    评估生成文本与参考文本在语义内容上的一致性
    """

    def __init__(self):
        if Spice is not None:
            self.spice_eval = Spice()
        else:
            self.spice_eval = None

    def compute(self, references, hypotheses):
        """
        计算SPICE分数
        
        Args:
            references: List[str] 参考文本列表
            hypotheses: List[str] 生成文本列表
            
        Returns:
            dict: SPICE分数及各子项分数
        """
        if self.spice_eval is None:
            return {
                "SPICE": 0.0,
                "SPICE-Precision": 0.0,
                "SPICE-Recall": 0.0,
                "SPICE-F1": 0.0
            }
            
        # 准备SPICE需要的数据格式
        gts = {i: [ref] for i, ref in enumerate(references)}
        res = {i: [hyp] for i, hyp in enumerate(hypotheses)}
        
        try:
            score, scores = self.spice_eval.compute_score(gts, res)
            
            # SPICE返回的是包含多个子分数的字典
            if isinstance(score, dict):
                return {
                    "SPICE": score.get('All', {}).get('f', 0.0),
                    "SPICE-Precision": score.get('All', {}).get('p', 0.0),
                    "SPICE-Recall": score.get('All', {}).get('r', 0.0),
                    "SPICE-F1": score.get('All', {}).get('f', 0.0)
                }
            else:
                return {
                    "SPICE": score,
                    "SPICE-Precision": score,
                    "SPICE-Recall": score, 
                    "SPICE-F1": score
                }
                
        except Exception as e:
            print(f"SPICE calculation failed: {e}")
            return {
                "SPICE": 0.0,
                "SPICE-Precision": 0.0,
                "SPICE-Recall": 0.0,
                "SPICE-F1": 0.0
            }