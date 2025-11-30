from evaluation.base_metric import BaseMetric
from evaluation.bleu import BLEUMetric
from evaluation.meteor import METEORMetric
from evaluation.rouge import ROUGEMetric
from evaluation.cider import CIDErMetric
from evaluation.spice import SPICEMetric


class CompositeMetric(BaseMetric):
    """
    组合评估指标：一次性计算所有指标
    """

    def __init__(self, metrics=None):
        """
        初始化组合评估器
        
        Args:
            metrics: List[str] 要计算的指标列表，默认计算所有指标
        """
        self.metrics = {}
        
        if metrics is None or "BLEU" in metrics:
            self.metrics["BLEU"] = BLEUMetric()
        if metrics is None or "METEOR" in metrics:
            self.metrics["METEOR"] = METEORMetric()
        if metrics is None or "ROUGE" in metrics:
            self.metrics["ROUGE"] = ROUGEMetric()
        if metrics is None or "CIDEr" in metrics:
            self.metrics["CIDEr"] = CIDErMetric()
        if metrics is None or "SPICE" in metrics:
            self.metrics["SPICE"] = SPICEMetric()

    def compute(self, references, hypotheses):
        """
        计算所有指标的分数
        
        Args:
            references: List[str] 参考文本列表
            hypotheses: List[str] 生成文本列表
            
        Returns:
            dict: 所有指标的分数
        """
        all_scores = {}
        
        for name, metric in self.metrics.items():
            try:
                scores = metric.compute(references, hypotheses)
                all_scores.update(scores)
                print(f"✅ {name} 计算完成")
            except Exception as e:
                print(f"❌ {name} 计算失败: {e}")
                # 为失败的指标设置默认值
                if name == "BLEU":
                    all_scores.update({"BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0})
                elif name == "METEOR":
                    all_scores.update({"METEOR": 0.0})
                elif name == "ROUGE":
                    all_scores.update({"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0})
                elif name == "CIDEr":
                    all_scores.update({"CIDEr": 0.0, "CIDEr-D": 0.0})
                elif name == "SPICE":
                    all_scores.update({"SPICE": 0.0, "SPICE-Precision": 0.0, "SPICE-Recall": 0.0, "SPICE-F1": 0.0})
        
        return all_scores