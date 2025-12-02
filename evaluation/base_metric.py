from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """
    图像描述评估指标基类。
    所有指标需要实现 compute() 方法。
    """

    @abstractmethod
    def compute(self, references, hypotheses):
        """
        计算指标分数。

        参数:
            references (List[str]): 真实文本列表
            hypotheses (List[str]): 模型生成文本列表
        返回:
            dict: 指标名称 -> 得分
        """
        pass

    def __call__(self, references, hypotheses):
        return self.compute(references, hypotheses)
