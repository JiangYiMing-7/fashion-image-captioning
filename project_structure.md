image_captioning_project/
│
├── README.md                     # 项目快速开始指南
├── requirements.txt              # 核心依赖（PyTorch, torchvision, jieba等）
│
├── configs/                      # 📁 配置管理中心
│   ├── __init__.py
│   ├── base_config.py            # 基础配置类
│   ├── model_configs.py          # 模型架构配置
│   └── training_configs.py       # 训练参数配置
│
├── data/                         # 📁 数据集处理   
│   ├── __init__.py
│   ├── dataset.py                # 数据集加载
│   └── vocabulary.py             # 词表管理
│
|—— dataset/                      # 📁 数据集存放
|    ├── __init__.py
|
├── models/                       # 🎯 核心模型
│   ├── __init__.py
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── cnn_encoder.py        # (1) CNN整体编码器 + (3) 局部网格特征输出
│   │   └── vit_encoder.py        # (6) 视觉Transformer编码器
│   │
│   ├── decoders/
│   │   ├── __init__.py
│   │   ├── rnn_decoder.py        # (1) 基础RNN解码器
│   │   ├── attention_decoder.py  # (3) RNN + 注意力（支持自注意力）
│   │   └── transformer_decoder.py # (6) Transformer解码器
│   │
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── attention.py          # 注意力机制（交叉+自注意力）
│   │   └── positional_encoding.py # Transformer位置编码
│   │
│   └── captioning_model.py       # 主模型：统一封装三模型的前向逻辑
|	
├── losses/                       # 📁 损失函数
│   ├── __init__.py
│   ├── cross_entropy_loss.py     # 基础交叉熵损失
│   ├── rl_loss.py                # (可选任务1) 强化学习损失
│   └── cider_reward.py           # 基于CIDEr的奖励函数
|	
├── evaluation/                   # 📁 评估指标（至少2种）
│   ├── __init__.py
│   ├── base_metric.py            # 评估基类
│   ├── meteor.py                 # METEOR指标实现
│   ├── rouge_l.py                # ROUGE-L指标实现
│   ├── cider_d.py                # CIDEr-D指标实现
│   └── spice.py                  # SPICE指标实现（调用场景图解析器）
│
├── training/
│   ├── __init__.py
│   └── trainer.py                # 统一训练器（支持Teacher-Forcing/自回归）
│
├── inference/                    # 📁 推理与部署
│   ├── __init__.py
│   ├── beam_search.py            # 束搜索解码
│   ├── greedy_search.py          # 贪心搜索
│   └── caption_generator.py      # 图像描述生成器接口
│
├── scripts/
│   ├── train.py                  # 一键训练脚本（指定模型类型）
│   └── evaluate.py               # 模型评估脚本
│
│
└── outputs/                      # 📁 实验输出（Git忽略）
    ├── checkpoints/              # 模型权重
    ├── logs/                     # 训练日志
    └── results/                  # 评估结果