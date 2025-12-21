"""
使用 Qwen-VL 为 ModaNet 数据集生成服饰描述和背景描述

使用方法:
    # 使用 DashScope API (阿里云)
    python scripts/generate_captions_qwen.py \
        --image_dir ./modanet/images \
        --output_dir ./dataset_modanet \
        --api_key YOUR_DASHSCOPE_API_KEY

    # 使用本地模型
    python scripts/generate_captions_qwen.py \
        --image_dir ./modanet/images \
        --output_dir ./dataset_modanet \
        --local_model Qwen/Qwen-VL-Chat
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

from tqdm.auto import tqdm

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def encode_image_base64(image_path: str) -> str:
    """将图像编码为base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_suffix(image_path: str) -> str:
    """获取图像格式."""
    suffix = Path(image_path).suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        return "jpeg"
    elif suffix == ".png":
        return "png"
    elif suffix == ".gif":
        return "gif"
    elif suffix == ".webp":
        return "webp"
    return "jpeg"


class QwenVLClient:
    """Qwen-VL 客户端，支持 DashScope API 和本地模型."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        local_model: Optional[str] = None,
        device: str = "cuda",
    ):
        self.api_key = api_key
        self.local_model = local_model
        self.device = device
        
        if local_model:
            self._init_local_model()
        elif api_key:
            self._init_api()
        else:
            raise ValueError("需要提供 api_key 或 local_model")
    
    def _init_api(self):
        """初始化 DashScope API."""
        try:
            import dashscope
            dashscope.api_key = self.api_key
            self.use_api = True
            print("使用 DashScope API")
        except ImportError:
            raise ImportError("请安装 dashscope: pip install dashscope")
    
    def _init_local_model(self):
        """初始化本地模型."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"加载本地模型: {self.local_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_model, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_model,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).eval()
            self.use_api = False
            print("本地模型加载完成")
        except ImportError:
            raise ImportError("请安装 transformers: pip install transformers")
    
    def generate(self, image_path: str, prompt: str) -> str:
        """生成图像描述."""
        if self.use_api:
            return self._generate_api(image_path, prompt)
        else:
            return self._generate_local(image_path, prompt)
    
    def _generate_api(self, image_path: str, prompt: str) -> str:
        """使用 DashScope API 生成."""
        from dashscope import MultiModalConversation
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"file://{os.path.abspath(image_path)}"},
                    {"text": prompt},
                ],
            }
        ]
        
        response = MultiModalConversation.call(
            model="qwen-vl-max",
            messages=messages,
        )
        
        if response.status_code == 200:
            return response.output.choices[0].message.content[0]["text"]
        else:
            print(f"API 错误: {response.code} - {response.message}")
            return ""
    
    def _generate_local(self, image_path: str, prompt: str) -> str:
        """使用本地模型生成."""
        query = self.tokenizer.from_list_format([
            {"image": image_path},
            {"text": prompt},
        ])
        response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        return response


def generate_fashion_caption(client: QwenVLClient, image_path: str) -> dict:
    """为单张图像生成服饰描述和背景描述."""
    
    # Prompt 1: 服饰描述
    fashion_prompt = """请详细描述这张图片中人物穿着的服饰。
包括：
1. 服装类型（上衣、裤子、裙子、外套等）
2. 颜色和图案
3. 材质和风格
4. 搭配的配饰

请用一段流畅的中文描述，不要分点。"""

    # Prompt 2: 背景描述
    background_prompt = """请描述这张图片的背景环境。
包括：
1. 场景类型（室内/室外、街道/商场/公园等）
2. 环境特征
3. 光线和氛围

请用一段简短的中文描述，不要分点。"""

    # Prompt 3: 综合描述 (用于训练)
    combined_prompt = """请描述这张时尚图片，包括：
1. 人物穿着的服饰（类型、颜色、风格）
2. 背景环境

用一段流畅的中文描述整个画面。"""

    try:
        fashion_desc = client.generate(image_path, fashion_prompt)
        time.sleep(0.5)  # API 限流
        
        background_desc = client.generate(image_path, background_prompt)
        time.sleep(0.5)
        
        combined_desc = client.generate(image_path, combined_prompt)
        
        return {
            "fashion_description": fashion_desc.strip(),
            "background_description": background_desc.strip(),
            "combined_description": combined_desc.strip(),
            "success": True,
        }
    except Exception as e:
        print(f"生成失败 {image_path}: {e}")
        return {
            "fashion_description": "",
            "background_description": "",
            "combined_description": "",
            "success": False,
        }


def find_images(image_dir: str, extensions: List[str] = None) -> List[str]:
    """查找目录下所有图像文件."""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".webp"]
    
    image_dir = Path(image_dir)
    images = []
    for ext in extensions:
        images.extend(image_dir.glob(f"**/*{ext}"))
        images.extend(image_dir.glob(f"**/*{ext.upper()}"))
    
    return [str(p) for p in sorted(images)]


def main():
    parser = argparse.ArgumentParser(description="使用Qwen-VL生成服饰描述")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="ModaNet 图像目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--api_key", type=str, default=None,
                        help="DashScope API Key")
    parser.add_argument("--local_model", type=str, default=None,
                        help="本地模型路径 (如 Qwen/Qwen-VL-Chat)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="最多处理的图像数量")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="训练集比例")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断处继续")
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.api_key and not args.local_model:
        print("错误：需要提供 --api_key 或 --local_model")
        print("\n使用 DashScope API:")
        print("  python scripts/generate_captions_qwen.py --api_key YOUR_KEY --image_dir ./images --output_dir ./output")
        print("\n使用本地模型:")
        print("  python scripts/generate_captions_qwen.py --local_model Qwen/Qwen-VL-Chat --image_dir ./images --output_dir ./output")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "processed").mkdir(exist_ok=True)
    
    # 查找图像
    print(f"扫描图像目录: {args.image_dir}")
    images = find_images(args.image_dir)
    print(f"找到 {len(images)} 张图像")
    
    if args.max_images:
        images = images[:args.max_images]
        print(f"限制处理 {args.max_images} 张")
    
    # 检查已处理的图像 (用于断点续传)
    cache_file = output_dir / "cache.jsonl"
    processed = {}
    if args.resume and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                processed[item["image_path"]] = item
        print(f"从缓存加载 {len(processed)} 条记录")
    
    # 初始化客户端
    client = QwenVLClient(
        api_key=args.api_key,
        local_model=args.local_model,
        device=args.device,
    )
    
    # 批量生成
    results = list(processed.values())
    
    with open(cache_file, "a", encoding="utf-8") as cache_f:
        for image_path in tqdm(images, desc="生成描述"):
            if image_path in processed:
                continue
            
            captions = generate_fashion_caption(client, image_path)
            
            if captions["success"]:
                item = {
                    "image_path": image_path,
                    "image_id": Path(image_path).stem,
                    "fashion_description": captions["fashion_description"],
                    "background_description": captions["background_description"],
                    "caption": captions["combined_description"],
                }
                results.append(item)
                
                # 写入缓存
                cache_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                cache_f.flush()
            
            time.sleep(1)  # API 限流
    
    print(f"\n成功处理 {len(results)} 张图像")
    
    # 划分训练集和验证集
    random.seed(42)
    random.shuffle(results)
    
    split_idx = int(len(results) * args.train_ratio)
    train_data = results[:split_idx]
    val_data = results[split_idx:]
    
    # 保存为训练格式
    train_file = output_dir / "processed" / "train.jsonl"
    val_file = output_dir / "processed" / "val.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n数据集保存完成:")
    print(f"  训练集: {train_file} ({len(train_data)} 条)")
    print(f"  验证集: {val_file} ({len(val_data)} 条)")
    
    # 保存完整数据
    full_file = output_dir / "all_captions.json"
    with open(full_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  完整数据: {full_file}")
    
    print("\n下一步: 使用新数据集训练模型")
    print(f"  python scripts/train_model3.py --data_root {output_dir} --hf_tokenizer bert-base-chinese")


if __name__ == "__main__":
    main()
