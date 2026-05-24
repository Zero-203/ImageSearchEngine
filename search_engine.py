"""
图像搜索引擎核心：加载 Chinese-CLIP 模型，封装向量生成与 Upstash 检索。
"""
import os
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from upstash_vector import Index
from dotenv import load_dotenv

load_dotenv()


class ProductSearchEngine:
    """中文商品图像搜索引擎"""

    def __init__(self, model_path: str = "./models/chinese-clip"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[引擎] 使用设备: {self.device}")

        # 加载本地 CLIP 模型
        self.model = ChineseCLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = ChineseCLIPProcessor.from_pretrained(model_path)
        self.model.eval()

        # 初始化 Upstash 向量索引
        self.index = Index.from_env()
        print("[引擎] 初始化完成，模型与向量索引已就绪。")

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        """L2 归一化，与余弦相似度匹配"""
        return features / features.norm(dim=-1, keepdim=True)

    def encode_text(self, text: str) -> list[float]:
        """将中文文本转换为归一化的 512 维向量"""
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            features = outputs.pooler_output
            features = self._normalize(features)
        return features.cpu().numpy().flatten().tolist()

    def encode_image(self, image_path: str) -> list[float]:
        """将本地图片转换为归一化的 512 维向量"""
        img = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_image_features(**inputs)
            features = outputs.pooler_output
            features = self._normalize(features)
        return features.cpu().numpy().flatten().tolist()

    def search_by_text(self, query: str, top_k: int = 10) -> list[dict]:
        """文本搜图，返回结果列表（url 与相似度）"""
        if not query.strip():
            return []
        vector = self.encode_text(query)
        results = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [
            {"url": r.metadata["url"], "score": r.score}
            for r in results
            if "url" in r.metadata
        ]

    def search_by_image(self, image_path: str, top_k: int = 10) -> list[dict]:
        """以图搜图，返回结果列表（url 与相似度）"""
        vector = self.encode_image(image_path)
        results = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [
            {"url": r.metadata["url"], "score": r.score}
            for r in results
            if "url" in r.metadata
        ]
