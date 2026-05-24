import os
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from upstash_vector import Index
from dotenv import load_dotenv
from tqdm import tqdm

# ---------- 加载环境变量 ----------
load_dotenv()

# ---------- 配置 ----------
MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"  # 输出 512 维向量
IMAGE_FOLDER = os.getenv('IMAGE_FOLDER')
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
BATCH_SIZE = 32           # RTX 4060 8G 推荐 32
UPSERT_BATCH_SIZE = 100   # 向 Upstash 批量写入的大小

# R2 公开 URL 根路径
public_url = os.getenv('R2_PUBLIC_URL')
if not public_url:
    raise ValueError("请设置环境变量 R2_PUBLIC_URL")

# ---------- 初始化设备与模型 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

model = ChineseCLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = ChineseCLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

# ---------- 初始化 Upstash ----------
index = Index.from_env()

# ---------- 收集图像路径 ----------
image_files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(SUPPORTED_FORMATS)
]
print(f"找到 {len(image_files)} 张图片")

# ---------- 分批处理 ----------
vectors_buffer = []

for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="生成嵌入向量"):
    batch_names = image_files[i:i + BATCH_SIZE]
    batch_images = []
    valid_names = []

    for fname in batch_names:
        filepath = os.path.join(IMAGE_FOLDER, fname)
        try:
            img = Image.open(filepath).convert("RGB")
            batch_images.append(img)
            valid_names.append(fname)
        except Exception as e:
            print(f"警告：跳过损坏图片 {fname}, 错误: {e}")

    if not batch_images:
        continue

    # ---------- 提取图像特征 ----------
    inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)   # 返回 BaseModelOutputWithPooling
        image_features = outputs.pooler_output        # 提取真实向量 (batch, 512)
        # 归一化（配合余弦相似度）
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    embeddings = image_features.cpu().numpy()

    # ---------- 构建 Upsert 记录 ----------
    for fname, emb in zip(valid_names, embeddings):
        image_url = f"{public_url}/{fname}"
        vectors_buffer.append((fname, emb.tolist(), {"url": image_url}))

    # ---------- 批量写入 Upstash ----------
    while len(vectors_buffer) >= UPSERT_BATCH_SIZE:
        index.upsert(vectors=vectors_buffer[:UPSERT_BATCH_SIZE])
        vectors_buffer = vectors_buffer[UPSERT_BATCH_SIZE:]

# 写入剩余记录
if vectors_buffer:
    index.upsert(vectors=vectors_buffer)

print(f"完成！共写入 {len(image_files)} 条向量记录")
