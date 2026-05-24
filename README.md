# ImageSearchEngine

## 环境与依赖安装

本项目基于 **Python 3.13** 开发。

### 1. 安装依赖

```bash
# 克隆或进入项目目录后
pip install -r requirements.txt
```

**重要**：PyTorch 需要根据硬件单独安装（requirements.txt 中已说明）：

- **推荐（NVIDIA GPU + CUDA 12.6）**：
  ```bash
  pip install torch==2.11.0+cu126 torchvision==0.26.0+cu126 --index-url https://download.pytorch.org/whl/cu126
  ```

- **CPU 模式**：
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

安装完成后验证：
```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### 2. 环境变量与使用指南（按用户类型区分）

#### 面向一般用户（推荐：仅使用现成服务）
如果你只是想**直接使用**本检索系统（提供方已完成图片上传至 R2、公有 URL 配置，以及所有图像特征向量的嵌入与 Upstash 写入），**无需**准备本地数据集、上传图片或重新生成向量。

**最小 .env 配置**（复制以下内容保存为 `.env` 文件即可）：
```env
R2_PUBLIC_URL=https://pub-4080b0f6f00f401dac8b1fa169e852da.r2.dev
UPSTASH_VECTOR_REST_URL=https://boss-lioness-52864-gcp-usc1-vector.upstash.io
UPSTASH_VECTOR_REST_TOKEN=ABsIMGJvc3MtbGlvbmVzcy01Mjg2NC1nY3AtdXNjMXJlYWRvbmx5WlRKak1EVmhOMkl0WlRneE1pMDBNbUZqTFRsak56QXROak5tTmpsaFptUXpNV0Zs
```
（以上为公开只读访问凭证，供一般用户直接使用）

**使用步骤**：
1. 安装依赖 + 配置上述最小环境变量
2. 首次使用时运行一次模型下载（用于查询时实时编码）：`python download_model.py`
3. 直接启动界面：`python ui.py`（浏览器访问 http://127.0.0.1:7860）

**无需运行**：upload_to_r2_parallel.py、image_embed_to_upstash_vector.py。

#### 面向二次开发人员（完整自定义）
如果你希望**二次开发**，使用自己的图片数据集、自己的对象存储（R2 或其他 S3 兼容服务）和自己的 Upstash Vector 实例（或替换其他向量数据库）：

**完整 .env 配置**：
- `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`, `R2_PUBLIC_URL`
- `IMAGE_FOLDER`（本地待处理图片目录路径）
- `UPSTASH_VECTOR_REST_URL`, `UPSTASH_VECTOR_REST_TOKEN`

**完整流程**：
1. `python download_model.py`（首次运行或更换模型时）
2. 将你的图片放入 `IMAGE_FOLDER` 目录
3. `python upload_to_r2_parallel.py`（并行上传到你的图床）
4. `python image_embed_to_upstash_vector.py`（使用 Chinese-CLIP 生成向量并批量写入你的 Upstash）
5. `python ui.py`

你也可以修改 `search_engine.py` 中的模型路径或替换为其他嵌入模型。

### 3. 主要依赖（requirements.txt）

详见 `requirements.txt`，包含：
- boto3 / python-dotenv（R2 上传）
- gradio（Web UI）
- huggingface_hub / transformers / Pillow / torch / tqdm（Chinese-CLIP 模型与推理）
- upstash-vector（向量数据库）

## 项目说明

中文商品图像检索系统，支持文字搜图和以图搜图，使用 Chinese-CLIP 模型提取特征，Upstash Vector 存储与检索。


