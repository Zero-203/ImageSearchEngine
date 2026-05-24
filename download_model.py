"""
预先下载 Chinese-CLIP 模型到本地目录。
运行前请确保已设置代理（如有必要），
然后执行：python download_model.py
"""
import os
from huggingface_hub import snapshot_download

MODEL_ID = "OFA-Sys/chinese-clip-vit-base-patch16"
LOCAL_DIR = "./models/chinese-clip"

print(f"开始下载模型 {MODEL_ID} 到 {LOCAL_DIR} ...")
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,   # 确保 Windows 下不使用符号链接，直接复制文件
    resume_download=True,           # 支持断点续传
)
print(f"模型下载完成，保存在 {os.path.abspath(LOCAL_DIR)}")
