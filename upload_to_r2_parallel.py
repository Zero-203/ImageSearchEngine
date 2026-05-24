import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import partial

load_dotenv()

# ---------- 配置 ----------
R2_ENDPOINT = os.getenv('R2_ENDPOINT')
R2_ACCESS_KEY = os.getenv('R2_ACCESS_KEY_ID')
R2_SECRET_KEY = os.getenv('R2_SECRET_ACCESS_KEY')
BUCKET_NAME = os.getenv('R2_BUCKET_NAME')
IMAGE_FOLDER = os.getenv('IMAGE_FOLDER')    

MAX_WORKERS = 20          # 并行线程数（根据网络带宽调整，10-30 为宜）
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# ---------- 初始化 S3 客户端（线程安全） ----------
s3_client = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version='s3v4', retries={'max_attempts': 3})
)

# ---------- 上传单文件函数 ----------
def upload_single_file(filename: str, folder: str, bucket: str) -> tuple[str, bool, str]:
    """上传一张图片，返回 (文件名, 成功标志, 信息)"""
    filepath = os.path.join(folder, filename)
    content_type = 'image/jpeg' if filename.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
    try:
        s3_client.upload_file(
            filepath,
            bucket,
            filename,
            ExtraArgs={'ContentType': content_type}
        )
        return (filename, True, 'OK')
    except Exception as e:
        return (filename, False, str(e))

# ---------- 批量收集待上传文件 ----------
files_to_upload = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(SUPPORTED_FORMATS)
]
total = len(files_to_upload)
print(f"待上传文件总数: {total}")

# ---------- 并行上传 ----------
start_time = time.time()
success = 0
failed = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # 提交所有任务
    futures = {
        executor.submit(upload_single_file, f, IMAGE_FOLDER, BUCKET_NAME): f
        for f in files_to_upload
    }
    
    # 处理完成的任务（按完成顺序）
    for future in as_completed(futures):
        fname, ok, msg = future.result()
        if ok:
            success += 1
            print(f"✅ [{success}/{total}] {fname}")
        else:
            failed += 1
            print(f"❌ [{success+failed}/{total}] {fname} 失败: {msg}")

elapsed = time.time() - start_time
print(f"\n上传完成：成功 {success} 张，失败 {failed} 张，耗时 {elapsed:.1f} 秒")
print(f"平均速度：{success/elapsed:.1f} 张/秒")
