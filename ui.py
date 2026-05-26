import gradio as gr
from search_engine import ProductSearchEngine

print("正在加载搜索引擎，请稍候...")
searcher = ProductSearchEngine()

# --- 核心检索逻辑 ---
def process_text_search(text_query, top_k, threshold):
    if not text_query:
        return None, "请输入搜索词"
    
    raw_results = searcher.search_by_text(text_query, top_k=int(top_k))
    if not raw_results:
        return None, "未找到相关结果，请换个词试试。"
    
    # 过滤出分数大于等于设定阈值的图片
    image_urls = [item["url"] for item in raw_results if item["score"] >= threshold]
    
    status_msg = f"共检索到 **{len(image_urls)}** 条与“{text_query}”相关的结果 (相似度 >= {threshold})。"
    return image_urls, status_msg

def process_image_search(image_path, top_k, threshold):
    if not image_path:
        return None, "请先上传图片"
    
    raw_results = searcher.search_by_image(image_path, top_k=int(top_k))
    if not raw_results:
        return None, "未找到相似图片。"
        
    # 过滤出分数大于等于设定阈值的图片
    image_urls = [item["url"] for item in raw_results if item["score"] >= threshold]
    
    status_msg = f"共检索到 **{len(image_urls)}** 条相似图片结果 (相似度 >= {threshold})。"
    return image_urls, status_msg

# --- 搜索结果画廊的点击逻辑（加入/取消 切换） ---
def toggle_favorite(evt: gr.SelectData, current_gallery, current_favorites):
    # 提取被点击的图片链接
    selected_image = current_gallery[evt.index][0] if isinstance(current_gallery[evt.index], (list, tuple)) else current_gallery[evt.index]
    
    if selected_image in current_favorites:
        # 如果已经在收藏夹里，就移出它
        current_favorites.remove(selected_image)
        gr.Info("🗑️ 已取消收藏。")
    else:
        # 如果不在，就加入
        current_favorites.append(selected_image)
        gr.Info("⭐ 已成功加入收藏夹！")
        
    return current_favorites, current_favorites

# --- 收藏夹画廊的点击逻辑（直接取消） ---
def remove_from_favorites(evt: gr.SelectData, current_favorites):
    # 提取在收藏夹中被点击的图片链接
    selected_image = current_favorites[evt.index]
    
    if selected_image in current_favorites:
        current_favorites.remove(selected_image)
        gr.Info("🗑️ 已从收藏夹中移除。")
        
    return current_favorites, current_favorites

# --- 界面构建 ---
with gr.Blocks(title="图像检索系统") as demo:
    # 隐藏状态：用于保存收藏的图片列表
    favorites_state = gr.State([])
    
    gr.Markdown(
        """
        # 跨模态图像搜索引擎
        支持文字搜图与以图搜图，基于 Chinese-CLIP 特征提取与 Upstash Vector 检索构建。
        """
    )
    
    with gr.Row():
        # === 左侧：输入区与参数调整区 ===
        with gr.Column(scale=1):
            gr.Markdown("### 1. 输入查询 (Formulation)")
            with gr.Tabs():
                with gr.TabItem("📝 文字搜图"):
                    text_input = gr.Textbox(label="输入描述", placeholder="例如：一瓶红色的饮料...")
                    text_search_btn = gr.Button("搜索 (文字)", variant="primary") 
                    
                with gr.TabItem("🖼️ 以图搜图"):
                    image_input = gr.Image(type="filepath", label="上传图片以寻找相似图")
                    img_search_btn = gr.Button("搜索 (图片)", variant="primary") 
            
            gr.Markdown("### 2. 参数设置 (Refinement)")
            # 滑块1：控制返回的最大数量
            top_k_slider = gr.Slider(minimum=1, maximum=50, value=12, step=1, label="期望返回的结果数量 (Top-K)")
            # 滑块2：控制相似度阈值（越高要求越相似）
            threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.05, label="最低相似度阈值")

        # === 右侧：搜索结果输出区 ===
        with gr.Column(scale=2):
            gr.Markdown("### 3. 搜索结果 (Review)")
            status_output = gr.Markdown("等待搜索...") 
            
            gr.Markdown("*提示：点击图片即可收藏，**再次点击即可取消收藏**；您也可以使用右上角的图标直接下载。*")
            gallery_output = gr.Gallery(
                label="检索结果画廊",
                show_label=False,
                elem_id="gallery",
                columns=[3], 
                rows=[2],
                object_fit="contain",
                height="auto",
                interactive=False # 禁止手动上传，消除视觉误导
            )

    # === 底部：收藏夹区 ===
    gr.Markdown("---")
    gr.Markdown("### ⭐ 我的收藏夹 (Use)")
    gr.Markdown("*提示：点击此处的图片即可将其移出收藏夹。*")
    favorites_gallery = gr.Gallery(
        label="收藏的图片",
        show_label=False,
        columns=[6], 
        object_fit="contain",
        height="auto",
        interactive=False # 禁止手动上传，消除视觉误导
    )

    # --- 绑定所有事件 ---
    # 1. 文本搜索按钮点击事件
    text_search_btn.click(
        fn=process_text_search,
        inputs=[text_input, top_k_slider, threshold_slider],
        outputs=[gallery_output, status_output]
    )
    
    # 2. 图片搜索按钮点击事件
    img_search_btn.click(
        fn=process_image_search,
        inputs=[image_input, top_k_slider, threshold_slider],
        outputs=[gallery_output, status_output]
    )

    # 3. 搜索结果画廊点击事件（加入/取消收藏）
    gallery_output.select(
        fn=toggle_favorite,
        inputs=[gallery_output, favorites_state],
        outputs=[favorites_gallery, favorites_state] 
    )
    
    # 4. 收藏夹画廊点击事件（移除收藏）
    favorites_gallery.select(
        fn=remove_from_favorites,
        inputs=[favorites_state],
        outputs=[favorites_gallery, favorites_state]
    )

# --- 启动命令 ---
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)