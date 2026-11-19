import os
import sys

# 求求了，别扫了，最后终极杀手锏：在导入任何 AI 库之前，直接对系统隐藏显卡！
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
import requests
from faster_whisper import WhisperModel

st.set_page_config(page_title="Sugen (溯迹)", page_icon="📚", layout="wide")
st.title("📚 Sugen (溯迹) - 通用 AI 课堂笔记助手")
st.write("上传任意课程的录音，Sugen 会帮你提取文字并生成结构化笔记。全程本地运行，隐私安全！")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. 设置与上传")
    course_context = st.text_input("💡 这是一节什么课？(有助于提高识别率)", placeholder="例如：高等数学，讲极限和导数...")
    uploaded_file = st.file_uploader("上传课堂录音 (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

    if uploaded_file is not None:
        audio_path = "temp_uploaded_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(uploaded_file, format='audio/mp3')

        if st.button("🚀 开始生成笔记", type="primary", use_container_width=True):
            st.session_state['process'] = True

with col2:
    st.subheader("2. 生成结果")

    if st.session_state.get('process', False):
        try:
            with st.spinner('正在加载语音识别模型 (CPU模式)...'):
                # 强制只用 CPU，再也不碰显卡了！
                model = WhisperModel("small", device="cpu", compute_type="int8")

            with st.spinner('正在听写音频...'):
                prompt_hint = f"这是一堂关于{course_context}的课程。" if course_context else "这是一堂大学课程。"
                segments, info = model.transcribe(audio_path, beam_size=5, language="zh", initial_prompt=prompt_hint)
                raw_text = "".join([segment.text for segment in segments])

            st.success("语音识别完成！")
            with st.expander("查看原始转写文字 (点击展开)"):
                st.write(raw_text)

            with st.spinner('正在呼叫本地千问大模型整理笔记...'):
                url = "http://localhost:11434/api/generate"
                prompt = f"""
                你是一个专业的 AI 学习助手。
                背景信息：这是一堂关于 {course_context if course_context else '大学课程'} 的录音。
                请将下面这段啰嗦的课堂录音文字稿，整理成结构化的 Markdown 格式笔记。
                要求：1. 自动纠错。2. 过滤废话。3. 用 # 和 ## 提取标题。4. 用无序列表提炼重点。
                原始录音文字：{raw_text}
                """
                data = {"model": "qwen2.5:7b", "prompt": prompt, "stream": False}
                response = requests.post(url, json=data)

                if response.status_code == 200:
                    note_content = response.json()['response']
                    st.markdown("### 🎉 你的专属笔记：")
                    st.markdown(note_content)
                    st.download_button(label="⬇️ 下载 Markdown 笔记文件", data=note_content,
                                       file_name="Sugen_课堂笔记.md", mime="text/markdown")
                else:
                    st.error("千问连接失败，请确保 Ollama 正在运行。")
        except Exception as e:
            st.error(f"发生错误：{e}")
        st.session_state['process'] = False