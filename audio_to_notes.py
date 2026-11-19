import os
import sys
# 将 NVIDIA 库的路径添加到环境变量中
for site_path in sys.path:
    nvidia_bin = os.path.join(site_path, 'nvidia', 'cublas', 'bin')
    if os.path.exists(nvidia_bin):
        os.environ['PATH'] = nvidia_bin + os.pathsep + os.environ['PATH']
    nvidia_cudnn_bin = os.path.join(site_path, 'nvidia', 'cudnn', 'bin')
    if os.path.exists(nvidia_cudnn_bin):
        os.environ['PATH'] = nvidia_cudnn_bin + os.pathsep + os.environ['PATH']

import requests
import os
from faster_whisper import WhisperModel

# ================= 1. 语音转文字 =================
print("1. 正在加载语音识别模型（首次运行会下载约 460MB 的 small 模型）...")

# 升级 1：换成 "small" 模型，中文准确率大幅提升！
# 升级 2：device="cuda" 改为用你的 5060 显卡跑，速度起飞！
try:
    model = WhisperModel("small", device="cuda", compute_type="float16")
    print("显卡加速已开启！")
except Exception as e:
    print("显卡加载失败，回退到 CPU 模式：", e)
    model = WhisperModel("small", device="cpu", compute_type="int8")

audio_file = "test_audio.mp3"
print(f"2. 正在听写音频文件：{audio_file} ...")

# 升级 3：给 Whisper 加点专业词汇的提示，告诉它这是编程课
segments, info = model.transcribe(
    audio_file,
    beam_size=5,
    language="zh",
    initial_prompt="这是一堂关于Python编程的课程，提到了代码、缩进、注释、循环、变量等专业术语。"
)

raw_text = ""
for segment in segments:
    raw_text += segment.text

print(f"\n【语音识别结果（升级后）】：\n{raw_text}\n")
print("-" * 50)

# ================= 2. 调用千问整理笔记 =================
print("3. 正在呼叫本地千问模型整理笔记...")

url = "http://localhost:11434/api/generate"

# 升级 4：在 prompt 里让千问帮忙“纠错”，它很聪明，能根据上下文把错别字改过来
prompt = f"""
你是一个专业的 AI 学习助手。
请将下面这段啰嗦的课堂录音文字稿，整理成结构化的 Markdown 格式笔记。
要求：
1. 这是一个编程课的录音，语音识别可能存在同音错别字（比如把“缩进”识别成“索尽”，把“注释”识别成“驻式”），请你根据上下文帮我自动纠错。
2. 过滤掉闲聊、废话。
3. 用 # 提取大标题，用 ## 提取小标题。
4. 用无序列表（-）提炼核心知识点。

原始录音文字：
{raw_text}
"""

data = {
    "model": "qwen2.5:7b",
    "prompt": prompt,
    "stream": False
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()

    # 把生成的内容保存到文件里
    output_filename = "课堂笔记_自动生成.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(result['response'])

    print(f"\n✅ 成功！笔记已自动保存到当前文件夹下的：{output_filename}")
    print("你可以用任何 Markdown 阅读器打开它了！")

else:
    print("千问连接失败，错误码：", response.status_code)