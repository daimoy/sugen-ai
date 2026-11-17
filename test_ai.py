import requests

url = "http://localhost:11434/api/generate"

# 假设这是你用手机录音，然后转成文字的一节课的“流水账”
with open("class.txt", "r", encoding="utf-8") as f:
    class_audio_text = f.read()

# 这是我们给 Sugen 设定的“学习助手”人设
prompt = f"""
你是一个专业的 AI 学习助手。
请将下面这段啰嗦的课堂录音文字稿，整理成结构化的 Markdown 格式笔记。
要求：
1. 过滤掉老师闲聊、维持纪律的废话（比如“大家安静一下”）。
2. 用 # 提取大标题，用 ## 提取小标题。
3. 用无序列表（-）提炼核心知识点（for循环、while循环、break）。
4. 把老师提到的“期末必考”内容，用粗体（**）标注出来。

原始录音文字：
{class_audio_text}
"""

data = {
    "model": "qwen2.5:7b",
    "prompt": prompt,
    "stream": False
}

print("Sugen 正在提炼课堂重点，请稍候...\n")

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("================【 Sugen 生成的笔记 】================")
    # 把 AI 整理好的笔记打印出来
    print(result['response'])
    print("======================================================")
else:
    print("连接失败了，错误码：", response.status_code)