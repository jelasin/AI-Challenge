# AI-Challenge 项目说明

本仓库包含 4 组挑战（LLM / Agent / MCP / Final）。

## 环境准备

- python 3.12
- 安装依赖（建议使用虚拟环境）
  - pip install -r requirements.txt
- 环境变量准备

```env
OPENAI_API_KEY=""
OPENAI_API_BASE=""

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=""
LANGSMITH_ENDPOINT=""
LANGSMITH_PROJECT=""

DEEPSEEK_API_KEY=""
DEEPSEEK_API_BASE=""

TAVILY_API_KEY=""
```

使用之前调用 `load_env.ps1/load_env.sh` 加载环境变量。

---

## LLM-Challenge 概览

目录：1-LLM-Challenge/

- challenge-1/ 基础翻译器（LCEL + 流式 + 可选结构化输出）
- challenge-2/ LangChain 工具调用（Function Calling）
- challenge-3/ Few-shot + 代码评审（结构化输出）
- challenge-4/ 文档处理与 RAG（向量检索增强生成）
- challenge-5/ 多模态图片识别（使用 init_chat_model，支持本地/URL 图片）
- challenge-6/ DALL·E 3 生图（LangChain 封装）
- challenge-7/ 文本转语音 TTS（OpenAI tts-1，封装为 LangChain 工具）
- challenge-8/ 语音转文本 STT（whisper-1，封装为 LangChain 工具）

下文为每题的简要说明与运行示例（命令仅示例，按需替换路径/参数）。

### Challenge 1：基础翻译器

要点：

- ChatPromptTemplate + LCEL 组合（prompt | model | parser）
- 两种模式：stream（流式输出）/ structured（结构化输出）

### Challenge 2：LangChain 工具调用系统

要点：

- @tool + Pydantic args_schema 定义工具，init_chat_model 一行启用函数调用
- llm.bind_tools(tools) 后模型可自动规划并调用工具

### Challenge 3：Few-shot + 代码评审（结构化输出）

要点：

- init_chat_model + with_structured_output(Pydantic)
- 语义相似度示例选择（FAISS + OpenAIEmbeddings）
- 自动识别语言 -> 规则拼装 -> few-shot -> 结构化评审

### Challenge 4：文档处理与 RAG

要点：

- 加载多种文档（md/txt/csv/pdf），文本切分，构建向量库（FAISS）
- 检索器 + ChatPromptTemplate 构建简单 RAG 链
- 提供交互式问答与进阶检索演示

### Challenge 5：多模态图片识别（图片理解）

要点：

- init_chat_model("gpt-4o")；HumanMessage 内容为 text + image_url 结构
- 支持本地图片与 URL

### Challenge 6：DALL·E 3 文生图（LangChain 封装）

要点：

- 使用 langchain_community.utilities.dalle_image_generator.DallEAPIWrapper
- 兼容 URL 或 base64 输出，自动保存为 PNG 至 --outdir（默认 image）

### Challenge 7：文本转语音 TTS（OpenAI tts-1）

要点：

- openai SDK 直连 tts-1，优先使用 streaming 写文件；封装为 LangChain 工具 openai_tts
- 输出保存到 voice 目录，可选语音风格与格式（mp3/wav/opus/aac/flac/pcm）

### Challenge 8：语音转文本 STT（whisper-1）

要点：

- openai SDK 直连 whisper-1；封装为 LangChain 工具 openai_transcribe
- 支持常见音频格式（mp3/mp4/mpeg/mpga/m4a/wav/webm）
