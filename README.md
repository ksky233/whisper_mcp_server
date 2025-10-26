# 虚拟环境
建议使用pytorch的项目都使用conda管理虚拟环境，因为pytorch的安装会自动安装cuda和cudnn，而conda会自动管理这些依赖。
# 安装whisper
直接 pip install openai-whisper  

这玩意容易冲突，特别是numpy 和 numba

# 安装ffmpeg
ffmpeg 是一个非常强大的工具，用于处理音频和视频文件。在使用 whisper 进行语音识别时，需要安装 ffmpeg。

使用本地的ffmpeg也可，如果没有搜索教程安装
