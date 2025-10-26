from uuid import uuid4
from fastmcp import FastMCP
import os
import whisper
import requests
import os
from urllib.parse import urlparse
import asyncio
from pydantic import BaseModel

class McpResponse(BaseModel):
    code: int
    message: str
    task_id: str
    url: str
    status: str
    result: str = ""

mcp = FastMCP("My MCP Server")

model = whisper.load_model(
    name="large-v3-turbo",
    download_root="models/whisper",
    device="cuda"
)

tasks = {}

def download_audio(url: str) -> str:

    try:
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        audio_path =  f"temp/{filename}"

        response = requests.get(url)
        response.raise_for_status()
        with open(audio_path, "wb") as f:
            f.write(response.content)
        return audio_path
    except Exception as e:
        print(f"下载音频失败：{e}")
        raise


async def transcribe_audio(audio_path: str) -> str:
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    # 在线程池中运行CPU密集型的转录任务
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        lambda: model.transcribe(
            audio_path, 
            language="zh",
            initial_prompt="下面是普通话句子。"
        )
    )

    transcribed_text = result["text"]
    
    # 删除临时文件
    os.remove(audio_path)
    
    return transcribed_text

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

@mcp.tool
def submit_transcribe_task(url: str) -> McpResponse:
    """提交音频转录任务"""
    # 只检查未完成的任务，避免阻止已完成任务的URL重新提交
    pending_urls = [task["url"] for task in tasks.values() if not task["task"].done()]
    if url in pending_urls:
        return McpResponse(
            code=0, 
            message="URL已存在未完成的任务，请勿重复提交", 
            task_id="", 
            url=url, 
            status="pending")

    audio_path = download_audio(url)
    task_id = str(uuid4())
    task = asyncio.create_task(transcribe_audio(audio_path))
    tasks[task_id] = {"task": task, "url": url, "status": "pending"}

    return McpResponse(
        code=1, 
        message="任务提交成功", 
        task_id=task_id,
        url=url,
        status="pending"
    )

@mcp.tool
def check_transcribe_task(task_id: str) -> McpResponse:
    """获取音频转录任务状态"""  
    if task_id not in tasks:
        return McpResponse(code=0, message="任务ID不存在", task_id=task_id, url="", status="", result=None)
    
    task = tasks[task_id]["task"]
    url = tasks[task_id]["url"]
    if task.done():
        result = task.result()
        # 任务完成后从字典中删除，释放URL
        del tasks[task_id]
        return McpResponse(
            code=1, 
            message="任务完成", 
            task_id=task_id,
            url=url,
            status="completed",
            result=result
        )
    else:
        return McpResponse(
            code=1, 
            message="任务进行中", 
            task_id=task_id,
            url=tasks[task_id]["url"],
            status="pending",
            result=None
        )


if __name__ == "__main__":
    mcp.run(transport="http", port=8001)