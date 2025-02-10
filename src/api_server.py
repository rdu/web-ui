from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, Literal
import asyncio
import uuid
import os
from datetime import datetime
from src.utils.deep_research import deep_research
from src.utils.utils import get_llm_model  # Fixed import

app = FastAPI(title="Deep Research API")

# Store for running tasks
tasks = {}
# Lock for ensuring single task execution
task_lock = asyncio.Lock()

# Create data directory for saving results
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "api_tasks")
os.makedirs(DATA_DIR, exist_ok=True)

class TaskStatus:
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "google", "azure_openai", "deepseek", "ollama"] = Field(
        description="The LLM provider to use"
    )
    model: str = Field(description="The model name to use")
    api_key: Optional[str] = Field(None, description="Optional API key. If not provided, will use environment variables")
    endpoint: Optional[str] = Field(None, description="Optional API endpoint. If not provided, will use environment variables")
    temperature: float = Field(default=0.8, description="The temperature for LLM responses")

class DeepResearchConfig(BaseModel):
    task: str
    llm_config: LLMConfig
    save_dir: Optional[str] = None
    max_query_per_iter: Optional[int] = Field(default=3, description="Maximum number of queries per iteration")
    max_search_iteration: Optional[int] = Field(default=10, description="Maximum number of search iterations")
    use_own_browser: Optional[bool] = False
    headless: Optional[bool] = True
    disable_security: Optional[bool] = True
    use_vision: Optional[bool] = False
    max_steps: Optional[int] = 10

class TaskResponse(BaseModel):
    task_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[str] = None
    report_path: Optional[str] = None
    error: Optional[str] = None

def is_task_running() -> bool:
    """Check if any task is currently running"""
    return any(task['status'] == TaskStatus.RUNNING for task in tasks.values())

async def execute_task(task_id: str, config: DeepResearchConfig):
    try:
        async with task_lock:  # Ensure only one task can execute at a time
            # Convert config to kwargs for deep_research
            kwargs = config.dict(exclude={'task', 'llm_config'})
            
            # Set default save_dir if not provided
            if not kwargs.get('save_dir'):
                kwargs['save_dir'] = os.path.join(DATA_DIR, task_id)
            
            # Rename parameters to match deep_research expectations
            kwargs['max_search_iterations'] = kwargs.pop('max_search_iteration')
            kwargs['max_query_num'] = kwargs.pop('max_query_per_iter')
            
            # Initialize LLM using the utility from web UI
            llm = get_llm_model(
                provider=config.llm_config.provider,
                model_name=config.llm_config.model,
                temperature=config.llm_config.temperature,
                base_url=config.llm_config.endpoint,  # Will fall back to env if None
                api_key=config.llm_config.api_key  # Will fall back to env if None
            )
            
            # Execute deep_research
            result, report_path = await deep_research(
                task=config.task,
                llm=llm,
                **kwargs
            )
            
            tasks[task_id].update({
                'status': TaskStatus.COMPLETED,
                'completed_at': datetime.now().isoformat(),
                'result': result,
                'report_path': report_path
            })
    except Exception as e:
        tasks[task_id].update({
            'status': TaskStatus.FAILED,
            'completed_at': datetime.now().isoformat(),
            'error': str(e)
        })

@app.post("/tasks", response_model=TaskResponse)
async def create_task(config: DeepResearchConfig, background_tasks: BackgroundTasks):
    # Check if any task is currently running
    if is_task_running():
        raise HTTPException(
            status_code=409,
            detail="Another task is currently running. Please wait for it to complete or cancel it."
        )
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'task_id': task_id,
        'status': TaskStatus.RUNNING,
        'created_at': datetime.now().isoformat()
    }
    
    background_tasks.add_task(execute_task, task_id, config)
    return TaskResponse(**tasks[task_id])

@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskResponse(**tasks[task_id])

@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    if tasks[task_id]['status'] == TaskStatus.RUNNING:
        tasks[task_id].update({
            'status': TaskStatus.CANCELED,
            'completed_at': datetime.now().isoformat()
        })
    return TaskResponse(**tasks[task_id])

@app.get("/tasks", response_model=Dict[str, TaskResponse])
async def list_tasks():
    return {task_id: TaskResponse(**task_info) for task_id, task_info in tasks.items()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)