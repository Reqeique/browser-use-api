from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, UUID4
from typing import Optional, Dict, Any, List, Union
import asyncio
import uuid
from datetime import datetime
from enum import Enum
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Browser-Use API v2",
    description="REST API compatible with Browser-Use Cloud API v2 specification",
    version="2.0.0"
)

class TaskStatus(str, Enum):
    STARTED = "started"
    PAUSED = "paused"
    FINISHED = "finished"
    STOPPED = "stopped"

class TaskUpdateAction(str, Enum):
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    STOP_TASK_AND_SESSION = "stop_task_and_session"

class SupportedLLMs(str, Enum):
    BROWSER_USE_LLM = "browser-use-llm"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    O4_MINI = "o4-mini"
    O3 = "o3"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_FLASH_LATEST = "gemini-flash-latest"
    GEMINI_FLASH_LITE_LATEST = "gemini-flash-lite-latest"
    CLAUDE_SONNET_4_20250514 = "claude-sonnet-4-20250514"
    CLAUDE_SONNET_4_5_20250929 = "claude-sonnet-4-5-20250929"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    LLAMA_4_MAVERICK_17B_128E_INSTRUCT = "llama-4-maverick-17b-128e-instruct"
    CLAUDE_3_7_SONNET_20250219 = "claude-3-7-sonnet-20250219"

class TaskStepView(BaseModel):
    number: int
    memory: str
    evaluationPreviousGoal: str
    nextGoal: str
    url: str
    screenshotUrl: Optional[str] = None
    actions: List[str]

class FileView(BaseModel):
    id: UUID4
    fileName: str

class CreateTaskRequest(BaseModel):
    task: str
    llm: Optional[SupportedLLMs] = Field(default=SupportedLLMs.BROWSER_USE_LLM)
    startUrl: Optional[str] = None
    maxSteps: Optional[int] = Field(default=100)
    structuredOutput: Optional[str] = None
    sessionId: Optional[UUID4] = None
    metadata: Optional[Dict[str, str]] = None
    secrets: Optional[Dict[str, str]] = None
    allowedDomains: Optional[List[str]] = None
    opVaultId: Optional[str] = None
    highlightElements: Optional[bool] = Field(default=False)
    flashMode: Optional[bool] = Field(default=False)
    thinking: Optional[bool] = Field(default=False)
    vision: Optional[Union[bool, str]] = Field(default="auto")
    systemPromptExtension: Optional[str] = None
    cdpUrl: Optional[str] = None
    pageExtractionLlm: Optional[SupportedLLMs] = Field(default=SupportedLLMs.BROWSER_USE_LLM)
    storageStateUrl: Optional[str] = None  # NEW FIELD
    keepAlive: Optional[bool] = Field(default=False)

class TaskCreatedResponse(BaseModel):
    id: UUID4
    sessionId: UUID4

class TaskView(BaseModel):
    id: UUID4
    sessionId: UUID4
    llm: str
    task: str
    status: TaskStatus
    startedAt: datetime
    finishedAt: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    steps: List[TaskStepView] = Field(default_factory=list)
    output: Optional[str] = None
    outputFiles: List[FileView] = Field(default_factory=list)
    browserUseVersion: Optional[str] = None
    isSuccess: Optional[bool] = None
    error: Optional[str] = None

class TaskItemView(BaseModel):
    id: UUID4
    sessionId: UUID4
    llm: str
    task: str
    status: TaskStatus
    startedAt: datetime
    finishedAt: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[str] = None
    browserUseVersion: Optional[str] = None
    isSuccess: Optional[bool] = None
    error: Optional[str] = None

class TaskListResponse(BaseModel):
    items: List[TaskItemView]
    totalItems: int
    pageNumber: int
    pageSize: int

class UpdateTaskRequest(BaseModel):
    action: TaskUpdateAction

class TaskLogFileResponse(BaseModel):
    downloadUrl: str

task_store: Dict[str, Dict[str, Any]] = {}
session_store: Dict[str, Dict[str, Any]] = {}
running_tasks: Dict[str, asyncio.Task] = {}
paused_tasks: Dict[str, bool] = {}

def get_llm_model(llm_name: str):
    llm_map = {
        "browser-use-llm": ("browser_use", "ChatBrowserUse", None),
        "gpt-4.1": ("browser_use", "ChatOpenAI", "gpt-4"),
        "gpt-4.1-mini": ("browser_use", "ChatOpenAI", "gpt-4-mini"),
        "o4-mini": ("browser_use", "ChatOpenAI", "o1-mini"),
        "o3": ("browser_use", "ChatOpenAI", "o1"),
        "gemini-2.5-flash": ("browser_use", "ChatGoogle", "gemini-flash-lite-latest"),
        "gemini-2.5-pro": ("browser_use", "ChatGoogle", "gemini-exp-1206"),
        "gemini-flash-latest": ("browser_use", "ChatGoogle", "gemini-flash-latest"),
        "gemini-flash-lite-latest": ("browser_use", "ChatGoogle", "gemini-flash-lite-latest"),
        "claude-sonnet-4-20250514": ("browser_use", "ChatAnthropic", "claude-sonnet-4-0"),
        "claude-sonnet-4-5-20250929": ("browser_use", "ChatAnthropic", "claude-sonnet-4-0"),
        "gpt-4o": ("browser_use", "ChatOpenAI", "gpt-4o"),
        "gpt-4o-mini": ("browser_use", "ChatOpenAI", "gpt-4o-mini"),
        "llama-4-maverick-17b-128e-instruct": ("browser_use", "ChatGroq", "meta-llama/llama-4-maverick-17b-128e-instruct"),
        "claude-3-7-sonnet-20250219": ("browser_use", "ChatAnthropic", "claude-3-7-sonnet-20250219"),
    }

    return llm_map.get(llm_name, llm_map["browser-use-llm"])

async def run_browser_task(task_id: str, request: CreateTaskRequest):
    browser = None
    try:
        logger.info(f"[Task {task_id}] Starting task: {request.task}")
        logger.info(f"[Task {task_id}] LLM: {request.llm}, Max Steps: {request.maxSteps}")

        if task_store[task_id]["status"] != TaskStatus.STOPPED:
            task_store[task_id]["status"] = TaskStatus.STARTED

        from browser_use import Agent, Browser

        module_name, class_name, model_name = get_llm_model(request.llm.value if request.llm else "browser-use-llm")
        logger.info(f"[Task {task_id}] Initializing LLM: module={module_name}, class={class_name}, model={model_name}")
        page_extraction_module_name, page_extraction_class_name, page_extraction_model_name = get_llm_model(request.pageExtractionLlm.value if request.pageExtractionLlm else "browser-use-llm")
        
        try:
            module = __import__(module_name, fromlist=[class_name])
            llm_class = getattr(module, class_name)
            page_extraction_module = __import__(page_extraction_module_name, fromlist=[page_extraction_class_name])
            page_extraction_llm_class = getattr(page_extraction_module, page_extraction_class_name)
            llm = llm_class(model=model_name) if model_name else llm_class()
            page_extraction_llm = llm_class(model=page_extraction_model_name) if page_extraction_model_name else llm_class()

            logger.info(f"[Task {task_id}] ✓ LLM initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize {request.llm} provider: {str(e)}"
            logger.error(f"[Task {task_id}] ✗ LLM initialization failed: {error_msg}")
            task_store[task_id]["status"] = TaskStatus.STOPPED
            task_store[task_id]["error"] = error_msg
            task_store[task_id]["finishedAt"] = datetime.now()
            task_store[task_id]["isSuccess"] = False
            if task_id in running_tasks:
                del running_tasks[task_id]
            return

        try:
            import requests
            logger.info(f"[Task {task_id}] Initializing browser (start_url={request.startUrl})")

            browser_config = {}

            if request.cdpUrl:
                browser_config['cdp_url'] = request.cdpUrl

            if request.startUrl:
                browser_config["start_url"] = request.startUrl

            if request.keepAlive:
                browser_config["keep_alive"] = request.keepAlive
            # Handle storageStateUrl - fetch and save to state.json
            if request.storageStateUrl:
                try:
                    logger.info(f"[Task {task_id}] Fetching storage state from: {request.storageStateUrl}")
                    response = requests.get(request.storageStateUrl)
                    response.raise_for_status()

                    # Create state.json file with absolute path
                    state_file_path = os.path.abspath("state.json")
                    with open(state_file_path, 'w') as f:
                        f.write(response.text)

                    browser_config["storage_state"] = state_file_path
                    logger.info(f"[Task {task_id}] ✓ Storage state saved to {state_file_path}")
                except Exception as e:
                    error_msg = f"Failed to fetch or save storage state: {str(e)}"
                    logger.error(f"[Task {task_id}] ✗ Storage state error: {error_msg}")
                    # Continue without storage state (or you can choose to fail the task)

            browser = Browser(**browser_config)
            logger.info(f"[Task {task_id}] ✓ Browser initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize browser: {str(e)}"
            logger.error(f"[Task {task_id}] ✗ Browser initialization failed: {error_msg}")
            task_store[task_id]["status"] = TaskStatus.STOPPED
            task_store[task_id]["error"] = error_msg
            task_store[task_id]["finishedAt"] = datetime.now()
            task_store[task_id]["isSuccess"] = False
            if task_id in running_tasks:
                del running_tasks[task_id]
            return

        agent_config = {
            "task": request.task,
            "llm": llm,
            "browser": browser,
            "page_extraction_llm": page_extraction_llm,
        }
        


        if request.maxSteps:
            agent_config["max_steps"] = request.maxSteps

        logger.info(f"[Task {task_id}] Creating agent with config: {list(agent_config.keys())}")
        agent = Agent(**agent_config)
        logger.info(f"[Task {task_id}] ✓ Agent created successfully")

        start_time = datetime.now()

        try:
            logger.info(f"[Task {task_id}] Starting agent.run() - executing browser automation...")
            history = await agent.run()
            logger.info(f"[Task {task_id}] ✓ Agent.run() completed successfully")

            if task_id in running_tasks:
                del running_tasks[task_id]

            if task_store[task_id]["status"] != TaskStatus.STOPPED:
                task_store[task_id]["status"] = TaskStatus.FINISHED
                task_store[task_id]["finishedAt"] = datetime.now()
                task_store[task_id]["output"] = str(history)
                task_store[task_id]["isSuccess"] = True
                logger.info(f"[Task {task_id}] Task finished successfully with {len(getattr(history, 'history', []))} steps")

            steps = []
            if hasattr(history, 'history') and history.history:
                for idx, item in enumerate(history.history):
                    step = TaskStepView(
                        number=idx + 1,
                        memory=str(getattr(item, 'memory', '')),
                        evaluationPreviousGoal=str(getattr(item, 'evaluation_previous_goal', '')),
                        nextGoal=str(getattr(item, 'next_goal', '')),
                        url=str(getattr(item, 'url', '')),
                        screenshotUrl=None,
                        actions=[str(action) for action in getattr(item, 'actions', [])]
                    )
                    steps.append(step)

            task_store[task_id]["steps"] = [step.dict() for step in steps]

        except asyncio.CancelledError:
            logger.warning(f"[Task {task_id}] Task was cancelled")
            if task_id in running_tasks:
                del running_tasks[task_id]
            if task_store[task_id]["status"] != TaskStatus.STOPPED:
                task_store[task_id]["status"] = TaskStatus.STOPPED
                task_store[task_id]["finishedAt"] = datetime.now()
                task_store[task_id]["isSuccess"] = False
            raise

        except Exception as e:
            logger.error(f"[Task {task_id}] ✗ Agent.run() failed with exception: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"[Task {task_id}] Traceback:\n{traceback.format_exc()}")
            if task_id in running_tasks:
                del running_tasks[task_id]
            task_store[task_id]["status"] = TaskStatus.STOPPED
            task_store[task_id]["finishedAt"] = datetime.now()
            task_store[task_id]["error"] = str(e)
            task_store[task_id]["isSuccess"] = False

    except asyncio.CancelledError:
        logger.warning(f"[Task {task_id}] Task cancelled at outer level")
        pass
    except Exception as e:
        logger.error(f"[Task {task_id}] ✗ Outer exception: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"[Task {task_id}] Traceback:\n{traceback.format_exc()}")
        if task_id in running_tasks:
            del running_tasks[task_id]
        task_store[task_id]["status"] = TaskStatus.STOPPED
        task_store[task_id]["finishedAt"] = datetime.now()
        task_store[task_id]["error"] = str(e)
        task_store[task_id]["isSuccess"] = False
    finally:
    # ✅ Always close the browser explicitly
        if browser is not None:
            try:
                await browser.stop()
                logger.info(f"[Task {task_id}] ✓ Browser closed successfully")
            except Exception as e:
                logger.warning(f"[Task {task_id}] Failed to close browser: {str(e)}")
    
        # Clean up state file if it was created
        if request.storageStateUrl:
            state_file_path = os.path.abspath(f"state_{task_id}.json")
            try:
                if os.path.exists(state_file_path):
                    os.remove(state_file_path)
                    logger.info(f"[Task {task_id}] ✓ Cleaned up state file")
            except Exception as e:
                logger.warning(f"[Task {task_id}] Failed to clean up state file: {str(e)}")

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Browser-Use API v2",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /tasks": "Create a new browser automation task",
            "GET /tasks": "List all tasks with pagination",
            "GET /tasks/{task_id}": "Get detailed task information",
            "PATCH /tasks/{task_id}": "Update task (stop, pause, resume)",
            "GET /tasks/{task_id}/logs": "Get task execution logs",
            "GET /health": "Health check"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}

@app.post("/tasks", response_model=TaskCreatedResponse, status_code=202, tags=["Tasks"])
async def create_task(request: CreateTaskRequest):
    task_id = uuid.uuid4()
    session_id = request.sessionId or uuid.uuid4()

    if session_id not in session_store:
        session_store[str(session_id)] = {
            "id": session_id,
            "created_at": datetime.now(),
            "active": True
        }

    task_store[str(task_id)] = {
        "id": task_id,
        "sessionId": session_id,
        "llm": request.llm.value if request.llm else "browser-use-llm",
        "task": request.task,
        "status": TaskStatus.STARTED,
        "startedAt": datetime.now(),
        "finishedAt": None,
        "metadata": request.metadata or {},
        "steps": [],
        "output": None,
        "outputFiles": [],
        "browserUseVersion": "0.9.5",
        "isSuccess": None,
        "cdpUrl": request.cdpUrl if hasattr(request, "cdpUrl") else None,
    }

    async_task = asyncio.create_task(run_browser_task(str(task_id), request))
    running_tasks[str(task_id)] = async_task

    return TaskCreatedResponse(
        id=task_id,
        sessionId=session_id
    )

@app.get("/tasks", response_model=TaskListResponse, tags=["Tasks"])
async def list_tasks(
    pageSize: int = Query(default=20, ge=1, le=100),
    pageNumber: int = Query(default=1, ge=1),
    sessionId: Optional[UUID4] = Query(default=None),
    filterBy: Optional[TaskStatus] = Query(default=None),
    after: Optional[datetime] = Query(default=None),
    before: Optional[datetime] = Query(default=None)
):
    filtered_tasks = []

    for task_id, task_data in task_store.items():
        if sessionId and str(task_data["sessionId"]) != str(sessionId):
            continue

        if filterBy and task_data["status"] != filterBy:
            continue

        task_started = task_data["startedAt"]
        if after and task_started < after:
            continue
        if before and task_started > before:
            continue

        filtered_tasks.append(TaskItemView(
            id=task_data["id"],
            sessionId=task_data["sessionId"],
            llm=task_data["llm"],
            task=task_data["task"],
            status=task_data["status"],
            startedAt=task_data["startedAt"],
            finishedAt=task_data.get("finishedAt"),
            metadata=task_data.get("metadata", {}),
            output=task_data.get("output"),
            browserUseVersion=task_data.get("browserUseVersion"),
            isSuccess=task_data.get("isSuccess"),
            error=task_data.get("error")
        ))

    filtered_tasks.sort(key=lambda x: x.startedAt, reverse=True)

    total_items = len(filtered_tasks)
    start_idx = (pageNumber - 1) * pageSize
    end_idx = start_idx + pageSize
    page_items = filtered_tasks[start_idx:end_idx]

    return TaskListResponse(
        items=page_items,
        totalItems=total_items,
        pageNumber=pageNumber,
        pageSize=pageSize
    )

@app.get("/tasks/{task_id}", response_model=TaskView, tags=["Tasks"])
async def get_task(task_id: UUID4):
    task_data = task_store.get(str(task_id))

    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    steps = [TaskStepView(**step) for step in task_data.get("steps", [])]

    return TaskView(
        id=task_data["id"],
        sessionId=task_data["sessionId"],
        llm=task_data["llm"],
        task=task_data["task"],
        status=task_data["status"],
        startedAt=task_data["startedAt"],
        finishedAt=task_data.get("finishedAt"),
        metadata=task_data.get("metadata", {}),
        steps=steps,
        output=task_data.get("output"),
        outputFiles=task_data.get("outputFiles", []),
        browserUseVersion=task_data.get("browserUseVersion"),
        isSuccess=task_data.get("isSuccess"),
        error=task_data.get("error")
    )

@app.patch("/tasks/{task_id}", response_model=TaskView, tags=["Tasks"])
async def update_task(task_id: UUID4, request: UpdateTaskRequest):
    task_data = task_store.get(str(task_id))

    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    task_id_str = str(task_id)

    if request.action == TaskUpdateAction.STOP:
        if task_data["status"] == TaskStatus.STARTED:
            task_data["status"] = TaskStatus.STOPPED
            task_data["finishedAt"] = datetime.now()

            if task_id_str in running_tasks:
                running_tasks[task_id_str].cancel()

    elif request.action == TaskUpdateAction.PAUSE:
        if task_data["status"] == TaskStatus.STARTED:
            task_data["status"] = TaskStatus.PAUSED
            paused_tasks[task_id_str] = True

    elif request.action == TaskUpdateAction.RESUME:
        if task_data["status"] == TaskStatus.PAUSED:
            task_data["status"] = TaskStatus.STARTED
            if task_id_str in paused_tasks:
                del paused_tasks[task_id_str]

    elif request.action == TaskUpdateAction.STOP_TASK_AND_SESSION:
        task_data["status"] = TaskStatus.STOPPED
        task_data["finishedAt"] = datetime.now()
        session_id = str(task_data["sessionId"])
        if session_id in session_store:
            session_store[session_id]["active"] = False

        if task_id_str in running_tasks:
            running_tasks[task_id_str].cancel()

    steps = [TaskStepView(**step) for step in task_data.get("steps", [])]

    return TaskView(
        id=task_data["id"],
        sessionId=task_data["sessionId"],
        llm=task_data["llm"],
        task=task_data["task"],
        status=task_data["status"],
        startedAt=task_data["startedAt"],
        finishedAt=task_data.get("finishedAt"),
        metadata=task_data.get("metadata", {}),
        steps=steps,
        output=task_data.get("output"),
        outputFiles=task_data.get("outputFiles", []),
        browserUseVersion=task_data.get("browserUseVersion"),
        isSuccess=task_data.get("isSuccess"),
        error=task_data.get("error")
    )

@app.get("/tasks/{task_id}/logs", response_model=TaskLogFileResponse, tags=["Tasks"])
async def get_task_logs(task_id: UUID4):
    task_data = task_store.get(str(task_id))

    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    log_content = {
        "task_id": str(task_id),
        "task": task_data["task"],
        "status": task_data["status"],
        "steps": task_data.get("steps", []),
        "output": task_data.get("output"),
        "error": task_data.get("error")
    }

    download_url = f"/tasks/{task_id}/logs/download"

    return TaskLogFileResponse(downloadUrl=download_url)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)