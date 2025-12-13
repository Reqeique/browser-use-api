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
from cloud_storage import CloudStorage
from custom_tools import (
    CreateToolRequest, ToolView, ToolListResponse,
    create_tool as create_tool_func, list_tools as list_tools_func,
    get_tool as get_tool_func, delete_tool as delete_tool_func, register_tools_with_agent
)
load_dotenv()
# Set DISPLAY environment variable
os.environ['DISPLAY'] = ':99'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Browser-Use API v2",
    description="REST API compatible with Browser-Use Cloud API v2 specification",
    version="2.1.0"
)

print("--- LOADING MAIN.PY v2.1.0 ---")

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
    QWEN_3_32B = "qwen/qwen3-32b"
    CLAUDE_3_7_SONNET_20250219 = "claude-3-7-sonnet-20250219"

# ============== PROXY CONFIGURATION ==============
class ProxyConfig(BaseModel):
    """Proxy configuration for browser connections"""
    server: str = Field(..., description="Proxy server URL (e.g., http://proxy.example.com:8080 or socks5://proxy.example.com:1080)")
    username: Optional[str] = Field(default=None, description="Proxy username for authentication")
    password: Optional[str] = Field(default=None, description="Proxy password for authentication")
    bypass: Optional[List[str]] = Field(default=None, description="List of domains to bypass proxy (e.g., ['localhost', '*.example.com'])")

class ProxyType(str, Enum):
    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"

class AdvancedProxyConfig(BaseModel):
    """Advanced proxy configuration with more options"""
    type: ProxyType = Field(default=ProxyType.HTTP, description="Proxy type")
    host: str = Field(..., description="Proxy host (e.g., proxy.example.com)")
    port: int = Field(..., description="Proxy port (e.g., 8080)")
    username: Optional[str] = Field(default=None, description="Proxy username")
    password: Optional[str] = Field(default=None, description="Proxy password")
    bypass: Optional[List[str]] = Field(default=None, description="Domains to bypass")
    
    def to_server_url(self) -> str:
        """Convert to server URL format"""
        if self.username and self.password:
            return f"{self.type.value}://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"{self.type.value}://{self.host}:{self.port}"
# =================================================

# ============== API KEY CONFIGURATION ==============
class LLMProvider(str, Enum):
    """LLM Provider types for API key mapping"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    BROWSER_USE = "browser_use"

class APIKeyConfig(BaseModel):
    """API key configuration for different providers"""
    openai: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic: Optional[str] = Field(default=None, description="Anthropic API key")
    google: Optional[str] = Field(default=None, description="Google AI API key")
    groq: Optional[str] = Field(default=None, description="Groq API key")
    browserUse: Optional[str] = Field(default=None, description="Browser-Use API key")

# Mapping of LLM names to their providers
LLM_PROVIDER_MAP: Dict[str, LLMProvider] = {
    "browser-use-llm": LLMProvider.BROWSER_USE,
    "gpt-4.1": LLMProvider.OPENAI,
    "gpt-4.1-mini": LLMProvider.OPENAI,
    "o4-mini": LLMProvider.OPENAI,
    "o3": LLMProvider.OPENAI,
    "gpt-4o": LLMProvider.OPENAI,
    "gpt-4o-mini": LLMProvider.OPENAI,
    "gemini-2.5-flash": LLMProvider.GOOGLE,
    "gemini-2.5-pro": LLMProvider.GOOGLE,
    "gemini-flash-latest": LLMProvider.GOOGLE,
    "gemini-flash-lite-latest": LLMProvider.GOOGLE,
    "claude-sonnet-4-20250514": LLMProvider.ANTHROPIC,
    "claude-sonnet-4-5-20250929": LLMProvider.ANTHROPIC,
    "claude-3-7-sonnet-20250219": LLMProvider.ANTHROPIC,
    "llama-4-maverick-17b-128e-instruct": LLMProvider.GROQ,
    "qwen/qwen3-32b": LLMProvider.GROQ,
}

# Environment variable names for each provider
PROVIDER_ENV_VARS: Dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProvider.GOOGLE: "GOOGLE_API_KEY",
    LLMProvider.GROQ: "GROQ_API_KEY",
    LLMProvider.BROWSER_USE: "BROWSER_USE_API_KEY",
}

def get_provider_for_llm(llm_name: str) -> LLMProvider:
    """Get the provider type for a given LLM name"""
    return LLM_PROVIDER_MAP.get(llm_name, LLMProvider.BROWSER_USE)

def get_api_key_for_provider(
    provider: LLMProvider,
    api_key: Optional[str] = None,
    api_keys: Optional[APIKeyConfig] = None
) -> Optional[str]:
    """
    Get API key for a provider with fallback logic:
    1. Use directly provided api_key if available
    2. Use provider-specific key from APIKeyConfig if available
    3. Fall back to environment variable
    
    Args:
        provider: The LLM provider
        api_key: Directly provided API key (highest priority)
        api_keys: APIKeyConfig object with provider-specific keys
    
    Returns:
        API key string or None
    """
    # Priority 1: Directly provided API key
    if api_key:
        return api_key
    
    # Priority 2: Provider-specific key from APIKeyConfig
    if api_keys:
        provider_key_map = {
            LLMProvider.OPENAI: api_keys.openai,
            LLMProvider.ANTHROPIC: api_keys.anthropic,
            LLMProvider.GOOGLE: api_keys.google,
            LLMProvider.GROQ: api_keys.groq,
            LLMProvider.BROWSER_USE: api_keys.browserUse,
        }
        if provider_key_map.get(provider):
            return provider_key_map[provider]
    
    # Priority 3: Environment variable
    env_var = PROVIDER_ENV_VARS.get(provider)
    if env_var:
        return os.environ.get(env_var)
    
    return None

def set_api_key_env(provider: LLMProvider, api_key: str) -> None:
    """Temporarily set API key in environment for LLM initialization"""
    env_var = PROVIDER_ENV_VARS.get(provider)
    if env_var and api_key:
        os.environ[env_var] = api_key
        logger.info(f"Set {env_var} from request")

def mask_api_key(api_key: Optional[str]) -> str:
    """Mask API key for logging (show first 4 and last 4 chars)"""
    if not api_key:
        return "None"
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:4]}...{api_key[-4:]}"
# ===================================================

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
    storageStateUrl: Optional[str] = None
    keepAlive: Optional[bool] = Field(default=False)
    headless: Optional[bool] = Field(default=True)
    profileDirectory: Optional[str] = None
    # ============== PROXY FIELDS ==============
    proxy: Optional[ProxyConfig] = Field(default=None, description="Proxy configuration for the browser")
    proxyUrl: Optional[str] = Field(default=None, description="Simple proxy URL (e.g., http://user:pass@proxy:8080)")
    # ==========================================
    toolIds: Optional[List[UUID4]] = Field(default=None, description="IDs of registered custom tools to use")
    # ============== API KEY FIELDS ==============
    apiKey: Optional[str] = Field(default=None, description="API key for the main LLM (overrides environment variable)")
    apiKeys: Optional[APIKeyConfig] = Field(default=None, description="Provider-specific API keys")
    pageExtractionApiKey: Optional[str] = Field(default=None, description="API key for page extraction LLM (falls back to apiKey or env)")
    # ============================================

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

class VncResponse(BaseModel):
    url: str
    status: str
    display: str

task_store: Dict[str, Dict[str, Any]] = {}
session_store: Dict[str, Dict[str, Any]] = {}
running_tasks: Dict[str, asyncio.Task] = {}
paused_tasks: Dict[str, bool] = {}
DEF_ARGS = [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu',
    '--disable-extensions',
    '--no-first-run',
    '--disable-default-apps',
]

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

# ============== PROXY HELPER FUNCTIONS ==============
def build_proxy_config(proxy: Optional[ProxyConfig] = None, proxy_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Build proxy configuration dictionary for Playwright/browser-use
    
    Args:
        proxy: ProxyConfig object with detailed proxy settings
        proxy_url: Simple proxy URL string
    
    Returns:
        Dictionary with proxy configuration or None
    """
    if proxy:
        proxy_dict = {"server": proxy.server}
        if proxy.username:
            proxy_dict["username"] = proxy.username
        if proxy.password:
            proxy_dict["password"] = proxy.password
        if proxy.bypass:
            proxy_dict["bypass"] = ",".join(proxy.bypass)
        return proxy_dict
    elif proxy_url:
        # Parse proxy URL to extract components
        return parse_proxy_url(proxy_url)
    return None

def parse_proxy_url(proxy_url: str) -> Dict[str, Any]:
    """
    Parse a proxy URL into components
    
    Supports formats:
    - http://proxy.example.com:8080
    - http://user:pass@proxy.example.com:8080
    - socks5://user:pass@proxy.example.com:1080
    """
    from urllib.parse import urlparse
    
    parsed = urlparse(proxy_url)
    
    proxy_dict = {
        "server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
    }
    
    if parsed.username:
        proxy_dict["username"] = parsed.username
    if parsed.password:
        proxy_dict["password"] = parsed.password
    
    return proxy_dict

def get_proxy_args(proxy_config: Optional[Dict[str, Any]]) -> List[str]:
    """
    Generate Chrome command-line arguments for proxy
    
    Args:
        proxy_config: Proxy configuration dictionary
    
    Returns:
        List of Chrome arguments for proxy
    """
    if not proxy_config:
        return []
    
    args = []
    server = proxy_config.get("server", "")
    
    if server:
        args.append(f"--proxy-server={server}")
    
    bypass = proxy_config.get("bypass", "")
    if bypass:
        args.append(f"--proxy-bypass-list={bypass}")
    
    return args
# ====================================================

# ============== LLM INITIALIZATION HELPER ==============
def initialize_llm(
    llm_name: str,
    api_key: Optional[str] = None,
    api_keys: Optional[APIKeyConfig] = None,
    task_id: Optional[str] = None
):
    """
    Initialize an LLM with the appropriate API key.
    
    Priority for API key:
    1. Directly provided api_key parameter
    2. Provider-specific key from api_keys config
    3. Environment variable
    
    Args:
        llm_name: Name of the LLM to initialize
        api_key: Direct API key override
        api_keys: APIKeyConfig with provider-specific keys
        task_id: Task ID for logging
    
    Returns:
        Initialized LLM instance
    """
    log_prefix = f"[Task {task_id}]" if task_id else "[LLM Init]"
    
    module_name, class_name, model_name = get_llm_model(llm_name)
    provider = get_provider_for_llm(llm_name)
    
    # Get the appropriate API key
    resolved_api_key = get_api_key_for_provider(provider, api_key, api_keys)
    
    logger.info(f"{log_prefix} Initializing LLM: {llm_name} (provider: {provider.value})")
    logger.info(f"{log_prefix} API key source: {'request' if api_key else 'config' if (api_keys and resolved_api_key) else 'environment'}")
    logger.info(f"{log_prefix} API key: {mask_api_key(resolved_api_key)}")
    
    if not resolved_api_key and provider != LLMProvider.BROWSER_USE:
        raise ValueError(f"No API key found for {provider.value}. Provide via request or set {PROVIDER_ENV_VARS.get(provider)} environment variable.")
    
    # Import and initialize the LLM
    module = __import__(module_name, fromlist=[class_name])
    llm_class = getattr(module, class_name)
    
    # Build initialization kwargs
    init_kwargs = {}
    if model_name:
        init_kwargs["model"] = model_name
    
    # Add API key to initialization based on provider
    if resolved_api_key:
        if provider == LLMProvider.OPENAI:
            init_kwargs["api_key"] = resolved_api_key
        elif provider == LLMProvider.ANTHROPIC:
            init_kwargs["api_key"] = resolved_api_key
        elif provider == LLMProvider.GOOGLE:
            init_kwargs["api_key"] = resolved_api_key
        elif provider == LLMProvider.GROQ:
            init_kwargs["api_key"] = resolved_api_key
        elif provider == LLMProvider.BROWSER_USE:
            init_kwargs["api_key"] = resolved_api_key
        
        # Also set environment variable as fallback for libraries that read from env
        set_api_key_env(provider, resolved_api_key)
    
    # Initialize the LLM
    llm = llm_class(**init_kwargs)
    
    logger.info(f"{log_prefix} ✓ LLM initialized successfully")
    return llm
# =======================================================

async def run_browser_task(task_id: str, request: CreateTaskRequest):
    session_id = uuid.uuid4()
    browser = None
    monitor_task = None
    
    # Store original env vars to restore later
    original_env_vars: Dict[str, Optional[str]] = {}
    
    try:
        logger.info(f"[Task {task_id}] Starting task: {request.task}")
        logger.info(f"[Task {task_id}] Using display: {os.environ.get('DISPLAY', 'Not set')}")

        if task_store[task_id]["status"] != TaskStatus.STOPPED:
            task_store[task_id]["status"] = TaskStatus.STARTED

        from browser_use import Agent, Browser

        # ============== LLM INITIALIZATION WITH API KEYS ==============
        llm_name = request.llm.value if request.llm else "browser-use-llm"
        page_extraction_llm_name = request.pageExtractionLlm.value if request.pageExtractionLlm else "browser-use-llm"
        
        try:
            # Store original env vars for cleanup
            for provider in LLMProvider:
                env_var = PROVIDER_ENV_VARS.get(provider)
                if env_var:
                    original_env_vars[env_var] = os.environ.get(env_var)
            
            # Initialize main LLM
            llm = initialize_llm(
                llm_name=llm_name,
                api_key=request.apiKey,
                api_keys=request.apiKeys,
                task_id=task_id
            )
            
            # Initialize page extraction LLM
            # Use pageExtractionApiKey if provided, otherwise fall back to main apiKey
            page_extraction_key = request.pageExtractionApiKey or request.apiKey
            page_extraction_llm = initialize_llm(
                llm_name=page_extraction_llm_name,
                api_key=page_extraction_key,
                api_keys=request.apiKeys,
                task_id=task_id
            )
            
            logger.info(f"[Task {task_id}] ✓ All LLMs initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize LLM: {str(e)}"
            logger.error(f"[Task {task_id}] ✗ LLM initialization failed: {error_msg}")
            task_store[task_id]["status"] = TaskStatus.STOPPED
            task_store[task_id]["error"] = error_msg
            task_store[task_id]["finishedAt"] = datetime.now()
            task_store[task_id]["isSuccess"] = False
            if task_id in running_tasks:
                del running_tasks[task_id]
            return
        # ==============================================================

        try:
            import requests
            logger.info(f"[Task {task_id}] Initializing browser (start_url={request.startUrl})")

            browser_config = {}

            # Force non-headless mode for VNC viewing
            browser_config['headless'] = False
            logger.info(f"[Task {task_id}] Browser set to non-headless mode for VNC")

            if request.cdpUrl:
                browser_config['cdp_url'] = request.cdpUrl

            if request.startUrl:
                browser_config["start_url"] = request.startUrl

            if request.keepAlive:
                browser_config["keep_alive"] = request.keepAlive

            # ============== PROXY CONFIGURATION ==============
            proxy_config = build_proxy_config(request.proxy, request.proxyUrl)
            if proxy_config:
                browser_config['proxy'] = proxy_config
                logger.info(f"[Task {task_id}] ✓ Proxy configured: {proxy_config.get('server', 'N/A')}")
                
                # Add proxy args to browser arguments
                proxy_args = get_proxy_args(proxy_config)
                if proxy_args:
                    logger.info(f"[Task {task_id}] Proxy args: {proxy_args}")
            # =================================================

            # Handle storageStateUrl
            if request.storageStateUrl:
                try:
                    logger.info(f"[Task {task_id}] Fetching storage state from: {request.storageStateUrl}")
                    response = requests.get(request.storageStateUrl)
                    response.raise_for_status()

                    state_file_path = os.path.abspath("state.json")
                    with open(state_file_path, 'w') as f:
                        f.write(response.text)

                    browser_config["storage_state"] = state_file_path
                    logger.info(f"[Task {task_id}] ✓ Storage state saved to {state_file_path}")
                except Exception as e:
                    logger.error(f"[Task {task_id}] ✗ Storage state error: {str(e)}")
            elif request.profileDirectory:
                browser_config["profile_directory"] = request.profileDirectory
                browser_config['user_data_dir'] = "/browser-use-profile/"
                logger.info(f"[Browser Session {session_id}] Using profile directory: {request.profileDirectory}")
                
                # Start background task to restore profile from R2 (non-blocking)
                async def restore_profile_background():
                    try:
                        cloud_storage = CloudStorage()
                        if cloud_storage.client:
                            full_profile_path = f"/browser-use-profile/{request.profileDirectory}"
                            logger.info(f"[Task {task_id}] Starting background restore from R2 to {full_profile_path}...")
                            await asyncio.to_thread(
                                cloud_storage.download_directory,
                                request.profileDirectory,
                                full_profile_path
                            )
                            logger.info(f"[Task {task_id}] Background restore completed")
                    except Exception as e:
                        logger.warning(f"[Task {task_id}] Background restore failed: {e}")
                
                # Fire and forget - don't wait for restore
                asyncio.create_task(restore_profile_background())
            
            # Launch browser session
            browser_config['args'] = DEF_ARGS.copy()
            
            # Add proxy args if configured
            if proxy_config:
                proxy_args = get_proxy_args(proxy_config)
                browser_config['args'].extend(proxy_args)
            
            print(browser_config)
            os.system("pkill -f chrome")
            browser = Browser(**browser_config)
            task_store[task_id]["browser"] = browser
            task_store[task_id]["vnc_enabled"] = True
            task_store[task_id]["proxy_enabled"] = proxy_config is not None
            
            logger.info(f"[Task {task_id}] ✓ Browser initialized (VNC available at /vnc.html)")
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
                # Register custom tools if specified
        if request.toolIds:
            from browser_use import Tools
            tools = Tools()
            register_tools_with_agent(tools, [str(tid) for tid in request.toolIds], task_id)
            agent_config["tools"] = tools
            logger.info(f"[Task {task_id}] Registered {len(request.toolIds)} custom tools")

        if request.maxSteps:
            agent_config["max_steps"] = request.maxSteps

        logger.info(f"[Task {task_id}] Creating agent with config: {list(agent_config.keys())}")
        agent = Agent(**agent_config)
        logger.info(f"[Task {task_id}] ✓ Agent created successfully")

        # ✅ Helper function to extract step data
        def extract_step_data(item, idx):
            """Extract step data from history item with multiple fallback strategies"""
            try:
                # Initialize empty values
                memory = ""
                eval_prev = ""
                next_goal = ""
                url = ""
                actions = []
                
                # Strategy 1: Check model_output (most common in browser-use)
                if hasattr(item, 'model_output') and item.model_output:
                    model_output = item.model_output
                    
                    # Extract current_state fields
                    if hasattr(model_output, 'current_state') and model_output.current_state:
                        current_state = model_output.current_state
                        memory = str(getattr(current_state, 'memory', ''))
                        eval_prev = str(getattr(current_state, 'evaluation_previous_goal', ''))
                        next_goal = str(getattr(current_state, 'next_goal', ''))
                    
                    # Extract action
                    if hasattr(model_output, 'action') and model_output.action:
                        action = model_output.action
                        if isinstance(action, list):
                            actions = [str(a) for a in action]
                        else:
                            # Action might have a name/description
                            
                            action_str = str(getattr(action, 'name', '') or getattr(action, 'action', '') or action)
                            if action_str:
                                actions = [action_str]
                
                # Strategy 2: Check for direct state attribute
                if hasattr(item, 'state') and item.state:
                    state = item.state
                    if not memory:
                        memory = str(getattr(state, 'memory', ''))
                    if not eval_prev:
                        eval_prev = str(getattr(state, 'evaluation_previous_goal', ''))
                    if not next_goal:
                        next_goal = str(getattr(state, 'next_goal', ''))
                
                # Strategy 3: Check direct attributes on item
                if not memory and hasattr(item, 'memory'):
                    memory = str(item.memory)
                if not eval_prev and hasattr(item, 'evaluation_previous_goal'):
                    eval_prev = str(item.evaluation_previous_goal)
                if not next_goal and hasattr(item, 'next_goal'):
                    next_goal = str(item.next_goal)
                
                # Extract URL
                if hasattr(item, 'state') and hasattr(item.state, 'url'):
                    url = str(item.state.url)
                elif hasattr(item, 'url'):
                    url = str(item.url)
                
                # Extract actions if not already found
                if not actions:
                    if hasattr(item, 'action'):
                        action = item.action
                        if isinstance(action, list):
                            actions = [str(a) for a in action]
                        elif action:
                            actions = [str(action)]
                    elif hasattr(item, 'result') and hasattr(item.result, 'extracted_content'):
                        actions = [str(a) for a in item.result.extracted_content]
                
                # Debug logging for first item
                if idx == 0:
                    logger.info(f"[Task {task_id}] === Step {idx} Debug Info ===")
                    logger.info(f"[Task {task_id}] Item type: {type(item).__name__}")
                    logger.info(f"[Task {task_id}] Item attrs: {[a for a in dir(item) if not a.startswith('_')]}")
                    
                    if hasattr(item, 'model_output') and item.model_output:
                        logger.info(f"[Task {task_id}] model_output type: {type(item.model_output).__name__}")
                        logger.info(f"[Task {task_id}] model_output attrs: {[a for a in dir(item.model_output) if not a.startswith('_')]}")
                        
                        if hasattr(item.model_output, 'current_state'):
                            logger.info(f"[Task {task_id}] current_state: {item.model_output.current_state}")
                        if hasattr(item.model_output, 'action'):
                            logger.info(f"[Task {task_id}] action: {item.model_output.action}")
                    
                    logger.info(f"[Task {task_id}] Extracted - memory: {memory[:50]}, goal: {next_goal[:50]}, actions: {len(actions)}")
                
                return {
                    "memory": memory,
                    "evaluationPreviousGoal": eval_prev,
                    "nextGoal": next_goal,
                    "url": url,
                    "actions": actions
                }
                
            except Exception as e:
                logger.error(f"[Task {task_id}] Error extracting step {idx}: {str(e)}")
                return {
                    "memory": "",
                    "evaluationPreviousGoal": "",
                    "nextGoal": "",
                    "url": "",
                    "actions": []
                }

        # ✅ NEW: Create background task to monitor agent progress in real-time
        async def monitor_agent_progress():
            """Poll agent history and update task_store in real-time"""
            last_step_count = 0
            while task_store[task_id]["status"] in [TaskStatus.STARTED, TaskStatus.PAUSED]:
                try:
                    # Access agent's history if available
                    if hasattr(agent, 'history') and agent.history:
                        current_history = agent.history
                        
                        # Check if new steps were added
                        if hasattr(current_history, 'history') and len(current_history.history) > last_step_count:
                            # Update steps in task_store
                            steps = []
                            for idx, item in enumerate(current_history.history):
                                step_data = extract_step_data(item, idx)
                                
                                step = TaskStepView(
                                    number=idx + 1,
                                    memory=step_data["memory"],
                                    evaluationPreviousGoal=step_data["evaluationPreviousGoal"],
                                    nextGoal=step_data["nextGoal"],
                                    url=step_data["url"],
                                    screenshotUrl=None,
                                    actions=step_data["actions"]
                                )
                                steps.append(step.dict())
                            
                            task_store[task_id]["steps"] = steps
                            last_step_count = len(current_history.history)
                            logger.info(f"[Task {task_id}] Updated with {last_step_count} steps")
                    
                    await asyncio.sleep(0.5)  # Poll every 500ms
                except Exception as e:
                    logger.error(f"[Task {task_id}] Error monitoring progress: {str(e)}")
                    await asyncio.sleep(1)

        # Start monitoring task
        monitor_task = asyncio.create_task(monitor_agent_progress())

        start_time = datetime.now()

        try:
            logger.info(f"[Task {task_id}] Starting agent.run() - executing browser automation...")
            history = await agent.run()
            logger.info(f"[Task {task_id}] ✓ Agent.run() completed successfully")

            # Cancel monitoring task
            if monitor_task:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

            if task_id in running_tasks:
                del running_tasks[task_id]

            if task_store[task_id]["status"] != TaskStatus.STOPPED:
                task_store[task_id]["status"] = TaskStatus.FINISHED
                task_store[task_id]["finishedAt"] = datetime.now()
                task_store[task_id]["isSuccess"] = True
                logger.info(f"[Task {task_id}] Task finished successfully")

            # ✅ Extract final steps with improved logic
            steps = []
            if hasattr(history, 'history') and history.history:
                for idx, item in enumerate(history.history):
                    step_data = extract_step_data(item, idx)
                    
                    step = TaskStepView(
                        number=idx + 1,
                        memory=step_data["memory"],
                        evaluationPreviousGoal=step_data["evaluationPreviousGoal"],
                        nextGoal=step_data["nextGoal"],
                        url=step_data["url"],
                        screenshotUrl=str(getattr(item, 'screenshot_url', None)) if hasattr(item, 'screenshot_url') else None,
                        actions=step_data["actions"]
                    )
                    steps.append(step.dict())

            task_store[task_id]["steps"] = steps
            
            # ✅ Extract final output (structured or text)
            final_output = history.final_result() if hasattr(history, 'final_result') else str(history)
            task_store[task_id]["output"] = final_output

        except asyncio.CancelledError:
            if monitor_task:
                monitor_task.cancel()
            logger.warning(f"[Task {task_id}] Task was cancelled")
            if task_id in running_tasks:
                del running_tasks[task_id]
            if task_store[task_id]["status"] != TaskStatus.STOPPED:
                task_store[task_id]["status"] = TaskStatus.STOPPED
                task_store[task_id]["finishedAt"] = datetime.now()
                task_store[task_id]["isSuccess"] = False
            raise

        except Exception as e:
            if monitor_task:
                monitor_task.cancel()
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
        # Restore original environment variables
        for env_var, original_value in original_env_vars.items():
            if original_value is not None:
                os.environ[env_var] = original_value
            elif env_var in os.environ:
                del os.environ[env_var]
        
        if task_id in task_store and "browser" in task_store[task_id]:
            del task_store[task_id]["browser"]
        # ✅ Always close the browser explicitly
        if browser is not None:
            try:
                await browser.stop()
                logger.info(f"[Task {task_id}] ✓ Browser closed successfully")
            except Exception as e:
                logger.warning(f"[Task {task_id}] Failed to close browser: {str(e)}")
        
        # Backup profile to cloud storage if used
        if request.profileDirectory:
            try:
                cloud_storage = CloudStorage()
                if cloud_storage.client:
                    full_profile_path = f"/browser-use-profile/{request.profileDirectory}"
                    logger.info(f"[Task {task_id}] Backing up profile from {full_profile_path} to cloud storage...")
                    # Run in thread pool to avoid blocking event loop
                    await asyncio.to_thread(
                        cloud_storage.upload_directory,
                        full_profile_path,
                        request.profileDirectory
                    )
            except Exception as e:
                logger.warning(f"[Task {task_id}] Failed to backup profile to cloud storage: {e}")
    
        # Clean up state file if it was created
        if request.storageStateUrl:
            state_file_path = os.path.abspath(f"state_{task_id}.json")
            try:
                if os.path.exists(state_file_path):
                    os.remove(state_file_path)
                    logger.info(f"[Task {task_id}] ✓ Cleaned up state file")
            except Exception as e:
                logger.warning(f"[Task {task_id}] Failed to clean up state file: {str(e)}")

@app.get("/tasks/{task_id}/vnc", response_model=VncResponse, tags=["Tasks"])
async def get_vnc_port(task_id: UUID4):
    """
    Get VNC connection details for watching the browser in real-time
    """
    task_data = task_store.get(str(task_id))
    
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check if task is running
    if task_data["status"] not in [TaskStatus.STARTED, TaskStatus.PAUSED]:
        raise HTTPException(status_code=400, detail="Task is not currently running")
    
    # Return VNC URL using the same port through nginx reverse proxy
    return VncResponse(
        url="/vnc.html?autoconnect=true&path=websockify",
        status="available",
        display=os.environ.get('DISPLAY', ':99')
    )

@app.get("/vnc/health", tags=["VNC"])
async def vnc_health_check():
    """Check if VNC services are running"""
    import subprocess
    
    checks = {}
    
    services = {
        'xvfb': 'Xvfb',
        'x11vnc': 'x11vnc',
        'websockify': 'websockify',
        'nginx': 'nginx'
    }
    
    for name, process in services.items():
        try:
            result = subprocess.run(['pgrep', '-f', process], capture_output=True)
            checks[name] = result.returncode == 0
        except:
            checks[name] = False
    
    all_healthy = all(checks.values())
    # Tool endpoints - see custom_tools.py for implementation
    return {
        "healthy": all_healthy,
        "services": checks,
        "vnc_url": "/vnc.html?autoconnect=true&path=websockify" if all_healthy else None,
        "display": os.environ.get('DISPLAY', 'Not set')
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
# ============== Custom Tools Endpoints ==============

@app.post("/tools", response_model=ToolView, status_code=201, tags=["Tools"])
async def create_tool(request: CreateToolRequest):
    """Register a new custom tool that browser agents can use."""
    return create_tool_func(request)

@app.get("/tools", response_model=ToolListResponse, tags=["Tools"])
async def list_tools():
    """List all registered custom tools"""
    return list_tools_func()

@app.get("/tools/{tool_id}", response_model=ToolView, tags=["Tools"])
async def get_tool(tool_id: UUID4):
    """Get a specific tool by ID"""
    tool = get_tool_func(str(tool_id))
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool

@app.delete("/tools/{tool_id}", status_code=204, tags=["Tools"])
async def delete_tool(tool_id: UUID4):
    """Delete a registered tool"""
    if not delete_tool_func(str(tool_id)):
        raise HTTPException(status_code=404, detail="Tool not found")
    return None

# =====================================================
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
        "proxyEnabled": request.proxy is not None or request.proxyUrl is not None,
        "apiKeyProvided": request.apiKey is not None or request.apiKeys is not None,
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

@app.get("/tasks/{task_id}/debug", tags=["Tasks"])
async def debug_task(task_id: UUID4):
    """Debug endpoint to see raw history structure"""
    task_data = task_store.get(str(task_id))
    
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "task_id": str(task_id),
        "status": task_data["status"],
        "raw_output": task_data.get("output"),
        "steps_count": len(task_data.get("steps", [])),
        "raw_steps": task_data.get("steps", []),
        "has_browser": "browser" in task_data,
        "vnc_enabled": task_data.get("vnc_enabled", False),
        "proxy_enabled": task_data.get("proxy_enabled", False),
        "api_key_provided": task_data.get("apiKeyProvided", False)
    }

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

# ---- Browser Session API (updated to match Browser / Page documentation) ----

import json
import tempfile
from pathlib import Path

class BrowserSessionRequest(BaseModel):
    profileDirectory: Optional[str] = Field(default=None, description="Profile name for the browser session")
    storageStateUrl: Optional[str] = Field(default=None, description="URL to fetch storage state from")
    storageStateJson: Optional[Dict[str, Any]] = Field(default=None, description="Direct storage state JSON")
    startUrl: Optional[str] = Field(default="https://google.com", description="Initial URL to navigate to")
    headless: Optional[bool] = Field(default=False, description="Run browser in headless mode")
    keepAlive: Optional[bool] = Field(default=True, description="Keep browser alive after opening")
    viewport: Optional[Dict[str, int]] = Field(default={"width": 1920, "height": 1080})
    # ============== PROXY FIELDS ==============
    proxy: Optional[ProxyConfig] = Field(default=None, description="Proxy configuration for the browser")
    proxyUrl: Optional[str] = Field(default=None, description="Simple proxy URL (e.g., http://user:pass@proxy:8080)")
    # ==========================================

class BrowserSessionResponse(BaseModel):
    sessionId: UUID4
    status: str
    vncUrl: str
    startUrl: str
    profileDirectory: Optional[str] = None
    createdAt: datetime
    proxyEnabled: bool = False

class BrowserSessionInfo(BaseModel):
    sessionId: UUID4
    status: str
    profileDirectory: Optional[str]
    startUrl: str
    createdAt: datetime
    vncUrl: str
    isActive: bool
    proxyEnabled: bool = False

# Browser session storage
browser_sessions: Dict[str, Dict[str, Any]] = {}

async def create_browser_profile(profile_directory: str, storage_state: Optional[Dict] = None) -> Path:
    """Create a browser profile directory with optional storage state"""
    profiles_dir = Path("/browser-use-profile")
    profiles_dir.mkdir(exist_ok=True)
    
    profile_path = profiles_dir / profile_directory
    profile_path.mkdir(exist_ok=True)
    
    if storage_state is not None:
        state_file = profile_path / "state.json"
        with open(state_file, 'w') as f:
            json.dump(storage_state, f)
        return state_file
    
    return profile_path

@app.post("/browser/launch", response_model=BrowserSessionResponse, tags=["Browser"])
async def launch_browser(request: BrowserSessionRequest, background_tasks: BackgroundTasks):
    """
    Launch a browser instance with a specific profile for manual interaction via VNC.
    The browser will stay open until explicitly closed.
    
    Supports proxy configuration via:
    - proxy: Detailed proxy configuration object
    - proxyUrl: Simple proxy URL string (e.g., http://user:pass@proxy.example.com:8080)
    """
    session_id = uuid.uuid4()
    
    try:
        from browser_use import Browser
        import requests
        
        logger.info(f"[Browser Session {session_id}] Launching browser with profile: {request.profileDirectory}")
        
        browser_config = {
            "headless": request.headless,
            "keep_alive": request.keepAlive,
        }
        
        # ============== PROXY CONFIGURATION ==============
        proxy_config = build_proxy_config(request.proxy, request.proxyUrl)
        proxy_enabled = False
        if proxy_config:
            browser_config['proxy'] = proxy_config
            proxy_enabled = True
            logger.info(f"[Browser Session {session_id}] ✓ Proxy configured: {proxy_config.get('server', 'N/A')}")
        # =================================================
        
        # Handle storage state from various sources
        storage_state_path: Optional[Path] = None
        
        if request.storageStateUrl:
            logger.info(f"[Browser Session {session_id}] Fetching storage state from URL")
            response = requests.get(request.storageStateUrl)
            response.raise_for_status()
            storage_state = response.json()
            
            if request.profileDirectory:
                storage_state_path = await create_browser_profile(request.profileDirectory, storage_state)
            else:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(storage_state, f)
                    storage_state_path = Path(f.name)
            
            browser_config["storage_state"] = str(storage_state_path)
            
        elif request.storageStateJson:
            logger.info(f"[Browser Session {session_id}] Using provided storage state JSON")
            
            if request.profileDirectory:
                storage_state_path = await create_browser_profile(request.profileDirectory, request.storageStateJson)
            else:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(request.storageStateJson, f)
                    storage_state_path = Path(f.name)
            
            browser_config["storage_state"] = str(storage_state_path)
            
        elif request.profileDirectory:
          
            browser_config["profile_directory"] = request.profileDirectory
            browser_config['user_data_dir'] = "/browser-use-profile"
            logger.info(f"[Browser Session {session_id}] Using profile directory: {request.profileDirectory}")

            # Start background task to restore profile from R2 (non-blocking)
            async def restore_profile_background():
                try:
                    cloud_storage = CloudStorage()
                    if cloud_storage.client:
                        full_profile_path = f"/browser-use-profile/{request.profileDirectory}"
                        logger.info(f"[Browser Session {session_id}] Starting background restore from R2 to {full_profile_path}...")
                        await asyncio.to_thread(
                            cloud_storage.download_directory,
                            request.profileDirectory,
                            full_profile_path
                        )
                        logger.info(f"[Browser Session {session_id}] Background restore completed")
                except Exception as e:
                    logger.warning(f"[Browser Session {session_id}] Background restore failed: {e}")
            
            # Fire and forget - don't wait for restore
            asyncio.create_task(restore_profile_background())
        
        # Launch browser session
        browser_config['args'] = DEF_ARGS.copy()
        
        # Add proxy args if configured
        if proxy_config:
            proxy_args = get_proxy_args(proxy_config)
            browser_config['args'].extend(proxy_args)
        
        print(browser_config)
        os.system("pkill -f chrome")
        browser = Browser(**browser_config)
        await browser.start()

        # Create initial page
        start_url = request.startUrl or "https://google.com"
        page = await browser.new_page(start_url)

        # Set viewport if requested
        if request.viewport:
            try:
                await page.set_viewport_size(
                    width=request.viewport.get("width", 1920),
                    height=request.viewport.get("height", 1080),
                )
            except Exception as e:
                logger.warning(f"[Browser Session {session_id}] Failed to set viewport: {str(e)}")

        # Store session info
        browser_sessions[str(session_id)] = {
            "id": session_id,
            "browser": browser,
            "page": page,
            "profile_directory": request.profileDirectory,
            "start_url": start_url,
            "created_at": datetime.now(),
            "status": "active",
            "storage_state_path": storage_state_path,
            "proxy_enabled": proxy_enabled,
        }

        # Start periodic backup task if profile directory is used
        if request.profileDirectory:
            async def periodic_backup():
                while str(session_id) in browser_sessions:
                    await asyncio.sleep(300)  # Backup every 5 minutes
                    if str(session_id) not in browser_sessions:
                        break
                    
                    try:
                        cloud_storage = CloudStorage()
                        if cloud_storage.client:
                            full_profile_path = f"/browser-use-profile/{request.profileDirectory}"
                            logger.info(f"[Browser Session {session_id}] Performing periodic backup from {full_profile_path} to cloud storage...")
                            # Run in thread pool to avoid blocking event loop
                            await asyncio.to_thread(
                                cloud_storage.upload_directory,
                                full_profile_path,
                                request.profileDirectory
                            )
                    except Exception as e:
                        logger.warning(f"[Browser Session {session_id}] Periodic backup failed: {e}")

            backup_task = asyncio.create_task(periodic_backup())
            browser_sessions[str(session_id)]["backup_task"] = backup_task
        
        logger.info(f"[Browser Session {session_id}] ✓ Browser launched successfully (proxy: {proxy_enabled})")
        
        return BrowserSessionResponse(
            sessionId=session_id,
            status="active",
            vncUrl="/vnc.html?autoconnect=true&path=websockify",
            startUrl=start_url,
            profileDirectory=request.profileDirectory,
            createdAt=datetime.now(),
            proxyEnabled=proxy_enabled
        )
        
    except Exception as e:
        logger.error(f"[Browser Session {session_id}] Failed to launch browser: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to launch browser: {str(e)}")

@app.delete("/browser/{session_id}", tags=["Browser"])
async def close_browser_session(session_id: UUID4):
    """Close a specific browser session"""
    session = browser_sessions.get(str(session_id))
    
    if not session:
        raise HTTPException(status_code=404, detail="Browser session not found")
    
    try:
        browser = session.get("browser")
        if browser:
            await browser.stop()
            logger.info(f"[Browser Session {session_id}] ✓ Browser closed")
        
        storage_state_path = session.get("storage_state_path")
        if storage_state_path and isinstance(storage_state_path, Path):
            if "tmp" in str(storage_state_path):
                try:
                    storage_state_path.unlink()
                    logger.info(f"[Browser Session {session_id}] ✓ Temp storage state cleaned up")
                except:
                    pass
        
        # Backup profile to cloud storage if used
        profile_directory = session.get("profile_directory")
        if profile_directory:
            try:
                cloud_storage = CloudStorage()
                if cloud_storage.client:
                    full_profile_path = f"/browser-use-profile/{profile_directory}"
                    logger.info(f"[Browser Session {session_id}] Backing up profile from {full_profile_path} to cloud storage...")
                    # Run in thread pool to avoid blocking event loop
                    await asyncio.to_thread(
                        cloud_storage.upload_directory,
                        full_profile_path,
                        profile_directory
                    )
            except Exception as e:
                logger.warning(f"[Browser Session {session_id}] Failed to backup profile to cloud storage: {e}")

        # Cancel periodic backup task
        backup_task = session.get("backup_task")
        if backup_task:
            backup_task.cancel()
            try:
                await backup_task
            except asyncio.CancelledError:
                pass

        del browser_sessions[str(session_id)]
        
        return {"message": "Browser session closed successfully", "sessionId": session_id}
        
    except Exception as e:
        logger.error(f"[Browser Session {session_id}] Failed to close browser: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to close browser: {str(e)}")

@app.get("/browser/sessions", response_model=List[BrowserSessionInfo], tags=["Browser"])
async def list_browser_sessions():
    """List all active browser sessions"""
    sessions: List[BrowserSessionInfo] = []
    
    for session_id, session_data in browser_sessions.items():
        sessions.append(BrowserSessionInfo(
            sessionId=session_data["id"],
            status=session_data["status"],
            profileDirectory=session_data.get("profile_directory"),
            startUrl=session_data.get("start_url", "about:blank"),
            createdAt=session_data["created_at"],
            vncUrl="/vnc.html?autoconnect=true&path=websockify",
            isActive=session_data["status"] == "active",
            proxyEnabled=session_data.get("proxy_enabled", False)
        ))
    
    return sessions

@app.post("/browser/{session_id}/navigate", tags=["Browser"])
async def navigate_browser(session_id: UUID4, url: str = Query(..., description="URL to navigate to")):
    """Navigate an active browser session to a specific URL"""
    session = browser_sessions.get(str(session_id))
    
    if not session:
        raise HTTPException(status_code=404, detail="Browser session not found")
    
    try:
        browser = session.get("browser")
        if not browser:
            raise HTTPException(status_code=400, detail="Browser instance not available")
        
        page = session.get("page")
        if page is None:
            # fallback: try to get current pages from browser
            pages = await browser.get_pages()
            if pages:
                page = pages[0]
            else:
                page = await browser.new_page("about:blank")
            session["page"] = page

        await page.goto(url)
        logger.info(f"[Browser Session {session_id}] ✓ Navigated to {url}")
        session["start_url"] = url
        
        return {"message": f"Navigated to {url}", "sessionId": session_id}
            
    except Exception as e:
        logger.error(f"[Browser Session {session_id}] Failed to navigate: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to navigate: {str(e)}")

@app.post("/browser/{session_id}/save-state", tags=["Browser"])
async def save_browser_state(session_id: UUID4, profile_directory: Optional[str] = None):
    """
    Save the current browser state (cookies, localStorage, etc.)

    NOTE: The public Browser API in the provided docs does not expose a direct
    storage-state saving method. This endpoint currently returns 501 to indicate
    that this operation is not supported with the documented API.
    """
    session = browser_sessions.get(str(session_id))
    
    if not session:
        raise HTTPException(status_code=404, detail="Browser session not found")
    
    # Placeholder until browser_use exposes a public storage-state API
    raise HTTPException(
        status_code=501,
        detail="Saving browser storage state is not supported by the documented browser_use API"
    )

@app.get("/browser/profiles", tags=["Browser"])
async def list_browser_profiles():
    """List all saved browser profiles"""
    profiles_dir = Path("/browser-use-profile")
    
    if not profiles_dir.exists():
        return {"profiles": []}
    
    profiles = []
    for profile_dir in profiles_dir.iterdir():
        if profile_dir.is_dir():
            state_file = profile_dir / "state.json"
            profile_info = {
                "name": profile_dir.name,
                "hasState": state_file.exists(),
                "created": datetime.fromtimestamp(profile_dir.stat().st_ctime) if profile_dir.exists() else None
            }
            
            profile_info["inUse"] = any(
                s.get("profile_directory") == profile_dir.name 
                for s in browser_sessions.values()
            )
            
            profiles.append(profile_info)
    
    return {"profiles": profiles}

@app.delete("/browser/profiles/{profile_directory}", tags=["Browser"])
async def delete_browser_profile(profile_directory: str):
    """Delete a saved browser profile"""
    if any(s.get("profile_directory") == profile_directory for s in browser_sessions.values()):
        raise HTTPException(status_code=400, detail="Profile is currently in use")
    
    profiles_dir = Path("/browser-use-profile")
    profile_path = profiles_dir / profile_directory
    
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")
    
    try:
        import shutil
        shutil.rmtree(profile_path)
        logger.info(f"✓ Profile deleted: {profile_directory}")
        
        # Delete from cloud storage
        try:
            cloud_storage = CloudStorage()
            if cloud_storage.client:
                logger.info(f"Deleting profile {profile_directory} from cloud storage...")
                # Run in thread pool to avoid blocking event loop
                await asyncio.to_thread(
                    cloud_storage.delete_directory,
                    profile_directory
                )
        except Exception as e:
            logger.warning(f"Failed to delete profile from cloud storage: {e}")

        return {"message": f"Profile '{profile_directory}' deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete profile {profile_directory}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete profile: {str(e)}")

# ============== PROXY TESTING ENDPOINTS ==============
@app.post("/proxy/test", tags=["Proxy"])
async def test_proxy(
    proxy: Optional[ProxyConfig] = None,
    proxyUrl: Optional[str] = Query(default=None, description="Simple proxy URL to test")
):
    """
    Test a proxy configuration by attempting to connect to a test URL.
    
    Usage examples:
    - With ProxyConfig object in body
    - With proxyUrl query parameter: /proxy/test?proxyUrl=http://user:pass@proxy:8080
    """
    import aiohttp
    
    proxy_config = build_proxy_config(proxy, proxyUrl)
    
    if not proxy_config:
        raise HTTPException(status_code=400, detail="No proxy configuration provided")
    
    test_url = "https://httpbin.org/ip"
    
    try:
        # Build proxy URL for aiohttp
        proxy_url_str = proxy_config.get("server", "")
        
        if proxy_config.get("username") and proxy_config.get("password"):
            # Parse and rebuild with auth
            from urllib.parse import urlparse
            parsed = urlparse(proxy_url_str)
            proxy_url_str = f"{parsed.scheme}://{proxy_config['username']}:{proxy_config['password']}@{parsed.netloc}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(test_url, proxy=proxy_url_str, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "proxy": proxy_config.get("server"),
                        "external_ip": data.get("origin"),
                        "test_url": test_url,
                        "response_status": response.status
                    }
                else:
                    return {
                        "success": False,
                        "proxy": proxy_config.get("server"),
                        "error": f"HTTP {response.status}",
                        "test_url": test_url
                    }
                    
    except aiohttp.ClientProxyConnectionError as e:
        return {
            "success": False,
            "proxy": proxy_config.get("server"),
            "error": f"Proxy connection failed: {str(e)}",
            "test_url": test_url
        }
    except Exception as e:
        return {
            "success": False,
            "proxy": proxy_config.get("server"),
            "error": str(e),
            "test_url": test_url
        }

@app.get("/proxy/current-ip", tags=["Proxy"])
async def get_current_ip():
    """Get the current external IP address (without proxy)"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://httpbin.org/ip", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "ip": data.get("origin"),
                        "source": "httpbin.org"
                    }
    except Exception as e:
        pass
    
    # Fallback
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.ipify.org?format=json", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "ip": data.get("ip"),
                        "source": "ipify.org"
                    }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get IP: {str(e)}")
# =====================================================

# ============== API KEY VALIDATION ENDPOINTS ==============
@app.post("/api-keys/validate", tags=["API Keys"])
async def validate_api_key(
    provider: LLMProvider = Query(..., description="LLM provider to validate"),
    apiKey: Optional[str] = Query(default=None, description="API key to validate (uses env if not provided)")
):
    """
    Validate an API key for a specific provider.
    
    Tests the API key by making a minimal request to the provider's API.
    """
    # Get the API key (from request or environment)
    resolved_key = apiKey or os.environ.get(PROVIDER_ENV_VARS.get(provider, ""))
    
    if not resolved_key:
        return {
            "valid": False,
            "provider": provider.value,
            "error": f"No API key provided and {PROVIDER_ENV_VARS.get(provider)} not set in environment",
            "source": "none"
        }
    
    source = "request" if apiKey else "environment"
    
    try:
        import aiohttp
        
        # Test endpoints for each provider
        test_configs = {
            LLMProvider.OPENAI: {
                "url": "https://api.openai.com/v1/models",
                "headers": {"Authorization": f"Bearer {resolved_key}"}
            },
            LLMProvider.ANTHROPIC: {
                "url": "https://api.anthropic.com/v1/messages",
                "headers": {
                    "x-api-key": resolved_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                "method": "POST",
                "body": {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "Hi"}]
                }
            },
            LLMProvider.GOOGLE: {
                "url": f"https://generativelanguage.googleapis.com/v1/models?key={resolved_key}",
                "headers": {}
            },
            LLMProvider.GROQ: {
                "url": "https://api.groq.com/openai/v1/models",
                "headers": {"Authorization": f"Bearer {resolved_key}"}
            },
            LLMProvider.BROWSER_USE: {
                "url": "https://api.browser-use.com/v1/health",  # Placeholder
                "headers": {"Authorization": f"Bearer {resolved_key}"}
            }
        }
        
        config = test_configs.get(provider)
        if not config:
            return {
                "valid": False,
                "provider": provider.value,
                "error": "Unknown provider",
                "source": source
            }
        
        async with aiohttp.ClientSession() as session:
            method = config.get("method", "GET")
            kwargs = {
                "headers": config["headers"],
                "timeout": aiohttp.ClientTimeout(total=10)
            }
            
            if method == "POST" and "body" in config:
                kwargs["json"] = config["body"]
            
            async with session.request(method, config["url"], **kwargs) as response:
                if response.status in [200, 201]:
                    return {
                        "valid": True,
                        "provider": provider.value,
                        "source": source,
                        "key_preview": mask_api_key(resolved_key)
                    }
                elif response.status == 401:
                    return {
                        "valid": False,
                        "provider": provider.value,
                        "error": "Invalid API key (401 Unauthorized)",
                        "source": source,
                        "key_preview": mask_api_key(resolved_key)
                    }
                else:
                    # Some providers return different status codes for valid keys
                    # (e.g., 400 for bad request but valid auth)
                    error_text = await response.text()
                    return {
                        "valid": False,
                        "provider": provider.value,
                        "error": f"HTTP {response.status}: {error_text[:200]}",
                        "source": source,
                        "key_preview": mask_api_key(resolved_key)
                    }
                    
    except Exception as e:
        return {
            "valid": False,
            "provider": provider.value,
            "error": str(e),
            "source": source,
            "key_preview": mask_api_key(resolved_key)
        }

@app.get("/api-keys/status", tags=["API Keys"])
async def get_api_keys_status():
    """
    Check which API keys are configured in the environment.
    Does not reveal the actual keys, only whether they are set.
    """
    status = {}
    
    for provider in LLMProvider:
        env_var = PROVIDER_ENV_VARS.get(provider)
        if env_var:
            value = os.environ.get(env_var)
            status[provider.value] = {
                "env_var": env_var,
                "configured": value is not None and len(value) > 0,
                "key_preview": mask_api_key(value) if value else None
            }
    
    return {
        "providers": status,
        "note": "API keys can be overridden per-request using the 'apiKey' or 'apiKeys' fields"
    }
# ==========================================================

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Browser-Use API v2",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "Tasks": {
                "POST /tasks": "Create a new browser automation task",
                "GET /tasks": "List all tasks with pagination",
                "GET /tasks/{task_id}": "Get detailed task information",
                "PATCH /tasks/{task_id}": "Update task (stop, pause, resume)",
                "GET /tasks/{task_id}/logs": "Get task execution logs",
                "GET /tasks/{task_id}/vnc": "Get VNC connection for task"
            },
            "Browser Sessions": {
                "POST /browser/launch": "Launch a browser with specific profile",
                "GET /browser/sessions": "List all active browser sessions",
                "DELETE /browser/{session_id}": "Close a browser session",
                "POST /browser/{session_id}/navigate": "Navigate browser to URL",
                "POST /browser/{session_id}/save-state": "Save current browser state (not yet supported via public API)",
                "GET /browser/profiles": "List saved browser profiles",
                "DELETE /browser/profiles/{profile_directory}": "Delete a browser profile"
            },
            "Proxy": {
                "POST /proxy/test": "Test a proxy configuration",
                "GET /proxy/current-ip": "Get current external IP address"
            },
            "API Keys": {
                "POST /api-keys/validate": "Validate an API key for a provider",
                "GET /api-keys/status": "Check which API keys are configured"
            },
            "System": {
                "GET /health": "Health check",
                "GET /vnc/health": "VNC services health check"
            }
        },
        "proxy_support": {
            "description": "All browser endpoints support proxy configuration",
            "options": [
                {
                    "field": "proxy",
                    "type": "ProxyConfig object",
                    "example": {
                        "server": "http://proxy.example.com:8080",
                        "username": "user",
                        "password": "pass",
                        "bypass": ["localhost", "*.internal.com"]
                    }
                },
                {
                    "field": "proxyUrl",
                    "type": "string",
                    "example": "http://user:pass@proxy.example.com:8080"
                }
            ],
            "supported_types": ["http", "https", "socks4", "socks5"]
        },
        "api_key_support": {
            "description": "API keys can be provided per-request or via environment variables",
            "priority": [
                "1. Direct apiKey field in request (highest priority)",
                "2. Provider-specific key in apiKeys object",
                "3. Environment variable (fallback)"
            ],
            "options": [
                {
                    "field": "apiKey",
                    "type": "string",
                    "description": "Single API key for the main LLM"
                },
                {
                    "field": "apiKeys",
                    "type": "APIKeyConfig object",
                    "example": {
                        "openai": "sk-...",
                        "anthropic": "sk-ant-...",
                        "google": "AIza...",
                        "groq": "gsk_...",
                        "browserUse": "bu-..."
                    }
                },
                {
                    "field": "pageExtractionApiKey",
                    "type": "string",
                    "description": "Separate API key for page extraction LLM (optional)"
                }
            ],
            "environment_variables": {
                "OPENAI_API_KEY": "OpenAI API key",
                "ANTHROPIC_API_KEY": "Anthropic API key",
                "GOOGLE_API_KEY": "Google AI API key",
                "GROQ_API_KEY": "Groq API key",
                "BROWSER_USE_API_KEY": "Browser-Use API key"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)


