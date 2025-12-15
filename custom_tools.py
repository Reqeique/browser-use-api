"""
Custom Tools Module for Browser-Use API

This module provides a REST API for registering and managing custom tools
that browser agents can use during task execution. Tools follow LangChain-compatible
JSON Schema format for parameter definitions.
"""

from pydantic import BaseModel, Field, UUID4
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import logging
import httpx

logger = logging.getLogger(__name__)

# ============== Custom Tools Models (LangChain-compatible) ==============

class ToolParameterProperty(BaseModel):
    """JSON Schema property definition for tool parameters"""
    type: str = Field(description="Parameter type: string, integer, number, boolean, object, array")
    description: str = Field(description="Description of what this parameter does")
    enum: Optional[List[str]] = Field(default=None, description="Allowed values for enum types")
    default: Optional[Any] = Field(default=None, description="Default value if not provided")

class ToolParameters(BaseModel):
    """JSON Schema for tool input parameters"""
    type: str = Field(default="object")
    properties: Dict[str, ToolParameterProperty] = Field(description="Parameter definitions")
    required: Optional[List[str]] = Field(default=None, description="List of required parameter names")

class CreateToolRequest(BaseModel):
    """Request to register a new custom tool"""
    name: str = Field(description="Tool function name (e.g., 'get_weather')")
    description: str = Field(description="Description of what the tool does - LLM uses this to decide when to call")
    parameters: ToolParameters = Field(description="JSON Schema defining the tool's input parameters")
    endpoint: str = Field(description="URL to call when tool is invoked")
    method: str = Field(default="POST", description="HTTP method: GET, POST, PUT, PATCH, DELETE")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Static headers to include in requests")
    allowedDomains: Optional[List[str]] = Field(default=None, description="Domain restrictions for this tool")
    payloadTemplate: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Template for transforming input. Use {{text}} as placeholder for the input text. Example for Gemini: {'contents': [{'parts': [{'text': '{{text}}'}]}]}"
    )

class ToolView(BaseModel):
    """Response model for tool details"""
    id: UUID4
    name: str
    description: str
    parameters: ToolParameters
    endpoint: str
    method: str
    headers: Optional[Dict[str, str]] = None
    allowedDomains: Optional[List[str]] = None
    payloadTemplate: Optional[Dict[str, Any]] = None
    createdAt: datetime

class ToolListResponse(BaseModel):
    """Response model for listing tools"""
    items: List[ToolView]
    totalItems: int

# ============== Tool Storage ==============

tool_store: Dict[str, Dict[str, Any]] = {}

# ============== Tool CRUD Functions ==============

def create_tool(request: CreateToolRequest) -> ToolView:
    """Register a new custom tool"""
    tool_id = uuid.uuid4()
    
    tool_data = {
        "id": tool_id,
        "name": request.name,
        "description": request.description,
        "parameters": request.parameters.dict(),
        "endpoint": request.endpoint,
        "method": request.method.upper(),
        "headers": request.headers,
        "allowedDomains": request.allowedDomains,
        "payloadTemplate": request.payloadTemplate,
        "createdAt": datetime.now()
    }
    
    tool_store[str(tool_id)] = tool_data
    logger.info(f"[Tools] Registered new tool: {request.name} (id={tool_id})")
    
    return ToolView(**tool_data)

def list_tools() -> ToolListResponse:
    """List all registered custom tools"""
    tools = [ToolView(**tool_data) for tool_data in tool_store.values()]
    return ToolListResponse(items=tools, totalItems=len(tools))

def get_tool(tool_id: str) -> Optional[ToolView]:
    """Get a specific tool by ID"""
    tool_data = tool_store.get(tool_id)
    if tool_data:
        return ToolView(**tool_data)
    return None

def delete_tool(tool_id: str) -> bool:
    """Delete a registered tool"""
    if tool_id in tool_store:
        tool_name = tool_store[tool_id]["name"]
        del tool_store[tool_id]
        logger.info(f"[Tools] Deleted tool: {tool_name} (id={tool_id})")
        return True
    return False

def get_tool_data(tool_id: str) -> Optional[Dict[str, Any]]:
    """Get raw tool data by ID (for internal use)"""
    return tool_store.get(tool_id)

# ============== Tool Executor Factory ==============

def _apply_template(template: Any, text_value: str) -> Any:
    """
    Recursively apply text replacement to a template.
    Replaces {{text}} placeholders with the actual text value.
    """
    if isinstance(template, str):
        return template.replace("{{text}}", text_value)
    elif isinstance(template, dict):
        return {k: _apply_template(v, text_value) for k, v in template.items()}
    elif isinstance(template, list):
        return [_apply_template(item, text_value) for item in template]
    else:
        return template

def create_tool_executor(tool_def: Dict[str, Any]):
    """
    Factory function to create a tool executor for a registered tool.
    Creates a function with a descriptive 'text' parameter that LLM agents understand.
    """
    endpoint = tool_def["endpoint"]
    method = tool_def["method"]
    static_headers = tool_def.get("headers") or {}
    tool_name = tool_def.get("name", "custom_tool")
    payload_template = tool_def.get("payloadTemplate")
    tool_description = tool_def.get("description", "")
    
    async def tool_executor(text: str) -> str:
        """
        Execute the tool with a text input.
        
        Args:
            text: The text input/prompt to send to the tool's endpoint
        
        Returns:
            Response from the tool endpoint
        """
        import json
        
        try:
            logger.info(f"[{tool_name}] Received text input: {text[:100]}...")
            
            # Apply payload template if available
            if payload_template:
                payload = _apply_template(payload_template, text)
                logger.info(f"[{tool_name}] Applied template, payload: {json.dumps(payload)[:200]}...")
            else:
                # Default: wrap text in a simple object
                payload = {"text": text}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                request_kwargs = {
                    "url": endpoint,
                    "headers": static_headers
                }
                
                # For GET requests, pass params as query string
                if method == "GET":
                    request_kwargs["params"] = payload
                else:
                    # For POST/PUT/PATCH, send as JSON body
                    request_kwargs["json"] = payload
                
                response = await client.request(method, **request_kwargs)
                
                try:
                    result = response.json()
                    return f"Status: {response.status_code}\nResponse: {json.dumps(result, indent=2)}"
                except:
                    return f"Status: {response.status_code}\nResponse: {response.text}"
                    
        except httpx.TimeoutException:
            return f"Error: Request to {endpoint} timed out after 30 seconds"
        except httpx.RequestError as e:
            return f"Error: Failed to make request to {endpoint}: {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error: {str(e)}"
    
    # Set function name for better identification
    tool_executor.__name__ = tool_name
    
    return tool_executor

def register_tools_with_agent(tools_instance, tool_ids: List[str], task_id: str = None):
    """
    Register custom tools with a Browser Use Tools instance.
    
    Args:
        tools_instance: Browser Use Tools instance
        tool_ids: List of tool IDs to register
        task_id: Optional task ID for logging
    """
    log_prefix = f"[Task {task_id}]" if task_id else "[Tools]"
    
    for tool_id in tool_ids:
        tool_def = get_tool_data(str(tool_id))
        if tool_def:
            # Create the tool executor
            executor = create_tool_executor(tool_def)
            
            # Register with Browser Use Tools
            allowed_domains = tool_def.get("allowedDomains")
            tools_instance.action(
                description=tool_def["description"],
                allowed_domains=allowed_domains
            )(executor)
            
            logger.info(f"{log_prefix} Registered tool: {tool_def['name']}")
        else:
            logger.warning(f"{log_prefix} Tool not found: {tool_id}")
