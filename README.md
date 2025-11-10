# Browser-Use REST API Wrapper

A REST API wrapper compatible with the [Browser-Use Cloud API v2 specification](https://docs.browser-use.com), enabling AI-powered browser automation through a standardized REST interface.

## Features

- ✅ **Browser-Use Cloud API v2 compatibility** - Drop-in replacement for the cloud service
- 🚀 RESTful API with standardized endpoints (`/tasks`, `/tasks/{task_id}`, etc.)
- 🤖 Support for 15+ LLM models (GPT-4.1, O4, Gemini 2.5, Claude Sonnet 4, etc.)
- ⚡ Asynchronous task execution with background processing
- 🎮 Task control actions (stop, pause, resume, stop_task_and_session)
- 📊 Comprehensive task tracking with step-by-step history
- 🔐 Secure API key management via environment variables
- 📝 Interactive API documentation with Swagger UI
- 🌐 Session management for multi-task workflows
- 🔍 Detailed execution logging with task ID tracking
- ⚠️ Error reporting with error messages in API responses

## Quick Start

### Prerequisites

- Python 3.11+
- pip package manager

### Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Chromium browser (required for browser automation):

```bash
python -m playwright install chromium
```

4. Create a `.env` file:

```bash
cp .env.example .env
```

5. Add your API keys to the `.env` file (optional - use browser-use-llm without keys)

### Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

The API will be available at `http://localhost:5000`

## Docker Deployment

> 📖 **For comprehensive deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)**

### Using Docker

The easiest way to run the API is using Docker, which includes all dependencies and Chromium pre-installed.

#### Option 1: Using Docker Compose (Recommended)

1. Clone this repository
2. Create a `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Start the container:

```bash
docker-compose up -d
```

The API will be available at `http://localhost:5000`

4. View logs:

```bash
docker-compose logs -f
```

5. Stop the container:

```bash
docker-compose down
```

#### Option 2: Using Docker directly

1. Build the image:

```bash
docker build -t browser-use-api .
```

2. Run the container:

```bash
docker run -d \
  -p 5000:5000 \
  -e BROWSER_USE_API_KEY=your-key-here \ #Optional - Make sure to call llm in request body
  -e OPENAI_API_KEY=your-key-here \
  --name browser-use-api \
  browser-use-api
```

#### Option 3: Using pre-built image from Docker

```bash
docker pull reqeique/browser-use-api:latest

docker run -d \
  -p 5000:5000 \
  -e BROWSER_USE_API_KEY=your-key-here \ #Optional - Make sure to call llm in request body
  -e OPENAI_API_KEY=your-key-here \
  --name browser-use-api \
  reqeique/browser-use-api:latest
```

### Docker Environment Variables

Pass environment variables to the container using the `-e` flag or a `.env` file:

- `BROWSER_USE_API_KEY` - Browser-Use LLM API key
- `OPENAI_API_KEY` - OpenAI API key for GPT models
- `ANTHROPIC_API_KEY` - Anthropic API key for Claude models
- `GOOGLE_API_KEY` - Google API key for Gemini models

### Docker Health Check

The container includes a health check that monitors the `/health` endpoint:

```bash
docker inspect --format='{{.State.Health.Status}}' browser-use-api
```

## CI/CD with GitHub Actions

This project includes a GitHub Actions workflow that automatically builds and publishes Docker images to Docker Container Registry.

### Automatic Builds

The workflow triggers on:
- **Push to main/master branch** - Builds and tags as `latest`
- **Git tags** - Builds and tags with version numbers (e.g., `v1.0.0`)
- **Pull requests** - Builds but doesn't push (validation only)

### Image Tags

Images are automatically tagged with:
- `latest` - Latest commit on default branch
- `main` or `master` - Latest commit on respective branch
- `v1.2.3` - Semantic version from git tags
- `v1.2` - Major.minor version
- `v1` - Major version only
- `main-sha-abc1234` - Branch + commit SHA

### Setup GitHub Actions

1. The workflow is already configured in `.github/workflows/docker-build.yml`
2. Push your code to GitHub
3. Images will be automatically published to `docker.com/reqeique/browser-use-api`

No additional configuration needed! GitHub Actions uses the built-in `GITHUB_TOKEN` for authentication.

### Multi-Architecture Support

The Docker images are built for both:
- `linux/amd64` (Intel/AMD)
- `linux/arm64` (ARM/Apple Silicon)

## API Endpoints

This API is fully compatible with the Browser-Use Cloud API v2 specification:

### Task Management

- `POST /tasks` - Create a new browser automation task (returns 202 Accepted)
- `GET /tasks` - List all tasks with pagination and filtering
- `GET /tasks/{task_id}` - Get detailed task information including steps
- `PATCH /tasks/{task_id}` - Update task (stop, pause, resume)
- `GET /tasks/{task_id}/logs` - Get task execution logs

### System

- `GET /` - API information and version
- `GET /health` - Health check endpoint

## Interactive Documentation

- **Swagger UI**: `http://localhost:5000/docs`
- **ReDoc**: `http://localhost:5000/redoc`

## Usage Examples

### 1. Create a Task

```bash
curl -X POST "http://localhost:5000/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Find the number of stars of the browser-use repo on GitHub",
    "llm": "gemini-flash-lite-latest",
    "maxSteps": 50
  }'
```

Response (202 Accepted):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "sessionId": "660e8400-e29b-41d4-a716-446655440001"
}
```

### 2. Get Task Status

```bash
curl -X GET "http://localhost:5000/tasks/550e8400-e29b-41d4-a716-446655440000"
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "sessionId": "660e8400-e29b-41d4-a716-446655440001",
  "llm": "browser-use-llm",
  "task": "Find the number of stars of the browser-use repo on GitHub",
  "status": "started",
  "startedAt": "2025-11-10T10:00:00Z",
  "finishedAt": null,
  "metadata": {},
  "steps": [
    {
      "number": 1,
      "memory": "Navigating to GitHub",
      "evaluationPreviousGoal": "None",
      "nextGoal": "Search for browser-use repository",
      "url": "https://github.com",
      "screenshotUrl": null,
      "actions": ["navigate", "search"]
    }
  ],
  "output": null,
  "outputFiles": [],
  "browserUseVersion": "0.1.17",
  "isSuccess": null,
  "error": null
}
```

### 3. List All Tasks

```bash
curl -X GET "http://localhost:5000/tasks?pageSize=20&pageNumber=1&filterBy=finished"
```

Response:
```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "sessionId": "660e8400-e29b-41d4-a716-446655440001",
      "llm": "browser-use-llm",
      "task": "Find the number of stars...",
      "status": "finished",
      "startedAt": "2025-11-10T10:00:00Z",
      "finishedAt": "2025-11-10T10:01:30Z",
      "metadata": {},
      "output": "The browser-use repo has 3,245 stars.",
      "browserUseVersion": "0.1.17",
      "isSuccess": true,
      "error": null
    }
  ],
  "totalItems": 1,
  "pageNumber": 1,
  "pageSize": 20
}
```

### 4. Stop a Running Task

```bash
curl -X PATCH "http://localhost:5000/tasks/550e8400-e29b-41d4-a716-446655440000" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "stop"
  }'
```

### 5. Reuse Session Across Tasks

```bash
curl -X POST "http://localhost:5000/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Now search for the latest release",
    "llm": "browser-use-llm",
    "sessionId": "660e8400-e29b-41d4-a716-446655440001"
  }'
```

## Request Schema

### CreateTaskRequest

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| task | string | Yes | - | The task for the agent to perform |
| llm | string | No | browser-use-llm | LLM model to use (see supported models) |
| startUrl | string | No | null | Starting URL for the browser |
| maxSteps | integer | No | 100 | Maximum steps the agent can take |
| structuredOutput | string | No | null | Schema for structured output |
| sessionId | UUID | No | null | Reuse existing browser session |
| metadata | object | No | {} | Custom metadata for the task |
| secrets | object | No | null | Task-specific secrets |
| allowedDomains | array | No | null | Restrict browsing to specific domains |
| opVaultId | string | No | null | 1Password vault ID |
| highlightElements | boolean | No | false | Highlight interactive elements |
| flashMode | boolean | No | false | Enable flash mode for faster execution |
| thinking | boolean | No | false | Enable thinking mode |
| vision | boolean/string | No | auto | Vision mode (true/false/auto) |
| systemPromptExtension | string | No | null | Additional system prompt |

## Supported LLM Models

This API supports all models from the Browser-Use Cloud API v2 specification:

### Browser-Use Native
- `browser-use-llm` (default, $10 free at [cloud.browser-use.com](https://cloud.browser-use.com/new-api-key))

### OpenAI Models
- `gpt-4.1` - Latest GPT-4 model
- `gpt-4.1-mini` - Smaller GPT-4 variant
- `gpt-4o` - GPT-4 Optimized
- `gpt-4o-mini` - GPT-4 Optimized Mini
- `o4-mini` - O-series Mini
- `o3` - O-series model

### Google Gemini Models
- `gemini-2.5-flash` - Gemini 2.5 Flash
- `gemini-2.5-pro` - Gemini 2.5 Pro
- `gemini-flash-latest` - Latest Gemini Flash
- `gemini-flash-lite-latest` - Latest Gemini Flash Lite

### Anthropic Claude Models
- `claude-sonnet-4-20250514` - Claude Sonnet 4
- `claude-sonnet-4-5-20250929` - Claude Sonnet 4.5
- `claude-3-7-sonnet-20250219` - Claude 3.7 Sonnet

### Other Models
- `llama-4-maverick-17b-128e-instruct` - Llama 4 Maverick

## Task Statuses

- `started` - Task is currently running
- `paused` - Task has been paused (status only, execution may continue)*
- `finished` - Task completed successfully
- `stopped` - Task was stopped by user or error

**Note**: Pause/resume functionality updates the task status but does not actually pause the underlying browser-use agent execution due to library limitations. The task will continue running in the background.

## Task Control Actions

Use the `PATCH /tasks/{task_id}` endpoint with these actions:

- `stop` - Stop the task immediately
- `pause` - Pause the task (status update only)*
- `resume` - Resume a paused task (status update only)*
- `stop_task_and_session` - Stop the task and close the browser session

## Environment Variables

Configure API keys in your `.env` file:

```env
# Browser-Use LLM (get $10 free at cloud.browser-use.com)
BROWSER_USE_API_KEY=your-browser-use-api-key

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key

# Google
GOOGLE_API_KEY=your-google-api-key
```

## Example Python Client

```python
import requests
import time

API_URL = "http://localhost:5000"

def create_task(task: str, llm: str = "browser-use-llm"):
    response = requests.post(
        f"{API_URL}/tasks",
        json={
            "task": task,
            "llm": llm,
            "maxSteps": 50
        }
    )
    return response.json()

def get_task_status(task_id: str):
    response = requests.get(f"{API_URL}/tasks/{task_id}")
    return response.json()

def wait_for_task(task_id: str):
    while True:
        status = get_task_status(task_id)
        if status["status"] in ["finished", "stopped"]:
            return status
        time.sleep(2)

# Create and run a task
result = create_task("Find the number of stars of the browser-use repo")
task_id = result["id"]
print(f"Task created: {task_id}")

# Wait for completion
final_status = wait_for_task(task_id)
print(f"Task completed: {final_status['output']}")
```

## Pagination and Filtering

The `GET /tasks` endpoint supports:

**Pagination**:
- `pageSize` (default: 20, max: 100)
- `pageNumber` (default: 1)

**Filtering**:
- `sessionId` - Filter by session ID (UUID)
- `filterBy` - Filter by task status (started, paused, finished, stopped)
- `after` - Filter tasks started after this datetime (ISO 8601)
- `before` - Filter tasks started before this datetime (ISO 8601)

Example:
```bash
curl "http://localhost:5000/tasks?pageSize=10&pageNumber=1&filterBy=finished&after=2025-11-01T00:00:00Z"
```

## Error Handling

Standard HTTP status codes:
- `200` - Success
- `202` - Accepted (task created)
- `404` - Task not found
- `422` - Validation error
- `500` - Internal server error

Error response format (HTTP errors):
```json
{
  "detail": "Error message"
}
```

Task error information (in task response):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "stopped",
  "isSuccess": false,
  "error": "Failed to initialize browser-use-llm provider: You need to set the BROWSER_USE_API_KEY environment variable. Get your key at https://cloud.browser-use.com/new-api-key"
}
```

### Debugging with Logs

The API provides comprehensive logging in the console to help you understand what's happening during task execution:

```
INFO:main:[Task 550e8400...] Starting task: Go to google.com
INFO:main:[Task 550e8400...] LLM: browser-use-llm, Max Steps: 5
INFO:main:[Task 550e8400...] Initializing LLM: module=browser_use, class=ChatBrowserUse, model=None
INFO:main:[Task 550e8400...] ✓ LLM initialized successfully
INFO:main:[Task 550e8400...] Initializing browser (start_url=None)
INFO:main:[Task 550e8400...] ✓ Browser initialized successfully
INFO:main:[Task 550e8400...] Creating agent with config: ['task', 'llm', 'browser']
INFO:main:[Task 550e8400...] ✓ Agent created successfully
INFO:main:[Task 550e8400...] Starting agent.run() - executing browser automation...
INFO:main:[Task 550e8400...] ✓ Agent.run() completed successfully
INFO:main:[Task 550e8400...] Task finished successfully with 3 steps
```

When errors occur, detailed error messages and stack traces are logged:

```
ERROR:main:[Task 550e8400...] ✗ LLM initialization failed: Failed to initialize browser-use-llm provider: You need to set the BROWSER_USE_API_KEY environment variable...
```

## Deployment Best Practices

### Production Deployment

For production deployments using Docker:

1. **Use specific version tags** instead of `latest`:
   ```bash
   docker pull reqeique/browser-use-api:v1.0.0
   ```

2. **Set resource limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
       reservations:
         cpus: '1'
         memory: 2G
   ```

3. **Use secrets management** instead of .env files:
   ```bash
   docker secret create browser_use_key /path/to/key
   ```

4. **Enable logging drivers** for centralized logs:
   ```yaml
   logging:
     driver: "json-file"
     options:
       max-size: "10m"
       max-file: "3"
   ```

5. **Monitor health checks**:
   ```bash
   docker ps --format "table {{.Names}}\t{{.Status}}"
   ```

### Scaling

For high-volume deployments:
- Use Kubernetes or Docker Swarm for orchestration
- Deploy multiple replicas behind a load balancer
- Consider adding Redis for shared state across instances
- Implement rate limiting to prevent abuse

## Differences from Cloud API

This local wrapper has the following differences from the official Browser-Use Cloud API:

1. **No Authentication** - As requested, this API does not implement authentication
2. **Pause/Resume Limitation** - Pause/resume actions update task status but don't actually pause execution
3. **Local Storage** - Tasks stored in memory (restart clears all tasks)
4. **Screenshot URLs** - Not yet implemented
5. **Output Files** - Not yet populated

## Development

### Project Structure

```
.
├── .github/
│   └── workflows/
│       └── docker-build.yml    # GitHub Actions CI/CD workflow
├── main.py                     # FastAPI application with all endpoints
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container configuration
├── docker-compose.yml          # Docker Compose configuration
├── .dockerignore              # Docker build ignore patterns
├── .env.example               # Example environment variables
├── .gitignore                 # Git ignore rules
├── README.md                  # This file (user documentation)
├── DEPLOYMENT.md              # Comprehensive deployment guide
└── replit.md                  # Project architecture and memory
```

### Running Tests

```bash
# Start the server
uvicorn main:app --reload

# In another terminal, test the API
curl http://localhost:5000/health
```

## License

MIT License - see [browser-use license](https://github.com/browser-use/browser-use/blob/main/LICENSE)

## Credits

This project is a REST API wrapper for [browser-use](https://github.com/browser-use/browser-use) by the Browser-Use team, implementing the Browser-Use Cloud API v2 specification.

## Support

- Browser-Use Documentation: [docs.browser-use.com](https://docs.browser-use.com/)
- Browser-Use Cloud API Docs: [cloud.browser-use.com/docs](https://cloud.browser-use.com/docs)
- Browser-Use GitHub: [github.com/browser-use/browser-use](https://github.com/browser-use/browser-use)
- Browser-Use Discord: [discord](https://link.browser-use.com/discord)
