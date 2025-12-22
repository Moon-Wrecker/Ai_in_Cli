# Architecture (Deep Dive)

## Key entrypoints

### Console entrypoint

- `Ai_in_Cli/app.py`
  - Prints a Rich banner
  - Instantiates `AIFileAssistant` from `ai_assistant.py`
  - Calls `assistant.run_interactive()`

### Typer CLI entrypoint (built into the assistant module)

- `Ai_in_Cli/ai_assistant.py`
  - Defines a Typer app with commands:
    - `interactive`
    - `execute "<natural language command>"`
    - `setup` (writes `.env`)
    - `list_tools`

### Textual UI entrypoints (optional)

- `Ai_in_Cli/ai_cli_simple.py`
- `Ai_in_Cli/ai_cli_beautiful.py`
- `Ai_in_Cli/beautiful_ai_cli.py`

These provide a UI shell but ultimately call into the same `AIFileAssistant.process_command(...)`.

## Core class: `AIFileAssistant`

Defined in `Ai_in_Cli/ai_assistant.py`.

### Initialization

When you create `AIFileAssistant()` it:

- chooses a workspace directory:
  - `workspace_dir = Path(workspace_dir or config.WORKSPACE_DIR)`
- enforces sandboxing by building tool objects with base_dir restrictions:
  - `FileSystemManager(base_dir=<workspace>)`
  - `SmartFileEditor(base_dir=<workspace>)`
  - `WorkspaceRAG(<workspace>)`
- initializes safe-mode terminal execution:
  - `TerminalManager(safe_mode=True)`
- indexes the workspace right away:
  - `self.rag_system.index_workspace()`
- configures the LLM + agent:
  - `setup_llm()` → `ChatGoogleGenerativeAI(model=config.LLM_MODEL, ...)`
  - `setup_memory()` → `ConversationBufferWindowMemory(k=10)`
  - `setup_tools()` → builds a list of LangChain `Tool(...)`
  - `setup_agent()` → builds a ReAct agent and `AgentExecutor`

### Runtime loop

Once running, the assistant uses:

- `process_command(user_input: str) -> str`
  - calls `self.agent_executor.invoke({"input": user_input})`
  - returns `response["output"]`

## “Tools” concept (LangChain)

The assistant registers a set of tools, each tool being:

- a name (e.g. `"read_file"`)
- a description (used by the LLM to decide if/when to call it)
- a function callback (Python function executed in-process)

The LLM chooses tools while following a strict schema in the prompt:

- `Action: <tool_name>`
- `Action Input: <one-line input>` (no extra formatting; no newlines)

## Component responsibilities

### `FileSystemManager` (`tools/file_fol_tools.py`)

- Resolves paths with base_dir sandbox enforcement
- Reads files with encoding fallbacks
- Creates files/directories
- Provides a directory-tree view (`folder_structure`)

### `SmartFileEditor` (`tools/smart_editor.py`)

- Provides **line-numbered read**
- Insert/replace/delete by line range
- Find-and-replace (plain or regex)
- Lightweight structure analysis for Python/JS/TS

### `WorkspaceRAG` (`tools/rag_system.py`)

- Indexes the workspace into chunks
- Stores metadata in `workspace/.rag_cache/`
- Optionally uses ChromaDB for vector search; otherwise uses substring fallback search

### `TerminalManager` (`tools/terminal_manager.py`)

- Detects OS/shell
- Blocks dangerous commands/patterns in safe mode
- Executes allowed commands with timeouts and captures stdout/stderr



