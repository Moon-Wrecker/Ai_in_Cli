# Runtime Flow (Step-by-step)

This is the “what happens when I type something?” explanation.

## 0) Preconditions

Before anything useful happens:

- `GOOGLE_API_KEY` must exist in env (usually via `Ai_in_Cli/.env`)
- dependencies installed (`pip install -r requirements.txt`)

If the key is missing, `AIFileAssistant.setup_llm()` exits the program.

## 1) Startup (console mode)

When you run:

```bash
cd Ai_in_Cli
python app.py
```

The program:

- prints a banner
- creates/prints the workspace path
- calls `launch_console_interface()`
- imports and instantiates `AIFileAssistant`
- enters an interactive prompt loop (`run_interactive()`)

## 2) Assistant boot (inside `AIFileAssistant.__init__`)

Boot does four big things:

### A) Build sandboxed managers

- `FileSystemManager(base_dir=<workspace>)`
- `SmartFileEditor(base_dir=<workspace>)`
- `WorkspaceRAG(<workspace>)`
- `TerminalManager(safe_mode=True)`

### B) Index workspace (RAG)

`WorkspaceRAG.index_workspace()`:

- walks all files under `workspace/`
- skips hidden files/directories (except its own `.rag_cache`)
- chunks text files; chunks Python by top-level `class`/`def` boundaries
- stores:
  - `workspace/.rag_cache/file_index.json`
  - `workspace/.rag_cache/chunks.json`
- if ChromaDB is available, also persists vectors in `workspace/.rag_cache/chroma_db/`

### C) Configure the LLM + memory

- LLM: `ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)`
- Memory: `ConversationBufferWindowMemory(k=10)`

### D) Register tools + create agent

The assistant creates many tools, e.g.:

- file: `list_directory`, `read_file`, `create_file`, `write_file`, `append_file`, `delete_file`
- editor: `read_file_with_lines`, `insert_lines`, `replace_lines`, `delete_lines`, `find_and_replace`, `analyze_file_structure`
- rag: `search_workspace`, `get_file_context`, `workspace_overview`, `get_related_code`
- terminal: `execute_command`, `check_command_safety`, `get_system_info`, `suggest_commands`

Then it creates a ReAct agent with a strict tool-calling prompt and wraps it in `AgentExecutor`.

## 3) User input → agent loop

When you type:

> “Create a file notes.txt with hello”

the agent will typically:

- decide it needs `create_file`
- call tool `create_file` with input `notes.txt|hello`
- observe the result (`✅ File created: ...`)
- then generate a final response

Important: tool inputs must be **one-line**. The prompt explicitly warns the model to avoid newlines/extra formatting in `Action Input`.

## 4) File safety (workspace sandbox)

Even if you ask for `/etc/passwd`:

- `FileSystemManager._resolve_path()` refuses to escape `base_dir`
- `SmartFileEditor._resolve_path()` does the same

So tools will error rather than touching non-workspace files.

## 5) Terminal safety (safe mode)

Terminal execution is gated:

- `TerminalManager.is_command_safe()` allows only a set of “safe” command prefixes
- it also blocks patterns like `rm -rf`, `shutdown`, pipes-to-shell, and operators like `&&` or `;`

So the assistant can run read-only/helpful commands, but many chained or destructive operations are blocked.

## 6) Response formatting

The console UI prints:

- your message
- then the assistant’s final answer (which may include tool outputs formatted as text)

The internal step-by-step tool reasoning is not shown because the agent executor uses `verbose=False` in `Ai_in_Cli/ai_assistant.py`.



