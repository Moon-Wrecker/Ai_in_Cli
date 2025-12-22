# Configuration

## Environment variables

### `GOOGLE_API_KEY` (required)

Used by `ChatGoogleGenerativeAI` to talk to Gemini.

Recommended: put it in `Ai_in_Cli/.env`:

```env
GOOGLE_API_KEY=your_key_here
```

### `WORKSPACE_DIR` (optional)

Defaults to `workspace`.

Important: this is resolved relative to the process working directory.
If you run from inside `Ai_in_Cli/`, the workspace becomes `Ai_in_Cli/workspace/`.

### `BACKUP_DIR` (optional)

Defaults to `.code_backups`.

Note: backup/restore features exist in `FileSystemManager`, but the current agent toolset does not heavily expose full snapshot restore flows.

## `Ai_in_Cli/config.py`

Key knobs:

- **`LLM_MODEL`**
  - default: `"gemini-1.5-flash-latest"`
  - used in `AIFileAssistant.setup_llm()`

- **`LLM_TEMPERATURE`**
  - default: `0.1`

- **`MEMORY_WINDOW_SIZE`**
  - default: `10`
  - the assistant uses `ConversationBufferWindowMemory(k=10)`

## Note on iterations / timeouts

`config.MAX_ITERATIONS` exists but the assistant currently hardcodes:

- `AgentExecutor(max_iterations=15)`

So changing `MAX_ITERATIONS` won’t affect the agent unless you also update `ai_assistant.py`.



