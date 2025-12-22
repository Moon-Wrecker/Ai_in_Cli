# Tools (What the AI can do)

This app’s “powers” are a set of LangChain tools registered in `AIFileAssistant.setup_tools()` (`Ai_in_Cli/ai_assistant.py`).

Below is what they do, and which lower-level module implements them.

## File operations (workspace-only)

- **`list_directory`**
  - **Does**: Returns a readable folder tree (📁/📄)
  - **Impl**: `FileSystemManager.folder_structure()` (`tools/file_fol_tools.py`)

- **`read_file`**
  - **Does**: Reads a file’s content (tries common encodings; returns bytes if binary)
  - **Impl**: `FileSystemManager.reader_func()`

- **`create_file`**
  - **Input**: `filepath|content`
  - **Does**: Creates a new file (creates parents if needed)
  - **Impl**: `FileSystemManager.file_creator()`

- **`create_directory`**
  - **Does**: Creates a directory (creates parents if needed)
  - **Impl**: `FileSystemManager.dir_creator()`

- **`write_file` / `append_file` / `delete_file`**
  - **Does**: Overwrite, append, or delete files
  - **Impl**: `FileSystemManager.writer_func()` (only some operations are fully implemented)

## Smart editing (line-based)

These use `SmartFileEditor` (`tools/smart_editor.py`) and are safer than “rewrite the whole file”.

- **`read_file_with_lines`**: reads content with line numbers
- **`insert_lines`**: insert content at a 1-based line number
- **`replace_lines`**: replace a line range
- **`delete_lines`**: delete a line range
- **`find_and_replace`**: replace text (plain or regex)
- **`analyze_file_structure`**: crude structural parsing for Python/JS/TS

## Workspace awareness / RAG

Implemented by `WorkspaceRAG` (`tools/rag_system.py`).

- **`search_workspace`**
  - **Does**: semantic search if ChromaDB works; otherwise substring fallback

- **`get_file_context`**
  - **Does**: returns functions/classes discovered + file “structure” list

- **`workspace_overview`**
  - **Does**: counts chunks, languages, and shows recent/large files

- **`get_related_code`**
  - **Does**: looks for names from a file in other files using search

## Terminal tools (safe mode)

Implemented by `TerminalManager` (`tools/terminal_manager.py`).

- **`execute_command`**: runs a command if it passes safety checks
- **`check_command_safety`**: explains whether a command is allowed/blocked
- **`suggest_commands`**: suggests a few commands based on intent
- **`get_system_info`**: returns OS/shell/env metadata (be careful: includes env vars)

## “Tool input format” is critical

The agent prompt tells the model:

- **No newlines** inside `Action Input`
- Prefer simple, raw inputs like `notes.txt|hello`

If the model violates this, tool calls can fail parsing or produce wrong paths.



