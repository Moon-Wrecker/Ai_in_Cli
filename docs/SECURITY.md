# Security Model (Sandbox + Safe Mode)

This application is designed to be useful while limiting blast radius.

## 1) Workspace sandboxing (filesystem safety)

All file operations are intended to be restricted to a single directory:

- default: `Ai_in_Cli/workspace/` (config: `WORKSPACE_DIR`)

Two independent subsystems enforce this:

### A) `FileSystemManager` sandbox

`FileSystemManager` is constructed with `base_dir=<workspace>`.

Its internal path resolution (`_resolve_path`) ensures:

- relative paths are joined under `base_dir`
- absolute paths must still fall under `base_dir`
- attempts to escape the base directory throw an error

### B) `SmartFileEditor` sandbox

`SmartFileEditor` performs similar enforcement in `_resolve_path`.

So even if the LLM tries to cheat with `/etc/...` paths or `../..`, it should fail.

## 2) Terminal safe mode (command execution safety)

Shell command execution is handled by `TerminalManager(safe_mode=True)`.

### Allowed command prefixes

`TerminalManager` maintains an allowlist of prefixes like:

- file listing / navigation: `ls`, `pwd`, `tree`, `cat`, `head`, `tail`
- system info: `uname`, `whoami`, `date`, `ps`, `top`
- text filtering: `grep`, `awk`, `sed`

If your command’s base program is not in the allowlist, it’s blocked.

### Dangerous pattern blocks

Even allowed commands are blocked if they contain suspicious patterns like:

- destructive ops: `rm -rf`, `format`, `shutdown`, `reboot`
- pipe-to-shell: `curl ... | sh`, `wget ... | bash`
- shell operators: `&&`, `||`, `;`, command substitution, backticks

### Implication for users

- You can run **simple, single** commands.
- You generally cannot run chained commands like `cd x && ls`.
- You should prefer asking the AI to use its **file tools** rather than shell tools when possible.

## 3) Data / privacy considerations

- The terminal `get_system_info` tool returns **environment variables**. If your environment contains secrets, they can appear in tool output.
- The RAG index stores chunk content and metadata in `workspace/.rag_cache/`. Treat that directory as sensitive if your workspace contains sensitive code.



