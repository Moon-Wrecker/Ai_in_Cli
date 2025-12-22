# Quick Start (AI in CLI)

## Prereqs

- Python 3.8+
- A Google Gemini API key in `GOOGLE_API_KEY`

## Install

From the repo root:

```bash
cd Ai_in_Cli
pip install -r requirements.txt
```

Create `Ai_in_Cli/.env`:

```env
GOOGLE_API_KEY=your_key_here
```

## Run modes (pick one)

### 1) Console interactive mode (most stable)

```bash
cd Ai_in_Cli
python app.py
```

This starts a Rich-based REPL loop and lets the agent use tools against the **sandboxed** `Ai_in_Cli/workspace/`.

### 2) “One-shot” command mode (Typer CLI inside `ai_assistant.py`)

```bash
cd Ai_in_Cli
python ai_assistant.py execute "list files"
```

### 3) Textual UI (optional)

These UIs wrap the same `AIFileAssistant` core:

```bash
cd Ai_in_Cli
python ai_cli_simple.py
# or
python ai_cli_beautiful.py
# or
python beautiful_ai_cli.py
```

## First commands to try

- `List files`
- `Create file notes.txt with a short TODO list`
- `Search for "snake" in workspace`
- `Read hello.py`
- `Show system info`

## Where your files go

By default, all AI-created/edited files go into:

- `Ai_in_Cli/workspace/`

The workspace gets indexed for search into:

- `Ai_in_Cli/workspace/.rag_cache/`



