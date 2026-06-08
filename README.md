# playtest

An AI-powered board game playtesting tool. It takes a game's rulebook, processes it into
a game configuration, then runs autonomous playtests with LLM agents — one Game Master
(GM) agent that enforces the rules and N player agents that play the game. This repository
is the project skeleton: structure, dependencies, configuration, and a verified connection
to the OpenAI API.

This project uses [uv](https://docs.astral.sh/uv/) as its package manager and task
runner. Install it first if you haven't:
[installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## Setup

```bash
git clone https://github.com/TayfunTurgut/phantom-table.git
cd phantom-table

# Create your environment file and add your OpenAI API key
cp .env.example .env
# edit .env and set OPENAI_API_KEY

# Create the virtual environment and install all dependencies (incl. dev tools)
uv sync --extra dev
```

`uv sync` creates a `.venv/` and installs the locked dependencies (from `uv.lock`)
plus the `playtest` package itself in editable mode.

## Running commands

Prefix commands with `uv run` — uv resolves the project's `.venv` automatically, so
there's no need to activate it:

```bash
uv run playtest --help                                          # show all subcommands
uv run playtest ingest --rulebook rules.txt --name love_letter  # (not yet implemented)
uv run playtest play --game love_letter --players 2             # (not yet implemented)
uv run playtest smoke-test                                      # verify OpenAI connectivity
```

> Prefer an activated shell? `source .venv/bin/activate` (Windows:
> `.venv\Scripts\activate`) lets you drop the `uv run` prefix and call `playtest`
> directly.

## Smoke test

Verify OpenAI API connectivity (makes a real chat completion call and an embedding call):

```bash
uv run playtest smoke-test
```

## Development

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy src         # type-check
uv run pytest           # run tests
```

## License

Released under the [MIT License](LICENSE).
