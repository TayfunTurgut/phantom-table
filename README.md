# playtest

An AI-powered board game playtesting tool. It takes a game's rulebook, processes it into
a game configuration, then runs autonomous playtests with LLM agents — one Game Master
(GM) agent that enforces the rules and N player agents that play the game.

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
uv run playtest ingest --rulebook rules.txt --name love_letter  # build a game config
uv run playtest show-config --game love_letter                  # inspect a built config
uv run playtest play --game love_letter --players 2             # run one playtest
uv run playtest bulk --game love_letter --num-games 10          # run many, aggregate stats
uv run playtest analyze --log-dir results                       # analytics from saved logs
uv run playtest review --log-file results/game_0.json           # review a single game log
uv run playtest smoke-test                                      # verify OpenAI connectivity
```

> Prefer an activated shell? `source .venv/bin/activate` (Windows:
> `.venv\Scripts\activate`) lets you drop the `uv run` prefix and call `playtest`
> directly.

## Ingestion

`ingest` turns a rulebook into a game config in `game_configs/<name>/`: an embedded
(ChromaDB) copy of the rules plus a generated state schema, initial-state template, player
action tools, GM prompt, and player prompt. Inspect the result with `show-config`.

Rulebooks are inputs you supply at ingest time — point `--rulebook` at any file on your
machine. Neither the rulebooks nor the generated `game_configs/` output are tracked in git
(configs are regenerated on demand), and re-running `ingest` with the same `--name`
overwrites that game's config.

## Playtesting

`play` runs one autonomous session of an ingested game. A [LangGraph](https://langchain-ai.github.io/langgraph/)
graph alternates between a Game Master (GM) agent — which deals, enforces the rules,
resolves actions, and decides round/game endings — and the player agents, which read the
public game state, optionally query the rulebook, and propose actions through the generated
tools. The GM rejects illegal moves and asks the player to try again. Live progress is
printed to the terminal and a structured summary is shown at the end.

```bash
uv run playtest play --game love_letter --players 2 \
  --seed 42 \                       # reproducible deal/shuffle
  --log-file results/game_42.json \ # persist the full event log
  --verbose \                       # also show private reasoning + GM notes
  --archetypes aggressive,cautious  # one archetype per player
```

Built-in player archetypes are prompt overlays that shape behavior: `default`,
`aggressive`, `cautious`, `analytical`, `newbie`, and `bluffer`. Pass one per player via
`--archetypes` (comma-separated); omit it for all-`default`.

## Bulk runs and analytics

`bulk` runs many playtests back-to-back, saves each game log to `--output-dir`, and prints
an aggregate report (win rates, round/turn counts, action frequencies, rule-query and
rejection stats). Game `i` uses seed `--seed-start + i` for reproducibility.

```bash
uv run playtest bulk --game love_letter --num-games 20 --output-dir results \
  --archetypes aggressive,cautious --seed-start 0
```

Recompute the same report from previously saved logs with `analyze`, or inspect a single
game with `review` (`--full` prints every event):

```bash
uv run playtest analyze --log-dir results
uv run playtest review --log-file results/game_0.json --full
```

## Observability (optional)

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in your `.env` to trace every LLM call
and graph step to [LangSmith](https://smith.langchain.com/) under the `LANGSMITH_PROJECT`
name. Tracing is off by default and requires no LangSmith account to run playtests.

## Smoke test

Verify OpenAI API connectivity (makes a real chat completion call and an embedding call):

```bash
uv run playtest smoke-test
```

## Configuration

All settings are read from `.env` (see `.env.example`). Beyond `OPENAI_API_KEY`, the
useful knobs are the models — `GM_MODEL` (default `gpt-4o`), `PLAYER_MODEL` (default
`gpt-4o-mini`), and `EMBEDDING_MODEL` (default `text-embedding-3-small`) — plus
`GAME_CONFIGS_DIR` (where ingested configs live) and `LOG_LEVEL`.

## Development

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy src         # type-check
uv run pytest           # run tests (integration tests deselected by default)
uv run pytest -m integration  # run tests that make real, paid OpenAI calls
```

## License

Released under the [MIT License](LICENSE).
