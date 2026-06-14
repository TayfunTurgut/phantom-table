# playtest

An AI-powered board game playtesting tool. It takes a game's rulebook, **generates a
deterministic Python game engine** for it, then runs autonomous playtests with LLM
player agents.

The architecture puts all mechanical work in code and all judgment in agents:

- **Ingestion** (one-time, per game): a strong model reads the rulebook and produces a
  human-reviewable *digest* (components, setup, decision flow, every action's effects,
  end conditions, scoring, ambiguity rulings), then generates a game engine module
  implementing a small fixed contract (`playtest.engine.GameEngine`) plus a pytest
  suite for it. The engine is validated by running its generated tests and a generic
  contract harness (hundreds of random self-play games checking termination,
  determinism, non-mutation, and hidden-information integrity); failures are fed back
  for automatic repair.
- **Runtime** (per playtest): the engine deals, enumerates every legal action, applies
  effects, enforces hidden information, and decides winners. Player agents only
  *choose* among the enumerated legal actions — one structured LLM call per decision.
  Illegal moves and corrupted state are impossible by construction.

The engine contract is designed for sequential, simultaneous-turn, co-op, and PvE games
(`to_act()` returns every seat that must decide right now); sequential games are what's
exercised today. A hand-written reference engine ships in `playtest.games` and serves
as the codegen exemplar and a permanent test fixture.

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
uv run playtest --help                                              # show all subcommands
uv run playtest ingest --rulebook rulebooks/my_game.txt --name my_game  # generate + validate an engine
uv run playtest show-config --game my_game                          # inspect a generated game
uv run playtest play --game my_game --players 2                     # run one playtest
uv run playtest play --game playtest.games.<module> --players 2     # run a hand-written reference engine
uv run playtest bulk --game my_game --num-games 10                  # run many, aggregate stats
uv run playtest analyze --log-dir results                           # analytics from saved logs
uv run playtest review --log-file results/game_001.json             # review a single game log
uv run playtest smoke-test                                          # verify LLM backend connectivity
```

> Prefer an activated shell? `source .venv/bin/activate` (Windows:
> `.venv\Scripts\activate`) lets you drop the `uv run` prefix and call `playtest`
> directly.

## Preparing a rulebook

Ingestion takes a single plain-text rulebook. Designers usually start from a PDF,
card/component images, and setup photos — convert those first by giving any capable
LLM (ChatGPT, Claude, Gemini, ...) your materials plus the prompt below, then save
its output as `your_game.txt` and point `ingest --rulebook` at it.

The quality of the generated engine is capped by the quality of this file: exact
counts, complete effect text, and explicit tiebreakers matter; page references and
"see image" pointers hurt.

```text
I'm preparing a board game rulebook for automated processing. Using ALL the
materials I've attached (rulebook pages, card images, board/setup photos,
reference sheets), produce ONE complete plain-text rulebook in markdown.

Requirements:

1. Use clear section headers, in this order where applicable: Overview,
   Components, Setup, How a Turn Works, Actions and Card Effects, End of
   Round / End of Game, Scoring and Tiebreakers, Hidden Information,
   Clarifications.
2. Components: list EVERY physical piece with its EXACT count (e.g. "Guard x5,
   Priest x2"). Pull counts from card images or reference sheets if the text
   omits them. Include card values/ranks where they exist.
3. Setup: describe the full setup procedure in words, separately for each
   supported player count if it differs. Convert anything shown only in a
   photo or diagram into explicit text (what goes where, face up or face
   down, who starts).
4. Actions and effects: transcribe every card/action effect COMPLETELY,
   including numbers, conditions, and timing. Cover the edge cases the
   components imply even if the rulebook is terse: what happens when there is
   no valid target, when a deck or supply runs out, when effects conflict,
   and how every tie is broken.
5. Hidden information: state explicitly what each player can and cannot see.
6. Self-contained text only: no page references, no "see image", no tables
   that depend on layout — restate everything in plain sentences or simple
   lists. Light flavor is fine; rules completeness comes first.
7. Do NOT invent rules. If something is genuinely unclear or missing from my
   materials, add it under "Clarifications" as "[UNCLEAR: ...]" with the
   options you considered, and I will resolve it.

Output only the finished rulebook text.
```

Review the result (especially any `[UNCLEAR: ...]` entries — resolve them by
editing the text) before ingesting. The ingestion digest will also surface
ambiguities it finds, with its chosen rulings, in `digest.md`.

## Ingestion

`ingest` turns a rulebook into a generated game in `game_configs/<name>/`:

| Artifact | Purpose |
| --- | --- |
| `digest.md` / `digest.json` | The structured understanding of the rulebook the engine was generated from — review this to sanity-check rules and ambiguity rulings |
| `engine.py` | The generated game engine (single `Game` class implementing the engine contract) |
| `test_engine.py` | The generated pytest suite anchored to the digest's rules |
| `player_briefing.txt` | Rules summary injected into player agents' prompts |
| `chromadb/`, `rulebook.txt` | Embedded rulebook for the players' `query_rulebook` tool |
| `meta.json` | Models used, validation attempt count, timestamp |

Validation runs in subprocesses: the generated tests must pass and the contract
harness must complete random self-play across all supported player counts with
identical replays per seed. On failure, the full diagnostic is fed back to the
codegen model and the engine is regenerated (up to 4 attempts).

Rulebooks are inputs you supply at ingest time — point `--rulebook` at any text file
(`rulebooks/` holds the ones used during development). The generated `game_configs/`
output is not tracked in git; re-running `ingest` with the same `--name` regenerates
that game from scratch.

## Playtesting

`play` runs one autonomous session. Each decision step, the engine reports who must
act, that player agent receives its private observation, the events since its last
decision, and the numbered list of legal actions — and picks one (plus private
reasoning and optional public table talk). The engine applies the action, auto-advances
through forced steps (draws, redeals, scoring), and emits factual event lines with
per-seat visibility. Live progress is printed to the terminal and a structured summary
is shown at the end.

```bash
uv run playtest play --game my_game --players 2 \
  --seed 42 \                       # reproducible deal/shuffle
  --log-file results/game_42.json \ # persist the full event log
  --verbose \                       # also show private reasoning
  --archetypes aggressive,cautious  # one archetype per player
```

Built-in player archetypes are prompt overlays that shape behavior: `default`,
`aggressive`, `cautious`, `analytical`, `newbie`, and `bluffer`. Pass one per player via
`--archetypes` (comma-separated); omit it for all-`default`.

## Bulk runs and analytics

`bulk` runs many playtests back-to-back, saves each game log to `--output-dir`, and prints
an aggregate report (win rates, decision counts, action frequencies, rule-query and
confusion stats). Game `i` uses seed `--seed-start + i` for reproducibility.

```bash
uv run playtest bulk --game my_game --num-games 20 --output-dir results \
  --archetypes aggressive,cautious --seed-start 0
```

Recompute the same report from previously saved logs with `analyze`, or inspect a single
game with `review` (`--full` prints every event):

```bash
uv run playtest analyze --log-dir results
uv run playtest review --log-file results/game_001.json --full
```

## Observability (optional)

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in your `.env` to trace every LLM call
to [LangSmith](https://smith.langchain.com/) under the `LANGSMITH_PROJECT`
name. Tracing is off by default and requires no LangSmith account to run playtests.

## Smoke test

Verify the active LLM and embedding backends (makes one real completion and one
embedding):

```bash
uv run playtest smoke-test
```

## Configuration

All settings are read from `.env` (see `.env.example`). The headline knob is the
LLM backend — all completions flow through one client interface
(`playtest.llm.LLMClient`) with selectable adapters:

- `LLM_BACKEND=openai` — the OpenAI API. Requires `OPENAI_API_KEY`. Models per
  role: `PLAYER_MODEL` (default `gpt-5-mini`),
  `DIGEST_MODEL`/`CODEGEN_MODEL` (default `gpt-5`).
- `LLM_BACKEND=claude_cli` — headless `claude -p` on your Claude subscription
  (works with enterprise OAuth; set `CLAUDE_CODE_OAUTH_TOKEN` from
  `claude setup-token`). Models per role: `CLAUDE_PLAYER_MODEL` /
  `CLAUDE_DIGEST_MODEL` / `CLAUDE_CODEGEN_MODEL`
  (default `sonnet`). Adds ~1-2s spawn overhead per call; players' rulebook
  tool is disabled (no tool round-trips through the CLI). LangSmith tracing
  covers the openai backend only.

`EMBEDDING_BACKEND` picks the rulebook index embeddings: `openai`
(`EMBEDDING_MODEL`, default `text-embedding-3-small`) or `local` (ChromaDB's
built-in ONNX model — free and offline). Indexes are backend-specific;
re-ingest after switching.

Other knobs: `GAME_CONFIGS_DIR` (where generated games live), `MAX_STEPS`
(default 1000, a crashing ceiling on decisions per session),
`PLAYER_RULEBOOK_QUERIES` (whether players may consult the rulebook before
choosing), and `LLM_TIMEOUT_SECONDS` (default 900).

## Development

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy src         # type-check
uv run pytest           # run tests (integration tests deselected by default)
```

## License

Released under the [MIT License](LICENSE).
