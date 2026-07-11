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
(`to_act()` returns every seat that must decide right now), and teaches two idioms that
unlock harder families: reaction/"may respond" windows flattened into explicit decision
points, and combinatorial choices staged into consecutive smaller menus. Two
hand-written reference engines ship in `playtest.games` — `love_letter` (sequential,
hidden hands) and `six_nimmt` (6 nimmt!/Take 5: simultaneous commits, mid-resolution
phase machine) — serving as codegen exemplars and permanent test fixtures; ingestion picks the exemplar
that matches the digest's mechanics.

This project uses [uv](https://docs.astral.sh/uv/) as its package manager and task
runner. Install it first if you haven't:
[installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## Setup

```bash
git clone https://github.com/TayfunTurgut/phantom-table.git
cd phantom-table

# Create your environment file. Auth is your Claude Code login — no API key.
cp .env.example .env
# Optional: set CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`) in .env,
# or just log in once with the Claude Code CLI.

# Create the virtual environment and install all dependencies (incl. dev tools)
uv sync --extra dev
```

`uv sync` creates a `.venv/` and installs the locked dependencies (from `uv.lock`)
plus the `playtest` package itself in editable mode.

Completions run through the **Claude Code CLI**, so it must be installed and
authenticated on this machine:

```bash
npm install -g @anthropic-ai/claude-code
claude --version
# Authenticate once: interactive `claude` login, or `claude setup-token`
# (then put the token in .env as CLAUDE_CODE_OAUTH_TOKEN).
```

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
6. Core gameplay only: capture the standard, base game's single core
   gameplay loop. Do NOT include game variants, optional/house rules,
   expansions, advanced or alternative game modes, solo/team variants, or
   difficulty tiers — even if the materials describe them. If the base game
   itself has no single default (e.g. the rulebook only presents variants),
   note that under "Clarifications" and pick the most standard one.
7. Self-contained text only: no page references, no "see image", no tables
   that depend on layout — restate everything in plain sentences or simple
   lists. Light flavor is fine; rules completeness comes first.
8. Do NOT invent rules. If something is genuinely unclear or missing from my
   materials, add it under "Clarifications" as "[UNCLEAR: ...]" with the
   options you considered, and I will resolve it.

Output only the finished rulebook text.
```

Review the result (especially any `[UNCLEAR: ...]` entries — resolve them by
editing the text) before ingesting. The ingestion digest will also surface
ambiguities it finds, with its chosen rulings, in `digest.md`.

If you specifically want to playtest a particular variant or advanced mode, tell
the AI to generate the rulebook for that mode instead of the base game. Keep it
to one rulebook per game, though: each rulebook should describe a single,
coherent set of rules. Mixing multiple modes into one file confuses both the
engine generation and the player agents — ingest each mode as its own game
(e.g. `--name my_game_advanced`) if you want to compare them.

## Ingestion

`ingest` turns a rulebook into a generated game in `game_configs/<name>/`:

| Artifact | Purpose |
| --- | --- |
| `digest.md` / `digest.json` | The structured understanding of the rulebook the engine was generated from — review this to sanity-check rules and ambiguity rulings |
| `engine.py` | The generated game engine (single `Game` class implementing the engine contract) |
| `test_engine.py` | The generated pytest suite anchored to the digest's rules |
| `player_briefing.txt` | Rules summary injected into player agents' prompts |
| `rulebook.txt` | The raw rulebook, injected in full into player prompts so agents can consult the exact rules |
| `meta.json` | Models used, exemplar and prompt fingerprint, attempt counts, per-game decision budget, timestamp |

The digest tags the game's structural `mechanics` (simultaneous decisions,
reaction windows, multi-stage turns, open supplies, boards, ...). Those tags
pick which hand-written reference engine is embedded in the codegen prompt as
the exemplar — `love_letter` for sequential hidden-hand games, `six_nimmt` (a
simultaneous-commit engine with a mid-resolution phase machine) for
simultaneous/staged/reaction games — and switch mechanic-specific guidance
blocks in the engine and test prompts. Override the routing with
`ingest --exemplar <name>` if a digest mis-tags.

Validation runs in subprocesses: generated code is first ruff-checked for
outright errors, then the contract harness must complete random self-play
across all supported player counts with identical replays per seed (and no
legal-action menu larger than 100 — oversized menus are sent back with "split
this into staged decisions" feedback), then the generated tests must pass. On
failure, the full diagnostic is fed back to the codegen model and the engine
is regenerated (default 4 attempts, `--max-attempts`); if every engine attempt
fails, the digest itself is re-derived with the failure feedback and the
engine budget starts over (default 2 digest attempts). Transient `claude -p`
failures are retried inside the LLM client, for every role.

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

## Smoke test

Verify Claude Code is reachable (makes one real `claude -p` completion):

```bash
uv run playtest smoke-test
```

## Configuration

All settings are read from `.env` (see `.env.example`). Every completion flows
through one client interface (`playtest.llm.LLMClient`), backed by **headless
`claude -p`** — one stateless, isolated subprocess per call, structured output
enforced via `--json-schema`. There is no per-turn memory: each decision is a
fresh, self-contained call (a player's only carried memory is the notebook it
rewrites for itself each turn).

Auth is your Claude Code login: interactive login, or set
`CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`) in `.env`. No API key is
used and nothing is billed per token — completions run against your Claude
subscription.

Models are set per role: `CLAUDE_PLAYER_MODEL` (runtime decisions),
`CLAUDE_DIGEST_MODEL` / `CLAUDE_CODEGEN_MODEL` (one-time game generation), all
defaulting to `sonnet`. Bump the generation models to `opus` for stronger
codegen if your plan allows. Each call adds ~1-2s of CLI spawn overhead.

Players consult the rules by reading the **full rulebook text**, injected
directly into their prompt (no vector search / RAG). Other knobs:
`CLAUDE_CLI_PATH` (default `claude`), `GAME_CONFIGS_DIR` (where generated games
live), `MAX_STEPS` (default 1000, the fallback crashing ceiling on decisions per
session — generated games carry their own budget in `meta.json` as
`max_decisions`, derived from the digest and floored at 3x the longest
validation self-play), `LLM_TIMEOUT_SECONDS` (default 900), transient-failure
retry (`LLM_RETRY_ATTEMPTS` / `LLM_RETRY_BACKOFF_SECONDS`), and the ingestion
budgets (`INGEST_MAX_ENGINE_ATTEMPTS`, `INGEST_MAX_TEST_REPAIRS`,
`INGEST_MAX_DIGEST_ATTEMPTS`, `INGEST_GAMES_PER_COUNT`,
`INGEST_VALIDATION_TIMEOUT_SECONDS`).

## Development

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy src         # type-check
uv run pytest           # run tests (integration tests deselected by default)
```

## License

Released under the [MIT License](LICENSE).
