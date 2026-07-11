# Generalization status: toward >80% of BGG

An honest assessment of what blocks simulating most board games. Updated after
the July 2026 generalization push, which closed the four gaps the previous
version of this document named. The engine **contract** was already general;
the push fixed the **codegen layer** and the structural caps around it.

## What now generalizes (implemented and test-pinned)

- **Multi-seat / simultaneous decisions.** Contract point 3 + the session loop
  handle multi-seat `to_act` with same-step information isolation
  (`tests/test_simultaneous.py`). The push added a *shipped* simultaneous
  engine: `playtest.games.six_nimmt` (6 nimmt!-style — simultaneous face-down
  commits, batch apply, a mid-resolution `choose_row` phase machine), with a
  34-test acceptance suite and contract-harness fuzzing at every player count.
- **Exemplar routing (was gap #1, the big one).** Codegen no longer embeds
  Love Letter unconditionally with "match precisely". The digest emits
  constrained `mechanics` tags; `select_exemplar`
  (`src/playtest/ingestion/codegen.py`) routes simultaneous / multi-stage /
  reaction games to the `six_nimmt` exemplar and everything else to
  `love_letter`, with a `--exemplar` CLI override. The engine prompt now says
  the digest, not the exemplar, defines the game, lists the binding
  CONVENTIONS explicitly, and appends mechanic-conditional guidance blocks
  (with literal code sketches for the pending-window and staged-decision
  patterns). The test prompt is de-Love-Letter-ified the same way:
  component-conservation advice disappears for `open_supply` games,
  elimination advice appears only under `player_elimination`, and the
  AUTO-ADVANCE TRAP is schematic rather than Priest/Princess-specific.
- **Reaction windows (was gap #2).** The contract docstring (point 5) now
  teaches the flattening pattern — pending action in state, responders from
  `to_act`, explicit decline, resolve/cancel when the window closes — plus the
  hidden-info rule that windows are offered by rules eligibility, never by
  hand contents (being asked is itself information). The pattern is pinned by
  an inline reference engine (`tests/test_reactions.py`: sequential priority
  window, counter-nope stack, decline-only-menu info-leak property, checkpoint
  resume mid-window) and taught in both digest and codegen prompts.
- **Large action spaces (was gap #3).** The contract docstring (point 4)
  teaches staged decisions (~50-action menu guideline); the contract harness
  measures the largest menu an engine ever presents and ingestion validation
  fails generated engines above 100 with "split this into staged decisions"
  repair feedback. Hand-written engines are unaffected. Pinned by
  `tests/test_staged.py` (staged open-auction engine: same-seat consecutive
  decisions, bound amount menus, whole-bid event at final stage, checkpoint
  resume mid-stage).
- **Digest schema (was gap #4).** `GameDigest` gained `mechanics` (the tag
  vocabulary that conditions codegen), `zones` (a prose home for boards, maps,
  tracks, markets, and open supplies, with visibility and conservation notes),
  and `max_decisions` (a per-game decision budget). `components` is now
  explicitly only fixed-count pieces. Old configs still load (defaults), while
  generation requires every field.
- **Ingestion robustness.** Transient `claude -p` failures retry inside the
  LLM client for all roles (a blip no longer kills a 20-minute ingest); when
  every engine attempt fails, the digest itself is regenerated with the
  failure feedback and the engine budget restarts; generated code passes a
  ruff error gate (undefined names, syntax) before any expensive validation,
  and generation-stage failures now reach the next engine prompt as feedback;
  budgets are `Settings`/CLI-configurable; `meta.json` records the exemplar,
  mechanics, attempt counts per stage, a prompt fingerprint (covering the live
  contract docstring, both exemplar sources, all guidance blocks and feedback
  templates, and the digest schema), and the package version.
- **Structural caps.** Seat colors cycle a 12-color palette (any player
  count); the session step budget is per-game (`max_decisions` from meta.json,
  floored during validation at 3x the longest observed self-play so a
  low-balled digest can't crash real playtests), with `MAX_STEPS` as the
  global fallback; sessions checkpoint every turn and can resume mid-window
  and mid-stage (test-pinned).

## Validation status

Offline: 207 tests pass, ruff and mypy clean. End-to-end (real ingests):

| Game | Purpose | Status |
|---|---|---|
| Love Letter (re-ingest) | sequential-path regression | **passed** — routes to the `love_letter` exemplar (an earlier run mis-tagged `multi_stage_turns` and routed to `six_nimmt`, yet still validated first try; the tag definition was tightened and the re-run tags correctly). Validated with 2 engine attempts (attempt 1 caught cheaply at the generation gate and repaired via feedback), 1 test repair, digest attempt 1. Budget: 1000 decisions |
| Hanabi (re-ingest) | co-op regression | **passed** — `love_letter` exemplar, mechanics `hidden_hands`; 2 engine attempts (same cheap generation-gate repair pattern), 0 test repairs, digest attempt 1. Budget: 600 decisions |
| Sushi Go | first generated simultaneous-drafting engine (headline proof of exemplar routing) | awaiting rulebook |
| Coup | reaction/challenge windows end-to-end | awaiting rulebook |
| For Sale (or High Society) | staged open bidding under the menu cap | awaiting rulebook |
| Splendor | open supply + market display (zones / de-conserved tests) | awaiting rulebook |

Prepare the pending rulebooks with the README's rulebook prompt and drop them
in `rulebooks/`, then `uv run playtest ingest --rulebook rulebooks/<game>.txt
--name <game>`.

## Remaining gaps (deferred by design)

| Game family | Blocking gap |
|---|---|
| Negotiation/free-deal games (Diplomacy, Catan trades) | free-form binding deals exceed `table_talk`; needs a deal protocol |
| Real-time / dexterity (Jenga, Klask) | out of scope by design |
| Legacy/campaign state | no cross-game persistence; out of scope for now |

Smaller known items, in priority order for a future pass:

- The `max_decisions = 0` fallback path skips the validation-time floor
  (`settings.max_steps` is assumed big enough); tighten if a 0-budget digest
  ever appears in practice.
- Simultaneous reaction *windows* (everyone may nope at once) are taught but
  only the sequential-priority default is reference-tested; the first truly
  simultaneous-window ingest is a known unknown.
- Mechanic-tag one-liners may under-select in two spots (static market
  displays vs `board_or_map`; occasional conditional stages vs
  `multi_stage_turns`) — tune from real ingest transcripts, not
  preemptively.
- A persistently-failing LLM backend is fed into the repair loop as if it
  were bad generated code (burns budgets before surfacing); distinguishing
  transport from content errors would fail faster.
- Notebook-only player memory will strain very long/heavy-state games (18xx
  class); revisit if such a game is ever targeted.

## Tuning feedback loop

Bulk runs log `num_legal_actions` per decision and `player_confusion` events;
the harness records the max menu seen. Use those to tune the ~50 menu
guideline / 100 hard cap and the mechanic-tag wording once several real games
are ingested.
