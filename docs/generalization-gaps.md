# Generalization gaps: from Love Letter to >90% of BGG

An honest assessment of what currently blocks simulating most board games, with
the smallest blocking component named for each gap. The engine **contract** is
already general; the gaps are concentrated in the **codegen layer** and a few
structural assumptions.

## What already generalizes (verified)

- **Multi-seat / simultaneous decisions.** The contract allows `to_act()` to
  return several seats (`src/playtest/engine/__init__.py`, contract point 3),
  and the harness (`src/playtest/engine/contract.py`) and session loop
  (`src/playtest/session.py`) handle it: one decision per acting seat against
  the same pre-step state, applied as a batch. This is now a *tested* property
  (`tests/test_simultaneous.py`), including information isolation — same-step
  table talk is broadcast only after every acting seat has committed.
- **Co-op and PvE.** Contract point 8 defines co-op winner semantics; automa
  behavior is deterministic logic inside `apply` (point 3). `GameStatus`
  carries winners/scores with no card-game assumptions.
- **The runtime loop, agents, and analytics** are game-agnostic: they treat
  observations, legal actions, and events as opaque.

Caveat: no shipped or *generated* engine uses simultaneity yet. The first real
simultaneous-game ingestion (e.g., a drafting or sealed-bid game) is the true
end-to-end test.

## Gaps

### 1. Single codegen exemplar (the big one)

Blocking component: `src/playtest/ingestion/codegen.py`.

- `_reference_engine_source()` embeds the full Love Letter module and the
  engine prompt demands matching "its style, structure, and conventions
  precisely". For games unlike Love Letter (no hands, no elimination, no
  rounds, simultaneous commits) the exemplar contradicts the digest, biasing
  generated engines toward small sequential card games.
- The test-generation prompt is saturated with Love Letter idioms: "verify
  setup deals correctly", elimination advice ("use enough players that
  eliminations don't immediately end a round"), and AUTO-ADVANCE examples
  naming Priest/Princess/hands/redeals. The `validate.py` repair hint likewise
  says "the next player has already auto-drawn".
- Component-conservation test guidance assumes fixed pools; many games create
  or destroy resources from an open supply.

Likely fix when ingesting game #2: make the exemplar/test-hint text
mechanic-conditional (or maintain a small exemplar library per game family:
trick-taking, drafting, worker placement, deck builder) and soften "match
precisely" to "match the conventions listed below".

### 2. No reaction-window / interrupt mechanism

Blocking component: the engine contract (`apply` auto-advance, point 5).

`apply` resolves and auto-advances to the next decision point; there is no way
to interrupt mid-resolution. Reaction mechanics (Nope cards, instants, "may
respond" windows) must be *flattened into discrete `to_act` decision points* by
the generated engine — possible, but the codegen prompt never tells the model
this pattern, so it is unlikely to emerge reliably.

### 3. Full enumeration of legal actions

Blocking component: contract point 4 + `agents/player.py` menu prompt.

`legal_actions` must enumerate every fully-bound action, and the player agent
renders them as a numbered menu. This blows up for combinatorial action spaces:
open-ended placement (Go, many area-control games), multi-part turns (move N
units along any path), auctions with arbitrary bid amounts. Needs either action
templates with parameter sub-decisions, or staged decision points.

### 4. Digest schema leans hidden-info-card-game

Blocking component: `src/playtest/ingestion/schemas.py`.

`components` ("conserved physical pieces") and `hidden_zones` fit card/tile
games well; `decision_flow` does ask for "simultaneity, reaction windows".
Mostly prose so it bends, but games with persistent boards, maps, or open
supplies have no natural home in the schema — the digest model will improvise
inconsistently.

### 5. Smaller structural caps

- `PLAYER_COLORS` styles 6 seats (`src/playtest/ui/observer.py`); extra seats
  render white. Cosmetic.
- `max_steps=1000` per session (`config.py`); long engines that auto-advance
  poorly will hit it. The cap is configurable.
- Determinism: engines are seed-deterministic (contract points 2, 4), but LLM
  player decisions are not — only scripted-player replays reproduce exactly
  (`tests/test_session.py::test_session_is_deterministic_with_scripted_players`).

## Coverage verdict for the >90% BGG target

**Coverable now:** sequential turn-based games with hidden or perfect
information and enumerable actions — most card games, roll-and-move, set
collection, simple economic/engine builders, tableau games. Simultaneous-reveal
games are supported by the runtime but unproven through codegen.

**Blocked, by gap:**

| Game family | Blocking gap |
|---|---|
| Drafting / sealed bids / simultaneous programming (7 Wonders, RoboRally) | #1 (codegen has no simultaneous exemplar); runtime is ready |
| Reaction/interrupt-heavy games (Exploding Kittens, Magic-likes) | #2 |
| Huge/combinatorial action spaces (Go, wargames, open auctions) | #3 |
| Negotiation/free-deal games (Diplomacy, Catan trades) | free-form binding deals exceed `table_talk`; needs a deal protocol |
| Real-time / dexterity (Jenga, Klask) | out of scope by design |
| Legacy/campaign state | no cross-game persistence; out of scope for now |

The highest-leverage next step is gap #1: it is prompt text, not architecture,
and it gates every game family the runtime can already host.
