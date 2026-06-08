# playtest

An AI-powered board game playtesting tool. It takes a game's rulebook, processes it into
a game configuration, then runs autonomous playtests with LLM agents — one Game Master
(GM) agent that enforces the rules and N player agents that play the game. This repository
is the project skeleton: structure, dependencies, configuration, and a verified connection
to the OpenAI API.

## Setup

```bash
git clone https://github.com/TayfunTurgut/phantom-table.git
cd phantom-table

# Create your environment file and add your OpenAI API key
cp .env.example .env
# edit .env and set OPENAI_API_KEY

# Install the package (with dev tools) in editable mode
pip install -e ".[dev]"
```

## Smoke test

Verify OpenAI API connectivity (makes a real chat completion call and an embedding call):

```bash
playtest smoke-test
```

## CLI

```bash
playtest --help                                              # show all subcommands
playtest ingest --rulebook rules.txt --name love_letter      # (not yet implemented)
playtest play --game love_letter --players 2                 # (not yet implemented)
playtest smoke-test                                          # verify OpenAI connectivity
```

## Development

```bash
ruff check .          # lint
ruff format .         # format
mypy src              # type-check
pytest                # run tests
```

## License

Released under the [MIT License](LICENSE).
