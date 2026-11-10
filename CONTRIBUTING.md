# Contributing

Thanks for your interest in podtrans.

## Development setup

The project is managed with [uv](https://docs.astral.sh/uv/). From a clone:

```bash
uv lock --check
uv sync
uv run python -m unittest discover -s tests -v
uv build
```

Supported platforms are Linux and macOS, with Python 3.11 or 3.12. See the README for optional extras (`inference`, `vllm`, `mlx`) and hardware flags.

## Pull requests

- Open PRs against `main`.
- Keep changes focused. Do not include podcast audio, transcripts, model weights, caches, or secrets.
- Add or update tests when behavior changes.
- Run the commands above before you open the PR.

Issues and pull requests on GitHub are the right place for bugs, questions, and proposals.

## License

By contributing, you agree that your contributions are licensed under the MIT License in [LICENSE](LICENSE).
