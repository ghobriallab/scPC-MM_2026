# GitHub Actions workflows

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `run-reproducibility-check.yml` | Weekly (Saturday 08:00 UTC) and manual `workflow_dispatch` | Runs `tools/check_reproducibility.py` and appends the score to `reproducibility.md` |

The scorer checks repository structure only: numeric prefixes, per-folder READMEs with a note on
where the data live, path hygiene, naming, and the absence of identifiers or credentials in code.
It reads no data and says nothing about whether the analyses are correct.

The workflow checks out tracked files only, so the gitignored data, results and figure directories
never reach the runner.
