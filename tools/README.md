# tools/

| Script | Output |
|--------|--------|
| `check_reproducibility.py` | A 0-100 repository structure score, run weekly by CI |

## check_reproducibility.py

Scores the repository on structure and hygiene, not on scientific correctness. Standard library
only.

| Category | Weight | Checks |
|----------|--------|--------|
| Step ordering | 20 | Scripts and folders carry numeric prefixes |
| Documentation | 25 | Header blocks in scripts, a README in each folder |
| Path hygiene | 20 | No hardcoded local paths |
| Data handling | 15 | `.gitignore` covers data files, READMEs say where the data live |
| Naming | 10 | No spaces or special characters in file names |
| PHI and credentials | 10 | No identifiers, passwords or API keys in code |

```bash
python3 tools/check_reproducibility.py            # current directory
python3 tools/check_reproducibility.py /path      # a specific directory
python3 tools/check_reproducibility.py --json     # write reproducibility_score.json
python3 tools/check_reproducibility.py --min-score 80   # exit 1 below the threshold
```

A score of 80 or above is good; below 60 means something structural is wrong.
