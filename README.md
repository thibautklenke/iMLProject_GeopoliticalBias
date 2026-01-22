# iMLProject_GeopoliticalBias

The code in this repository is based on https://github.com/carolinmschuster/profiling-bias-in-llms.

Installation:

```bash
uv venv
source .venv/bin/activate
uv sync
```

Minimal Example:

```bash
python -m geobias.geobias +model=qwen
```

Plotting:

```bash
python -m geobias.plotting +model=qwen
```

## References
Schuster, C. M., Roman, M. A., Ghatiwala, S., & Groh, G. (2025, March). Profiling bias in llms: Stereotype dimensions in contextual word embeddings. In Proceedings of the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies (NoDaLiDa/Baltic-HLT 2025) (pp. 639-650).