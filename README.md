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