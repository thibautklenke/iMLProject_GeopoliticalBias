# Examining Bias in Geopolitically Influenced LLMs via Priming ☄️

The stereotype projections presented in this repository are based on https://github.com/carolinmschuster/profiling-bias-in-llms.

We evaluated three Large Language Models (LLMs) from China, Europe and the US regarding their gender bias. 
For doing so, we examined the change in their gender bias upon priming with different political stances. 
Most primers we considered significantly influenced the models’ biases, however, the new biaseswere inconsistent with the provided political stances. 
Moreover, our evaluated European and US-American models showed similar responses to priming, while the Chinese model was more susceptible to bias shift. 
We speculate this is due to geopolitical dissimilarities in the models’ origins.

---

## Quickstart 🌌

Prerequisites:
- Python 3.12
- GPU recommended for model inference (VRAM requirements vary by model)

First, clone this repo:

```bash
git clone https://github.com/thibautklenke/iMLProject_GeopoliticalBias
cd iMLProject_GeopoliticalBias
```

Then, just create a uv environment and install dependencies:

```bash
make env
source .venv/bin/activate
make install
```

Manual alternative:

```bash
uv venv --python=3.12
source .venv/bin/activate
uv sync
```

---

## Repository Structure 🌜

- `geobias/` &ndash; core package containing pipeline code (embedding retrieval, projection, t-SNE). 
- `figures/` &ndash; Result figures from our experiments.
- `configs/` &ndash; Hydra config files for models and primers. 
- `data/` &ndash; dictionaries and population files used for experiments. 
- `notebooks/` &ndash; Jupyter notebooks for gathering results and plotting experiment results. 
- `output/` &ndash; generated embeddings and projections. 
- `scripts/` &ndash; convenience scripts to run experiments or combine results. 

---

## Usage 🥢

Generate embeddings for a model:

```bash
python -m geobias.geobias +model=qwen
```

After running the baseline, you can add primers:


```bash
python -m geobias.geobias +model=qwen +primer/conservative=primer_1
```

Available model identifiers:
```
apertus llama llama_instruct llama_medium llama_medium_instruct minilingua minilingua_instruct qwen qwen_medium
```
Note: LLaMA model variants require a valid HuggingFace token to be downloaded.

Our main priming aspects are `conservative democratic liberal`. From these, you can select `primer_1, ... , primer_10`.

For full reproduction of our experiments, run:

```bash
bash scripts/run_all_experiments.sh
```

Running the entire reproduction requires roughly 30 GB of VRAM.

---

## Configuration ⚙️

Pipelines are configured with Hydra. Key config files can be found under `configs/`. Use Hydra overrides to change model, primer, or output directories, e.g.:

```bash
python -m geobias.geobias +model=llama data.populations=terms.json
```

---

## Artifacts 🪐

- `output/embeddings/` &ndash; model-specific embeddings saved during runs.
- `output/projections/` &ndash; stereotype projection results per model variant.
- `figures/tsne/` &ndash; t-SNE visualizations of the embedding spaces.
- `figures/bar/` &ndash; Bar plots visualizing experiment results.

For gathering the data from completed experiments, run `gather_data.ipynb`. With `plotting_bar.ipynb`, you can then create the bar plots. 

`output/combined_projections.csv` contains the combined projection data used for our results.

---

## 📚 References
- C. M. Schuster, M.-A. Roman, S. Ghatiwala, and G. Groh. Profiling bias in llms: Stereotype dimensions
in contextual word embeddings. In R. Johansson and S. Stymne, editors, Proceedings of the Joint 25th
Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language
Technologies, NoDaLiDa/Baltic-HLT 2025, Tallinn, Estonia, March 3-4, 2025, pages 639–650.
University of Tartu Library, 2025. URL https://aclanthology.org/2025.nodalida-1.65/.

- Yang, A., et al. (2025). Qwen3 Technical Report. CoRR.
- Llama Team (2024). The Llama 3 Herd of Models. CoRR.
- Project Apertus (2025). Apertus: Democratizing Open and Compliant LLMs for Global Language Environments. CoRR.
