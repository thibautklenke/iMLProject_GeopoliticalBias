# Examining Bias in Geopolitically Influenced LLMs via Prompting

The code in this repository is based on https://github.com/carolinmschuster/profiling-bias-in-llms.

We evaluated three Large Language Models (LLMs) from China, the EU and the US regarding their gender bias.
For doing so, we examined the change in their gender bias upon priming with different political stances.
In our experiments, we observed that all of our considered LLMs are susceptible to priming to varying degrees in the sense that their inherent gender bias can be influenced.

Installation:

```bash
make env
source .env/bin/activate
make install
```

## Minimal Examples

Generate Embeddings:

```bash
python -m geobias.geobias +model=qwen +primer/conservative=primer_1
```

Plot those Embeddings:

```bash
python -m geobias.plotting +model=qwen
```

## Our Experiments

We started with a handwritten seed prompt for three different political stances (economically liberal, nationalist conservative & democratic socialist and anti-fascist) and then had Gemini 3 Pro write 9 additional prompts for each stance using this template:

```
You are given this prompt:

{PROMPT}

Info: It conveys the political stance "{STANCE}".

Your task: Reformulate it into 9 variations of similar length while maintaining the topic and underlying stance.
```

Generate Embeddings for all models and all primers:

```bash
./scripts/run_all_experiments.sh
```

Plot the results by executing all cells in these jupyter notebooks:

```
notebooks/gather_data.ipynb
notebooks/plotting.ipynb
```


## References
Schuster, C. M., Roman, M. A., Ghatiwala, S., & Groh, G. (2025, March). Profiling bias in llms: Stereotype dimensions in contextual word embeddings. In Proceedings of the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies (NoDaLiDa/Baltic-HLT 2025) (pp. 639-650).

Yang, A., et al. (2025). Qwen3 Technical Report. CoRR, abs/2505.09388.

Llama Team (2024). The Llama 3 Herd of Models. CoRR, abs/2407.21783.

Project Apertus (2025). Apertus: Democratizing Open and Compliant LLMs for Global Language Environments. CoRR, abs/2509.14233.