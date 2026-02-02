#!/bin/bash

MODELS=(
    "+model=apertus"
    "+model=llama_medium"
    "+model=qwen_medium"
)

PRIMER=(
    "+primer/conservative=primer_1.yaml"
    "+primer/democratic=primer_1.yaml"
    "+primer/liberal=primer_1.yaml"
)

model=""
for model in "${MODELS[@]}"; do

    python -m geobias.tsne $model +plot=True

    for primer in "${PRIMER[@]}"; do

        python -m geobias.tsne $model $primer +plot=True -m

    done

done