#!/bin/bash

MODELS=(
    "+model=apertus"
    "+model=llama_medium"
    "+model=qwen_medium"
)

PRIMER=(
    "+primer/conservative=primer_1"
    "+primer/democratic=primer_1"
    "+primer/liberal=primer_1"
)

for model in "${MODELS[@]}"; do

    for primer in "${PRIMER[@]}"; do

        python -m geobias.tsne $model $primer +plot=True

    done

done