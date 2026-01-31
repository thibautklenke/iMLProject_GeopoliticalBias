#!/bin/bash

MODELS=(
    "+model=apertus"
    "+model=llama_medium"
    "+model=qwen_medium"
)

PRIMER=(
    "" # Default
    "+primer/conservative=glob(primer_*)"
    "+primer/democratic=glob(primer_*)"
    "+primer/liberal=glob(primer_*)"
)

for model in "${MODELS[@]}"; do

    for primer in "${PRIMER[@]}"; do

        python -m geobias.geobias $model $primer -m

    done

done