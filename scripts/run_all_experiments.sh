#!/bin/bash

MODELS=(
    "+model=apertus"
    "+model=llama_medium"
    "+model=qwen_medium"
)

ASPECTS=(
    "+primer/conservative=primer_"
    "+primer/democratic=primer_"
    "+primer/liberal=primer_"
)

for model in "${MODELS[@]}"; do

    # Default
    echo -e "\033[31mNOW RUNNING:\033[0m python -m geobias.geobias $model"
    python -m geobias.geobias $model

    for aspect in "${ASPECTS[@]}"; do

        for primer in {1..10}; do # CUDA does not like sweeping :C

            echo -e "\033[31mNOW RUNNING:\033[0m python -m geobias.geobias $model $aspect$primer"
            python -m geobias.geobias $model $aspect$primer

        done

    done

done