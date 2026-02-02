#!/bin/bash

MODELS=(
    "+model=apertus"
    "+model=llama_medium"
    "+model=qwen_medium"
)

ASPECTS=(
    "+primer/plain=conservative"
    "+primer/plain=democratic"
    "+primer/plain=liberal"
)

for model in "${MODELS[@]}"; do

    # Default
    python -m geobias.geobias "$model"

    for aspect in "${ASPECTS[@]}"; do

        echo -e "\033[31mNOW RUNNING:\033[0m python -m geobias.geobias \"$model\" \"$aspect\""
        python -m geobias.geobias "$model" "$aspect"


    done

done