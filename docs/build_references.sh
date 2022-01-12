#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

lazydocs \
    --output-path="${SCRIPT_DIR}/references/" \
    --overview-file="README.md" \
    --src-base-url="https://github.com/tjyuyao/ice-learn/blob/main/" \
    ice