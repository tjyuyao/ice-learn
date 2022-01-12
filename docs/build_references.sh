#!/usr/bin/env bash

# Note: run this script in REPOSITORY_ROOT.
lazydocs \
    --output-path="./docs/references/" \
    --overview-file="README.md" \
    --src-base-url="https://github.com/tjyuyao/ice-learn/blob/main/" \
    ice
