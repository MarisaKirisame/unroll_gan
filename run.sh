#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py --config unroll_0
CUDA_VISIBLE_DEVICES=1 python main.py --config no_higher_unroll_10
CUDA_VISIBLE_DEVICES=2 python main.py --config yes_higher_unroll_10