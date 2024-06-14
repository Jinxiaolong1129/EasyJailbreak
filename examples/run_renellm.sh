#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python run_renellm.py --target_model gpt-3.5-turbo --start_num 0 --end_num 50 > run_renellm_gpt-3.5-turbo.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python run_renellm.py --target_model gemini-1.5-flash --start_num 0 --end_num 50 > run_renellm_gemini-1.5-flash.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python run_renellm.py --target_model gpt-4 --start_num 0 --end_num 50 > run_renellm_gpt-4.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python run_renellm.py --target_model vicuna-7b --start_num 0 --end_num 50 > run_renellm_vicuna-7b.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python run_renellm.py --target_model llama-2-7b --start_num 0 --end_num 50 > run_renellm_llama-2-7b.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python run_renellm.py --target_model chatglm2-6b --start_num 0 --end_num 50 > run_renellm_chatglm2-6b.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python run_renellm.py --target_model chatglm3-6b --start_num 0 --end_num 50 > run_renellm_chatglm3-6b.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 nohup python run_renellm.py --target_model qwen-7b --start_num 0 --end_num 50 > run_renellm_qwen-7b.log 2>&1 &

