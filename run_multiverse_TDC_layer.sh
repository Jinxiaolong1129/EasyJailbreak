#!/bin/bash


nohup python run_multiverse_TDC_layer.py --target_model gpt-3.5-turbo --start_num 0 --end_num 50 > run_multiverse_TDC_layer_gpt-3.5-turbo.log 2>&1 &

# nohup python run_multiverse_TDC_layer.py --target_model gpt-4 --start_num 0 --end_num 50 > run_multiverse_TDC_layer_gpt-4.log 2>&1 &

nohup python run_multiverse_TDC_layer.py --target_model gemini-1.5-flash --start_num 0 --end_num 50 > run_multiverse_TDC_layer_gemini-1.5-flash.log 2>&1 &


CUDA_VISIBLE_DEVICES=7 nohup python run_multiverse_TDC_layer.py --target_model llama-2-7b --start_num 0 --end_num 50 > run_multiverse_TDC_layer_llama-2-7b.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python run_multiverse_TDC_layer.py --target_model chatglm2-6b --start_num 0 --end_num 50 > run_multiverse_TDC_layer_chatglm2-6b.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python run_multiverse_TDC_layer.py --target_model chatglm3-6b --start_num 0 --end_num 50 > run_multiverse_TDC_layer_chatglm3-6b.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python run_multiverse_TDC_layer.py --target_model vicuna-7b --start_num 0 --end_num 50 > run_multiverse_TDC_layer_vicuna-7b.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python run_multiverse_TDC_layer.py --target_model qwen-7b --start_num 0 --end_num 50 > run_multiverse_qwen-7b.log 2>&1 &

