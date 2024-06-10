# # Define model paths
# model_paths = {
#     'llama-2-7b': 'meta-llama/Llama-2-7b-chat-hf',
#     'llama-2-13b': 'meta-llama/Llama-2-13b-chat-hf',
#     'vicuna-7b': "lmsys/vicuna-7b-v1.5",
#     'mixtral-7b': "mistralai/Mistral-7B-v0.1",
#     'chatglm3-6b': "THUDM/chatglm3-6b",
#     'gpt-4': 'gpt-4', 
#     'gpt-3.5-turbo': 'gpt-3.5-turbo'  
# }



nohup python run_multiverse.py --target_model gpt-3.5-turbo --start_num 250 --end_num 520 > run_multiverse_gpt-3.5-turbo.log 2>&1 &

nohup python run_multiverse.py --target_model gpt-4 --start_num 0 --end_num 520 > run_multiverse_gpt-4.log 2>&1 &

nohup python run_multiverse.py --target_model vicuna-7b > run_multiverse_vicuna-7b.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 nohup python run_multiverse.py --target_model llama-2-7b --start_num 251 --end_num 520 > run_multiverse_llama-2-7b.log 2>&1 &




CUDA_VISIBLE_DEVICES=2,3 nohup python run_multiverse.py --target_model llama-2-13b --start_num 0 --end_num 200 > run_multiverse_llama-2-13b.log 2>&1 &


CUDA_VISIBLE_DEVICES=2,3 nohup python run_multiverse.py --target_model chatglm2-6b --start_num 0 --end_num 520 > run_multiverse_chatglm2-6b.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5 nohup python run_multiverse.py --target_model chatglm3-6b --start_num 0 --end_num 520 > run_multiverse_chatglm3-6b.log 2>&1 &


CUDA_VISIBLE_DEVICES=4,5 nohup python run_multiverse.py --target_model vicuna-7b --start_num 0 --end_num 520 > run_multiverse_vicuna-7b.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5 nohup python run_multiverse.py --target_model qwen-7b --start_num 251 --end_num 520 > run_multiverse_qwen-7b.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5 nohup python run_multiverse.py --target_model gemini-1.5-flash --start_num 0 --end_num 520 > run_multiverse_gemini-1.5-flash.log 2>&1 &
