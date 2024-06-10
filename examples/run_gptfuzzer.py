import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.attacker import GPTFuzzer
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel
import argparse
sys.path.append(os.getcwd())



def main(api_key, target_model_name, start_num, end_num):
    generation_config = {'max_new_tokens': 500}
    
    dataset_name = 'TDC'
    
    dataset = JailbreakDataset(dataset='data/TDC_data.csv', local_file_type='csv')

    save_parent_dir = f'results/{dataset_name}/GPTFuzzer'
    os.makedirs(save_parent_dir, exist_ok=True)
    log_path = os.path.join(save_parent_dir, f'TDC_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_start{start_num}_end{end_num}.log')
    print(f"log_path: {log_path}")
    save_dir = os.path.join(save_parent_dir, f'TDC_target.{target_model_name}_eval.gpt-3.5-turbo_start{start_num}_end{end_num}.jsonl')
    print(f"Running jailbreak attack on target model: {save_dir}")
    
    
    model_paths = {
        'llama-2-7b': 'meta-llama/Llama-2-7b-chat-hf',
        'llama-2-13b': 'meta-llama/Llama-2-13b-chat-hf',
        'vicuna-7b': "lmsys/vicuna-7b-v1.5",
        'mixtral-7b': "mistralai/Mistral-7B-v0.1",
        'chatglm3-6b': "THUDM/chatglm3-6b",
        'chatglm2-6b': "THUDM/chatglm2-6b",
        "qwen-7b": "Qwen/Qwen-7B-Chat",
        "intern-7b": "internlm/internlm-7b",
        'gpt-4': 'gpt-4', 
        'gpt-3.5-turbo': 'gpt-3.5-turbo',  
    }
    
    if target_model_name not in model_paths:
        raise ValueError(f"Invalid target model name: {target_model_name}")

    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    # Load the target model
    if target_model_name.startswith('llama') or target_model_name.startswith('vicuna') or target_model_name.startswith('mixtral'):
        model_path = model_paths[target_model_name]
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
    
    elif target_model_name == 'qwen-7b':
        model_path = model_paths[target_model_name]
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
    
    elif target_model_name == 'intern-7b':
        model_path = model_paths[target_model_name]
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
        
    elif target_model_name == 'chatglm3-6b' or target_model_name == 'chatglm2-6b':
        model_path = model_paths[target_model_name]
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
    
    elif target_model_name == 'gpt-4':
        target_model = OpenaiModel(model_name='gpt-4', api_keys=api_key)
    
    elif target_model_name == 'gpt-3.5-turbo':
        target_model = OpenaiModel(model_name='gpt-3.5-turbo', api_keys=api_key)
        
        
    eval_model = OpenaiModel(model_name='gpt-3.5-turbo', api_keys=api_key)
    
    model_path = 'hubert233/GPTFuzz'
    judge_model = RobertaForSequenceClassification.from_pretrained(model_path)
    judge_tokenizer = RobertaTokenizer.from_pretrained(model_path)
    judge_eval_model = HuggingfaceModel(model=judge_model, tokenizer=judge_tokenizer, model_name='zero_shot')

    attacker = GPTFuzzer(attack_model=eval_model,
                        target_model=target_model,
                        eval_model=judge_eval_model,
                        jailbreak_datasets=dataset,
                        max_iteration=100,
                        max_query=  1000,
                        max_jailbreak= 1000,
                        max_reject= 10000)
    
    attacker.attack()
    attacker.log()
    attacker.attack_results.save_to_jsonl(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run jailbreak attacks on specified target model.")
    parser.add_argument('--target_model', type=str, default='gpt-3.5-turbo',
                        help="Specify the target model for the attack.")
    parser.add_argument('--start_num', type=int, default=0)
    parser.add_argument('--end_num', type=int, default=1)
    
    args = parser.parse_args()
    
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"API key: {api_key}")
    main(api_key, args.target_model, args.start_num, args.end_num)
    
