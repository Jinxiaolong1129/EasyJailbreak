import os, sys

# gpu_list = [2,3]
# gpu_str = ','.join([str(gpu) for gpu in gpu_list])
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

from transformers import AutoModelForCausalLM, AutoTokenizer
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.attacker.Multilingual_Deng_2023 import *
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel
sys.path.append(os.getcwd())

chat_name = 'gpt-3.5-turbo'
api_key = os.getenv('OPENAI_API_KEY')
print(api_key)
# GPT_model = OpenaiModel(model_name= chat_name,api_keys=api_key)
