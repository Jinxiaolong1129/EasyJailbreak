import os

os.environ['HF_HOME'] = '/data3/user/jin509/hf_cache'

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0)

# llm = LLM(model="THUDM/chatglm2-6b",trust_remote_code=True)
llm = LLM(model='meta-llama/Llama-2-7b-chat-hf',trust_remote_code=True)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")