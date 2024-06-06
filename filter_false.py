# 导入所需的库
import json

# 初始化索引列表
false_indices = []

# 打开文件
with open('results/advbench_target.llama-2-7b_eval.gpt-3.5-turbo.jsonl', 'r') as file:
    # 逐行读取
    for index, line in enumerate(file):
        # 将JSON字符串转换为字典
        data = json.loads(line)
        # 检查 eval_results 字段是否为 [false]
        if data.get('eval_results') == [False]:
            # 将索引添加到列表中
            false_indices.append(index)

# 输出结果
print("Indices with false eval_results:", false_indices)
