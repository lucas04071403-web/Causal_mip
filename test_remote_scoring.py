#!/usr/bin/env python3
"""
远程评分测试脚本
测试与超算服务器的连接和评分功能
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_by_llm import load_remote_llm, score_by_remote_llm_batch, build_conversation_for_gen_task, build_conversation_for_clf_task

# 超算服务器配置
REMOTE_URL = "http://210.40.56.85:21936/v1"
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"  # 服务器上加载的模型

print("=" * 60)
print("远程评分连接测试")
print("=" * 60)

# 1. 测试连接
print(f"\n[1] 连接到: {REMOTE_URL}")
try:
    client = load_remote_llm(
        base_url=REMOTE_URL,
        model_name=MODEL_NAME
    )
    print("✓ 连接成功!")
except Exception as e:
    print(f"✗ 连接失败: {e}")
    sys.exit(1)

# 2. 测试单次 API 调用
print(f"\n[2] 测试单次 API 调用...")
test_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2? Answer with a JSON object: {'answer': number}"}
]
try:
    response = client.chat(test_messages, max_tokens=50, temperature=0.1)
    print(f"✓ API 调用成功!")
    print(f"   响应: {response}")
except Exception as e:
    print(f"✗ API 调用失败: {e}")
    sys.exit(1)

# 3. 测试批量评分 - 生成任务
print(f"\n[3] 测试批量评分 - 生成任务...")
gen_preds = [
    {
        'question': 'What is the capital of France?',
        'gt': 'Paris',
        'pred': 'Paris is the capital of France.'
    },
    {
        'question': 'What is 2+2?',
        'gt': '4',
        'pred': 'The answer is 4.'
    },
]

try:
    results = score_by_remote_llm_batch(client, gen_preds, batch_size=2, task='gen')
    print(f"✓ 批量评分成功!")
    for r in results:
        print(f"   问题: {r['question'][:30]}...")
        print(f"   真实答案: {r['gt']}")
        print(f"   预测答案: {r['pred'][:30]}...")
        print(f"   评分结果: {r['correct']}")
        print()
except Exception as e:
    print(f"✗ 批量评分失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 测试批量评分 - 分类任务
print(f"\n[4] 测试批量评分 - 分类任务...")
clf_preds = [
    {
        'question': 'What is the capital of France?',
        'options': {'A': 'London', 'B': 'Paris', 'C': 'Berlin', 'D': 'Madrid'},
        'gt': 'B',
        'pred': 'B'
    },
    {
        'question': 'Which planet is closest to the Sun?',
        'options': {'A': 'Venus', 'B': 'Mercury', 'C': 'Mars', 'D': 'Earth'},
        'gt': 'B',
        'pred': 'The answer is B.'
    },
]

try:
    results = score_by_remote_llm_batch(client, clf_preds, batch_size=2, task='clf')
    print(f"✓ 分类评分成功!")
    for r in results:
        print(f"   问题: {r['question'][:30]}...")
        print(f"   真实答案: {r['gt']}")
        print(f"   预测答案: {r['pred']}")
        print(f"   评分结果: {r['correct']}")
        print()
except Exception as e:
    print(f"✗ 分类评分失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 总结
print("=" * 60)
print("测试完成!")
print("=" * 60)
print(f"\n配置成功后，你可以在 main.py 中使用:")
print(f"  --use_remote_scoring True \\")
print(f"  --remote_scoring_url \"{REMOTE_URL}\" \\")
print(f"  --score_llm {MODEL_NAME}")
