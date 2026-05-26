from tqdm import tqdm
import transformers
import torch
import json
import os
from typing import Optional, Union

# ==================== 远程 API 调用支持 ====================
class RemoteLLMClient:
    """
    远程 LLM API 客户端（OpenAI 兼容格式）
    支持调用部署在超算服务器上的 Qwen2-7B-Instruct
    """
    def __init__(self, base_url: str, api_key: str = "empty", model_name: str = "Qwen2.5-VL-7B-Instruct"):
        """
        Args:
            base_url: API 服务地址，如 "http://your-supercomputer:9192/v1"
            api_key: API Key（可选，默认为 "empty"）
            model_name: 模型名称，如 "Qwen2.5-VL-7B-Instruct"
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai 库: pip install openai")

        # 创建禁用代理的 httpx 客户端
        import httpx
        # 清除代理环境变量并创建不使用代理的客户端
        no_proxy_backup = os.environ.get('no_proxy', '')
        os.environ['no_proxy'] = '*'
        http_client = httpx.Client()
        os.environ['no_proxy'] = no_proxy_backup
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        self.model_name = model_name

    def chat(self, messages: list, max_tokens: int = 50, temperature: float = 0.0) -> str:
        """发送聊天请求并返回响应文本"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def batch_chat(self, messages_list: list, max_tokens: int = 50, temperature: float = 0.1) -> list:
        """批量发送聊天请求（temperature 默认为 0.1 以兼容部分 API 服务）"""
        responses = []
        for messages in tqdm(messages_list, desc="远程 API 调用"):
            try:
                response = self.chat(messages, max_tokens, temperature)
                responses.append(response)
            except Exception as e:
                print(f"API 调用失败: {e}, 返回 None")
                responses.append(None)
        return responses


def load_remote_llm(base_url: str, api_key: str = "empty", model_name: str = "Qwen2.5-VL-7B-Instruct") -> RemoteLLMClient:
    """
    加载远程 LLM API 客户端

    Args:
        base_url: API 服务地址，如 "http://your-supercomputer:9192/v1"
        api_key: API Key（可选）
        model_name: 模型名称

    Returns:
        RemoteLLMClient 实例
    """
    return RemoteLLMClient(base_url, api_key, model_name)


# ==================== 本地模型加载 ====================
def load_llm(path: str, device: str):
    """
    Load a pre-trained LLM model from the specified path.

    Args:
        path (str): Path to the pre-trained model.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        transformers.PreTrainedModel: Loaded LLM model.
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    model.to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, use_fast=True, padding_side='left')
    return model, tokenizer


def build_conversation_for_gen_task(item: dict) -> list[dict]:
    system_prompt = """You are a helpful assistant that determines whether the prediction is correct.
Given a question with correct answer and a predicted answer, you will output a JSON object with a boolean field 'answer' indicating
whether the 'pred' answer is semantically the same (true) or different (false) as the 'truth' option.
The sentences will be provided in the fields 'input', 'question', 'truth', and 'pred'.
Please respond only with the JSON object, without any additional text or explanation."""
    data = {
        "question": item['question'],
        "truth": item['gt'],
        "pred": item['pred'][:500],
    }
    prompt = json.dumps(data, ensure_ascii=False, indent=2)
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return conversation


def build_conversation_for_clf_task(item: dict) -> list[dict]:
    system_prompt = """You are a helpful assistant that determines whether the prediction is correct.
Given a multiple-choice question with correct answer and a predicted answer, you will output a JSON object with a boolean field 'answer' indicating
whether the 'pred' sentence has correctly answered the question (true) or not (false) according to the 'truth' option.
The 'pred' may be a single option or repeated characters of the same option or a complete sentence. It's okay if the 'pred' is not exactly the same as the 'truth' option, as long as it conveys the same meaning. 
But it should not be a combination of multiple options, as that would be incorrect.
The sentences will be provided in the fields 'input', 'question', 'truth', and 'options'.
Please respond only with the JSON object, without any additional text or explanation."""
    data = {
        "question": item['question'],
        "options": item['options'],
        "truth": item['gt'],
        "pred": item['pred'][:500].split(',')[0],
    }
    prompt = json.dumps(data, ensure_ascii=False, indent=2)
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return conversation


@torch.no_grad()
def score_by_llm_batch(model, tokenizer, pred_list: list[dict], batch_size: int, task: str, device: str = 'cuda'):
    res = []
    for i in tqdm(range(0, len(pred_list), batch_size)):
        batch = pred_list[i:i + batch_size]

        conversation_list = []
        for item in batch:
            if task == 'gen':
                conversation = build_conversation_for_gen_task(item)
            elif task == 'clf':
                conversation = build_conversation_for_clf_task(item)
            else:
                assert False, f"Unknown task type: {task}"
            conversation_list.append(conversation)

        inputs_temp = tokenizer.apply_chat_template(conversation_list, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(inputs_temp, return_tensors="pt", truncation=True, max_length=1024, padding=True)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
        )
        outputs = outputs[:, inputs['input_ids'].shape[1]:]  # Skip the input part
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for response, item in zip(responses, batch):
            response = response.strip()
            response = response.replace("'", '"')  # Ensure valid JSON format
            try:
                scores = json.loads(response)
                if isinstance(scores, dict) and 'answer' in scores:
                    scores = scores['answer']
                elif isinstance(scores, bool):
                    scores = scores
                else:
                    print(f"Unexpected response format: {response}")
                    scores = None
            except Exception as e:
                print(f"Error parsing response: {response}, Error: {e}")
                scores = None
            res.append({
                **item,
                'correct': scores
            })

    return res


def score_by_remote_llm_batch(remote_client: RemoteLLMClient, pred_list: list[dict], batch_size: int, task: str):
    """
    使用远程 API 进行批量评分（替代本地模型评分）

    Args:
        remote_client: RemoteLLMClient 实例
        pred_list: 预测结果列表
        batch_size: 批量大小
        task: 任务类型 ('gen' 或 'clf')

    Returns:
        包含评分结果的列表
    """
    res = []
    for i in tqdm(range(0, len(pred_list), batch_size), desc=f"远程评分 ({task})"):
        batch = pred_list[i:i + batch_size]

        conversation_list = []
        for item in batch:
            if task == 'gen':
                conversation = build_conversation_for_gen_task(item)
            elif task == 'clf':
                conversation = build_conversation_for_clf_task(item)
            else:
                assert False, f"Unknown task type: {task}"
            conversation_list.append(conversation)

        # 调用远程 API（temperature 设为 0.1 以兼容部分 API 服务）
        responses = remote_client.batch_chat(conversation_list, max_tokens=50, temperature=0.1)

        for response, item in zip(responses, batch):
            if response is None:
                res.append({**item, 'correct': None})
                continue

            response = response.strip()
            response = response.replace("'", '"')
            try:
                scores = json.loads(response)
                if isinstance(scores, dict) and 'answer' in scores:
                    scores = scores['answer']
                elif isinstance(scores, bool):
                    scores = scores
                else:
                    print(f"Unexpected response format: {response}")
                    scores = None
            except Exception as e:
                print(f"Error parsing response: {response}, Error: {e}")
                scores = None
            res.append({
                **item,
                'correct': scores
            })

    return res



def eval_file(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        preds = json_data['preds']
        args = json_data['args']
    
    batch_size = 16
    

    score_llm = None
    try:
        score_llm, tokenizer = load_llm(f"{args['llm_directory']}{args['score_llm']}", args['device'])
        print(f'scoring by llm')
        preds = score_by_llm_batch(score_llm, tokenizer, preds, batch_size, 'clf', args['device'])
        acc = sum(map(lambda r: int(r['correct']), preds)) / len(preds)
        json_data['acc_by_llm'] = acc
        json_data['preds'] = preds
        new_filepath = filepath.replace('.json', '_scored.json')
        with open(new_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"Scoring completed. acc_by_llm: {acc:.4f}, saved to {new_filepath}")
    except Exception as e:
        print(f"Error during scoring: {type(e)}  {e}")
        if score_llm is not None:
            del score_llm

        return preds, -1
