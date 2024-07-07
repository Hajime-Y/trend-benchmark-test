"""
マルチモーダルモデルベンチマークスクリプト

このスクリプトは、時系列+言語のマルチモーダルモデルのベンチマークテストを実行します。
HuggingFace Hubの HachiML/timeseries_simpleQA_ja のtest splitを使用し、
指定されたモデルの性能を評価します。

使用方法:
    python benchmark.py --model_id <HuggingFace_model_id> --output_dir <output_directory>

引数:
    --model_id: 評価するHuggingFaceモデルのID
    --output_dir: 結果を保存するディレクトリ（デフォルト: 'results'）

結果:
    指定された出力ディレクトリにJSONファイルとして保存されます。
    ファイル名は 'results_<model_id>.json' の形式です。
"""

import argparse
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import os

def generate_response(content, time_series_data, max_new_tokens=100):
    """
    指定されたコンテンツと時系列データに基づいてモデルの応答を生成します。

    Args:
        content (str): ユーザーからの入力テキスト
        time_series_data (list): 時系列データ
        max_new_tokens (int): 生成する最大トークン数

    Returns:
        str: モデルが生成した応答
    """
    messages = [{
        "role": "user",
        "content": f"<time_series>\n{content}"
    }]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    inputs = processor(
        prompt,
        time_series_data,
        return_tensors='pt',
        time_series_padding="max_length",
        time_series_max_length=512
    )

    for key, item in inputs.items():
        inputs[key] = inputs[key].to(device)

    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    output = processor.decode(output[0], skip_special_tokens=False)

    return output

def evaluate_trend(model_output):
    """
    モデルの出力からトレンドを評価します。

    Args:
        model_output (str): モデルが生成した応答

    Returns:
        str: 評価されたトレンド（"上昇", "下降", "トレンドなし", または "不明"）
    """
    if "上" in model_output:
        return "上昇"
    elif "下" in model_output:
        return "下降"
    elif "せん" in model_output:
        return "トレンドなし"
    else:
        return "不明"

def get_true_trend(messages):
    """
    データセットのメッセージから真のトレンドを取得します。

    Args:
        messages (list): データセットのメッセージリスト

    Returns:
        str: 真のトレンド（"上昇", "下降", "トレンドなし", または "不明"）
    """
    assistant_content = messages[1]["content"]
    if "上昇" in assistant_content:
        return "上昇"
    elif "下降" in assistant_content:
        return "下降"
    elif "せん" in assistant_content:
        return "トレンドなし"
    else:
        return "不明"

def run_benchmark(model_id, output_dir):
    """
    指定されたモデルIDに対してベンチマークテストを実行します。

    Args:
        model_id (str): 評価するHuggingFaceモデルのID
        output_dir (str): 結果を保存するディレクトリ

    Returns:
        float: モデルのスコア（正解率）
    """
    global tokenizer, processor, model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    print(f"モデルを読み込んでいます: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )

    model.language_model.to(torch.bfloat16)

    print("データセットを読み込んでいます")
    dataset = load_dataset("HachiML/timeseries_simpleQA_ja", split="test")

    correct = 0
    total = 0

    print("評価を開始します")
    for item in tqdm(dataset, desc="評価中"):
        content = "与えられた期間でデータは上がっていますか。下がっていますか。"
        time_series_data = item["time_series_values"]
        response = generate_response(content, time_series_data, max_new_tokens=20)

        model_prediction = evaluate_trend(response)
        true_trend = get_true_trend(item["messages"])

        if model_prediction == true_trend:
            correct += 1
        total += 1

    score = correct / total if total > 0 else 0
    print(f"評価完了。スコア: {score:.4f}")

    results = {model_id: score}
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{model_id.replace('/', '_')}.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"結果を保存しました: {output_file}")
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="マルチモーダルモデルのベンチマークを実行します")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFaceモデルID")
    parser.add_argument("--output_dir", type=str, default="results", help="結果を保存するディレクトリ")
    
    args = parser.parse_args()
    
    run_benchmark(args.model_id, args.output_dir)