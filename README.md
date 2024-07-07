# マルチモーダルモデルベンチマーク

このリポジトリには、時系列+言語のマルチモーダルモデルのベンチマークテストを実行するためのスクリプトが含まれています。

## セットアップ

1. clone
```
git clone <repository_url>
cd <repository_name>
```

2. packeges
```
pip install -r requirements.txt
```

## 使用方法

スクリプトは以下のように実行します：
```
python benchmark.py --model_id <HuggingFace_model_id> --output_dir <output_directory>
```

例：
```
python benchmark.py --model_id HachiML/Mists-7B-v01-simpleQA --output_dir .
```

## Notebookでの使用

Jupyter Notebookで使用する場合は、以下のようにスクリプトを実行できます：

```python
!python benchmark.py --model_id HachiML/Mists-7B-v01-simpleQA --output_dir results
```

結果は指定された出力ディレクトリに保存され、ノートブック上にも進捗と最終スコアが表示されます。
注意事項

 - このスクリプトは大量のメモリを使用する可能性があります。十分なリソースを確保してください。
 - GPU環境を推奨しますが、利用できない場合はCPUにフォールバックします。
