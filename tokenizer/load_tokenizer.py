from transformers import AutoTokenizer

tokenizer = AutoTokenizer().from_pretrained("Qwen/Qwen3-1.7B")
tokenizer.save_pretrained("tokenizer/qwen")