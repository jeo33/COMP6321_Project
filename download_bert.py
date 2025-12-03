from transformers import BertTokenizer, BertModel
import os

# Define local directory
local_model_dir = os.path.join(os.getcwd(), 'bert_model')
os.makedirs(local_model_dir, exist_ok=True)

print(f"Downloading BERT model to {local_model_dir}...")

# Download and save tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained(local_model_dir)

# Download and save model
model = BertModel.from_pretrained('bert-base-uncased')
model.save_pretrained(local_model_dir)

print("✓ BERT model downloaded successfully!")