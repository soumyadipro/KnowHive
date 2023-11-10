import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

df = pd.read_csv('D:\internship\KnowHive\Field Engineer dataset.csv')
df = df.dropna()
input_data = df['Incident Description'].tolist()
target_data = df['Resolution Steps'].tolist()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2LMHeadModel.from_pretrained("incident_model_checkpoint")

# Function to calculate self-perplexity
def calculate_self_perplexity(data, model, tokenizer):
    total_loss = 0.0
    total_tokens = 0

    for text in data:
        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=50, truncation=True)
        with torch.no_grad():
            output = model(input_ids, labels=input_ids)
            loss = output.loss
            total_loss += loss.item()
            total_tokens += len(input_ids[0])

    average_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(average_loss))  # Corrected the calculation
    return perplexity.item()

# Calculate self-perplexity
self_perplexity = calculate_self_perplexity(input_data, model, tokenizer)
print(f"Self-Perplexity: {self_perplexity:.2f}")
