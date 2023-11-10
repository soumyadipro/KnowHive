import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset
import torch
import os

df = pd.read_csv('D:\internship\KnowHive\Field Engineer dataset.csv')
df = df.dropna()
input_data = df['Incident Description'].tolist()
target_data = df['Resolution Steps'].tolist()
with open("train.txt", "w") as f:
    for incident_desc, resolution_steps in zip(input_data, target_data):
        f.write(f"Incident Description: {incident_desc}\n")
        f.write(f"Resolution Steps: {resolution_steps}\n")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./incident_model",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

# Check if a pre-trained model checkpoint exists and load it
if not os.path.exists(training_args.output_dir):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Fine-tune the model
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="train.txt",
        block_size=512,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model checkpoint
    trainer.save_model("./incident_model_checkpoint")

else:
    # Load the pre-trained model checkpoint
    model = GPT2LMHeadModel.from_pretrained("incident_model_checkpoint")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load old conversation history from file
conversation_history = []
if os.path.exists("conversation_history.txt"):
    with open("conversation_history.txt", "r") as history_file:
        content = history_file.read()
        if content.strip():  # Check if the file is not empty
            conversation_history = eval(content)

def update_conversation_history(question, answer, is_new_chat=False):
    if is_new_chat:
        conversation_history.insert(0, (f"New Incident Description: {question}", f"New Chat Solution: {answer}"))
    else:
        conversation_history.append((f"Old Chats of the Field Engineer: {question}", f"Old Chat History: {answer}"))

def save_conversation_history():
    with open("conversation_history.txt", "w") as history_file:
        history_file.write(str(conversation_history))

def display_conversation_history():
    for i, (q, a) in enumerate(conversation_history, 1):
        print(f"{i}. {q}\n   Answer: {a}\n")

def generate_answer(question):
    input_text = question
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=50, truncation=True)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long) 
    response = model.generate(input_ids, max_length=300, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7, do_sample=True, pad_token_id=model.config.eos_token_id)

    answer = tokenizer.decode(response[0], skip_special_tokens=True)
    return answer

# User input for new incident description
new_incident_description = input("Enter your incident description: ")
new_answer = generate_answer(new_incident_description)
update_conversation_history(new_incident_description, new_answer, is_new_chat=True)

# Save conversation history
save_conversation_history()

# Display the conversation history
display_conversation_history()
