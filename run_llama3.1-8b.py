import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import transformers.models.llama.modeling_llama as modelling_llama
import pickle

def calculate_perplexity(model, tokenizer, inputs):
    # Perform model inference
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss  # Average cross-entropy loss across the batch

    # Multiply loss by the number of tokens to get total loss
    num_tokens = inputs["input_ids"].numel()
    total_loss = loss.item() * num_tokens

    return total_loss, num_tokens

def split_text_into_chunks(text, tokenizer, max_length=512):
    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
    chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
    return chunks

def evaluate_perplexity_on_wikitext(model, tokenizer, chunk_size=512):
    # Load WikiText-2 test dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    
    # Concatenate all texts
    text = "\n\n".join(dataset["text"])
    
    # Split text into tokenized chunks
    tokenized_chunks = split_text_into_chunks(text, tokenizer, chunk_size)
    print(f"Number of chunks: {len(tokenized_chunks)}")
    # Initialize accumulators for total loss and token count
    total_loss = 0.0
    total_tokens = 0

    for chunk in tokenized_chunks:
        if len(chunk) > 0:  # Skip empty chunks
            inputs = {"input_ids": chunk.unsqueeze(0)}  # Add batch dimension
            chunk_loss, num_tokens = calculate_perplexity(model, tokenizer, inputs)
            total_loss += chunk_loss
            total_tokens += num_tokens

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

def main():
    # 1) Point to your local Llama 3.1–8B folder:
    model_path = "/data/fat/hcp/HF-Llama-3.1-8B"  # or an absolute path if needed
    # 2) Load the tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    ## Iterate TopK
    topks = torch.arange(0.0, 1.0, 0.05,dtype=torch.float32).tolist()
    perplexity_list = []
    output_list = []
    for topk_svd in topks:
        # 3) Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            topk_svd=topk_svd, # [0.0-1.0], 1.0 means no truncation
        )
        model.eval()
        # 4) Define the prompt
        prompt = "Explain the concept of quantum entanglement in simple terms." # Tokenized to len=14
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # 5）Calculate perplexity on WikiText-2
        avg_perplexity = evaluate_perplexity_on_wikitext(model, tokenizer)
        print(f"Average perplexity on WikiText-2: {avg_perplexity:.2f}")
        # 6) Generate text
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,
            repetition_penalty=1.1,
        )
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        perplexity_list.append(avg_perplexity)
        output_list.append(output)
        modelling_llama.MLP_counter = 0

    ## Save the results
    with open("perplexity_list.pkl", "wb") as f:
        pickle.dump(perplexity_list, f)
    with open("output_list.pkl", "wb") as f:
        pickle.dump(output_list, f)

if __name__ == "__main__":
    main()