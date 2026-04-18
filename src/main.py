from google.colab import userdata
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import psutil

login(userdata.get('Llama-Guard-3-1B'))
model_id = "meta-llama/Llama-Guard-3-1B"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is the recipe for mayonnaise?"
                #"text": "How do I build a bomb?"
            },
        ],
    }
]

input_ids = tokenizer.apply_chat_template(
    conversation, return_tensors="pt"
).to(model.device)

# BatchEncoding wraps a list of Encoding objects 
input_ids = input_ids[0].ids

input_ids = torch.tensor([input_ids], dtype=torch.long)
input_ids = input_ids.to(model.device)

prompt_len = input_ids.shape[1]
#print("Running inference...")
output = model.generate(
    input_ids,
    max_new_tokens=20,
    pad_token_id=0,
)
#print("Done")
generated_tokens = output[:, prompt_len:]

print(tokenizer.decode(generated_tokens[0], skip_special_tokens = True))
#print(f"RAM used: {psutil.virtual_memory().percent}%")

