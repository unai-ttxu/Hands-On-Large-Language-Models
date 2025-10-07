import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Choose a model ID from the Hugging Face Hub
# For this example, we'll use a small model for efficiency
model_id = "microsoft/Phi-3-mini-4k-instruct"

# 2. Load the tokenizer and the model
# The library will download the files if they aren't cached
# The torch_dtype argument helps with memory usage
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency on Apple Silicon
    low_cpu_mem_usage=True,  # Reduces CPU memory during loading
)

# 3. Move the model to the GPU (Apple Silicon's MPS device)
# This is crucial for performance on your M4 MacBook
model = model.to("mps")

# 4. Prepare your input text and tokenize it
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# 5. Move the input tensors to the MPS device
input_ids = input_ids.to("mps")

# 6. Generate the generation_output
# The generate() method handles the generation process
generation_output = model.generate(input_ids, max_new_tokens=20)

# 7. Decode the output tokens back into human-readable text
print(tokenizer.decode(generation_output[0]))
