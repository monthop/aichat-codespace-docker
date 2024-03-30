from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("/tmp/my_gpt2_tokenizer")
model = GPT2LMHeadModel.from_pretrained("/tmp/my_gpt2_model")

def generate_response(user_input, max_length=100):
    # Tokenize user input and generate response
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = generate_response(user_input)
    print("Bot:", response)
