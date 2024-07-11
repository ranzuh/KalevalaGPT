import sys
import time
import torch
from gpt_model import encode, decode, GPTModel, device

def clear_screen():
    # Move the cursor to the top-left corner
    sys.stdout.write("\033[H")
    # Clear the screen
    sys.stdout.write("\033[J")

def generate_sentence(model, initial_sentence, max_length=10000, delay=0.05):
    sentence_idxs = encode(initial_sentence)
    for _ in range(max_length - len(initial_sentence)):
        context = torch.tensor([sentence_idxs], dtype=torch.long, device=device)
        sentence_idxs = model.generate(context, 1)[0].tolist()
        sentence = decode(sentence_idxs)
        # Clear the line, return to the beginning
        clear_screen()
        # Print the sentence so far
        sys.stdout.write('\r' + sentence)
        sys.stdout.flush()
        time.sleep(delay)  # Delay to simulate real-time generation

save_path = sys.argv[1]

model = GPTModel().to(device)

model.load_state_dict(torch.load(save_path))
model.eval()

initial_sentence = "\n"
generate_sentence(model, initial_sentence)