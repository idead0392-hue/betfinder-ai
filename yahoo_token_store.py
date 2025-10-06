import json
import os

def save_tokens(token_dict, filename=".yahoo_tokens.json"):
    with open(filename, "w") as f:
        json.dump(token_dict, f)

def load_tokens(filename=".yahoo_tokens.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None

def clear_tokens(filename=".yahoo_tokens.json"):
    if os.path.exists(filename):
        os.remove(filename)
