import argparse
import os
import openai
import numpy as np
import nltk
import tiktoken
from collections import defaultdict
from tqdm import tqdm
nltk.download("punkt_tab")

# Configure
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "text-embedding-3-small"
MAX_TOKENS = 512

# Tokenizer
enc = tiktoken.encoding_for_model(MODEL_NAME)


def split_text_into_chunks(text, max_tokens=MAX_TOKENS, overlap_tokens=64):
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk = [], []
    current_chunk_tokens = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = len(enc.encode(sentence))
        if current_chunk_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens
            i += 1
        else:
            chunks.append(" ".join(current_chunk))
            if overlap_tokens > 0:
                overlap_chunk = []
                overlap_tokens_count = 0
                j = len(current_chunk) - 1
                while j >= 0 and overlap_tokens_count < overlap_tokens:
                    s = current_chunk[j]
                    s_tokens = len(enc.encode(s))
                    overlap_chunk.insert(0, s)
                    overlap_tokens_count += s_tokens
                    j -= 1
                current_chunk = overlap_chunk
                current_chunk_tokens = overlap_tokens_count
            else:
                current_chunk = []
                current_chunk_tokens = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def get_embedding(texts, model=MODEL_NAME):
    response = openai.embeddings.create(input=texts, model=model)
    return [d.embedding for d in response.data]


def load_all_fomc(folder_path):
    fomc_dict = defaultdict(str)
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                content = f.read().strip()
                key = os.path.splitext(filename)[0]
                fomc_dict[key] = content
    return fomc_dict


def main(args):
    fomc_dict = load_all_fomc(args.fomc_dir)
    for date, text in tqdm(fomc_dict.items()):
        chunks = split_text_into_chunks(text)
        chunk_embeddings = get_embedding(chunks)
        np.save(os.path.join(os.getcwd(), f"fomc/embeddings/{date}.npy"), np.array(chunk_embeddings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fomc_dir",
        default=os.path.join(os.getcwd(), "fomc/minutes")
    )
    args = parser.parse_args()
    main(args)