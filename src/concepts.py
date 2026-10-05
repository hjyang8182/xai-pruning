import csv
import torch
import torch.nn.functional as F
import clip
import pandas as pd
from src.config import VOCAB_PATH, EMB_PATH, CSV_PATH, VOCAB_BATCH, N_LEARNED_FEATURES, clip_device

# Load raw vocabulary (20k most frequent)
def load_vocab():
    with open(VOCAB_PATH) as f:
        return [line.strip() for line in f if line.strip()]

# Encode a list of text strings with CLIP's text encoder
def embed_texts(clip_model, texts, batch_size=VOCAB_BATCH):
    clip_model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            tokens = clip.tokenize(texts[i:i+batch_size], truncate=True).to(clip_device)
            emb = clip_model.encode_text(tokens).float().cpu()
            all_embeddings.append(emb)
            if i % 5000 == 0:
                print(f'  {i}/{len(texts)}')
    return torch.cat(all_embeddings, dim=0)

# Save clip embeddings of the vocabulary
def save_vocab_embeddings(clip_model, words):
    embeddings = embed_texts(clip_model, words)
    torch.save(embeddings, EMB_PATH)
    print(f'Saved embeddings: {embeddings.shape}')


def load_vocab_embeddings():
    return torch.load(EMB_PATH, map_location='cpu').float()

def name_concepts(autoencoder, text_emb, vocab, csv_path=None):
    csv_path = csv_path or CSV_PATH
    dic_vec = autoencoder.decoder.weight.detach().cpu().squeeze()
    dic_vec = dic_vec / dic_vec.norm(dim=0, keepdim=True)
    top_idxs = torch.matmul(text_emb, dic_vec).argmax(dim=0)
    concept_names = [vocab[i] for i in top_idxs.tolist()]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for idx, name in enumerate(concept_names):
            writer.writerow([idx, name])
    print(f'Saved {len(concept_names)} concept names to {csv_path}')
    return concept_names

def load_concept_names(csv_path=None):
    return pd.read_csv(csv_path or CSV_PATH, header=None)[1].tolist()

def compute_pruning_criteria(train_acts, autoencoder, text_emb):
    # TODO: change to gpu?
    dic_norm = F.normalize(autoencoder.decoder.weight.detach().cpu().squeeze(), dim=0)
    name_alignment = torch.zeros(N_LEARNED_FEATURES)
    for c in range(N_LEARNED_FEATURES):
        name_alignment[c] = F.cosine_similarity(dic_norm[:, c].unsqueeze(0), text_emb[c].unsqueeze(0))

    return {
        'activation_freq': (train_acts > 0).float().mean(dim=0),
        'max_act':         train_acts.max(dim=0).values,
        'name_alignment':  name_alignment,
        'act_variance':    train_acts.var(dim=0),
    }