from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import os


def load_bilingual_dict_ja(txt_path: str) -> Dict[str, List[str]]:

    ja_to_en = defaultdict(list)

    def is_safe_char(char: str) -> bool:

        code = ord(char)

        if code < 0x0080:
            return True
        if 0x3040 <= code <= 0x309F:  # Hiragana
            return True
        if 0x30A0 <= code <= 0x30FF:  # Katakana
            return True
        if 0x31F0 <= code <= 0x31FF:  # Katakana Phonetic Extensions
            return True
        if 0xFF65 <= code <= 0xFF9F:  # Halfwidth Katakana
            return True
        if 0x4E00 <= code <= 0x9FFF:  # CJK Unified Ideographs
            return True
        if 0x3400 <= code <= 0x4DBF:  # CJK Extension A
            return True
        if 0x20000 <= code <= 0x2A6DF:  # CJK Extension B
            return True
        if 0x2A700 <= code <= 0x2B73F:  # CJK Extension C
            return True
        if 0x2B740 <= code <= 0x2B81F:  # CJK Extension D
            return True
        if 0x2B820 <= code <= 0x2CEAF:  # CJK Extension E
            return True
        if 0x2CEB0 <= code <= 0x2EBEF:  # CJK Extension F
            return True
        if 0x30000 <= code <= 0x3134F:  # CJK Extension G
            return True
        if 0x3000 <= code <= 0x303F:
            return True
        if 0xFF00 <= code <= 0xFF64:  # Fullwidth Forms
            return True

        return False

    def has_unsafe_chars(word: str) -> bool:

        for char in word:
            if not is_safe_char(char):
                return True
        return False

    skipped = 0
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                ja_word, en_word = parts[0], parts[1]
                if ja_word.isascii():
                    continue
                if has_unsafe_chars(ja_word):
                    skipped += 1
                    continue
                ja_to_en[ja_word].append(en_word)

    print(f"Dictionary: {len(ja_to_en)} words")
    if skipped > 0:
        print(f"  (Skip {skipped} Japanese words with special characters)")

    return dict(ja_to_en)


def get_embedding_layer(model) -> nn.Embedding:

    if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):
        return model.embeddings.word_embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        return model.transformer.wte
    raise ValueError("Cannot find embedding layer")


def set_embedding_layer(model, new_embedding: nn.Embedding):

    if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):
        model.embeddings.word_embeddings = new_embedding
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        model.model.embed_tokens = new_embedding
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        model.transformer.wte = new_embedding
    else:
        raise ValueError("Cannot set embedding layer")


def lexicon_walk_mapping_ja(
        ja_to_en: Dict[str, List[str]],
        tokenizer,
        embed_weight: torch.Tensor
) -> Dict[str, torch.Tensor]:

    new_vocab = {}
    for ja_word, en_words in ja_to_en.items():
        all_embeds = []
        for en_word in en_words:
            en_ids = tokenizer.encode(en_word, add_special_tokens=False)
            if len(en_ids) == 0:
                continue
            en_embed = embed_weight[en_ids].mean(dim=0)
            all_embeds.append(en_embed)

        if all_embeds:
            new_vocab[ja_word] = torch.stack(all_embeds).mean(dim=0)

    return new_vocab


def filter_new_tokens_ja(
        tokenizer,
        new_vocab: Dict[str, torch.Tensor]
) -> Tuple[List[str], List[torch.Tensor]]:

    words_to_add = []
    embeds_to_add = []

    def is_safe_char(char: str) -> bool:

        code = ord(char)
        if code < 0x0080:
            return True
        if 0x3040 <= code <= 0x309F:  # Hiragana
            return True
        if 0x30A0 <= code <= 0x30FF:  # Katakana
            return True
        if 0x31F0 <= code <= 0x31FF:  # Katakana Extensions
            return True
        if 0xFF65 <= code <= 0xFF9F:  # Halfwidth Katakana
            return True
        if 0x4E00 <= code <= 0x9FFF:
            return True
        if 0x3400 <= code <= 0x4DBF:
            return True
        if 0x20000 <= code <= 0x2A6DF:
            return True
        if 0x2A700 <= code <= 0x2CEAF:
            return True
        if 0x2CEB0 <= code <= 0x3134F:
            return True
        if 0x3000 <= code <= 0x303F:
            return True
        if 0xFF00 <= code <= 0xFF64:  # Fullwidth Forms
            return True
        return False

    def has_unsafe_chars(word: str) -> bool:
        for char in word:
            if not is_safe_char(char):
                return True
        return False

    skipped_count = 0
    for ja_word, embed in new_vocab.items():

        if has_unsafe_chars(ja_word):
            skipped_count += 1
            continue

        ids = tokenizer.encode(ja_word, add_special_tokens=False)
        decoded = tokenizer.decode(ids).replace(' ', '').replace('▁', '')
        if decoded != ja_word:
            words_to_add.append(ja_word)
            embeds_to_add.append(embed)

    if skipped_count > 0:
        print(f"Skip {skipped_count} words with special characters")

    return words_to_add, embeds_to_add



def greenplm_create_ja(
        model_name: str,
        dict_path: str,
        output_dir: str = './greenplm_output_ja',
        device: str = 'cuda',
        verify: bool = True
):

    print(f"Original Model: {model_name}")
    print(f"Dictionary: {dict_path}")
    print(f"Output: {output_dir}")
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    print(f"Tokenizer: {type(tokenizer).__name__} (Slow)")
    print(f"Original Vocab size: {len(tokenizer)}")
    embed_layer = get_embedding_layer(model)
    original_vocab_size, embed_dim = embed_layer.weight.shape
    original_weight = embed_layer.weight.data.clone()
    original_dtype = original_weight.dtype
    original_device_tensor = original_weight.device
    print(f"Embedding: {original_vocab_size} × {embed_dim}")
    ja_to_en = load_bilingual_dict_ja(dict_path)
    new_vocab = lexicon_walk_mapping_ja(ja_to_en, tokenizer, original_weight)
    words_to_add, embeds_to_add = filter_new_tokens_ja(tokenizer, new_vocab)
    print(f"  Need to add: {len(words_to_add)} tokens")
    if len(words_to_add) == 0:
        print("  No tokens are needed to be added")
        return model, tokenizer

    num_added = tokenizer.add_tokens(words_to_add)
    new_vocab_size = len(tokenizer)
    print(f"  Vocab size: {original_vocab_size} → {new_vocab_size}")
    print("Updating embedding...")
    new_weight = torch.zeros(
        new_vocab_size, embed_dim,
        dtype=original_dtype,
        device=original_device_tensor
    )

    new_weight[:original_vocab_size] = original_weight

    for word, embed in zip(words_to_add, embeds_to_add):
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id >= original_vocab_size:
            new_weight[token_id] = embed.to(dtype=original_dtype, device=original_device_tensor)

    new_embedding = nn.Embedding(
        new_vocab_size, embed_dim,
        padding_idx=embed_layer.padding_idx,
        dtype=original_dtype,
        device=original_device_tensor
    )
    new_embedding.weight.data = new_weight
    set_embedding_layer(model, new_embedding)
    model.config.vocab_size = new_vocab_size
    new_embed_layer = get_embedding_layer(model)
    diff = (original_weight - new_embed_layer.weight.data[:original_vocab_size]).abs().max().item()
    print(f"  The embedding diff: {diff:.2e}")

    if verify:
        print("Verifying...")
        _verify_tokenization_ja(tokenizer, model_name)
        _verify_model_output_ja(model, model_name, tokenizer, device)

    print(f"Saved into: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


    tokenizer_json_path = os.path.join(output_dir, 'tokenizer.json')
    if os.path.exists(tokenizer_json_path):
        os.remove(tokenizer_json_path)
        print("  (Deleted tokenizer.json to ensure Fast tokenizer compatibility)")

    import json
    from sentence_transformers import SentenceTransformer
    try:
        st_model_orig = SentenceTransformer(model_name, device='cpu')
        has_normalize = any(
            type(module).__name__ == 'Normalize'
            for module in st_model_orig
        )
        orig_max_seq_length = st_model_orig.max_seq_length
        del st_model_orig
    except:
        has_normalize = False
        orig_max_seq_length = 512

    # modules.json
    modules = [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}
    ]
    if has_normalize:
        modules.append({"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"})

    with open(os.path.join(output_dir, 'modules.json'), 'w') as f:
        json.dump(modules, f, indent=2)

    # 1_Pooling/config.json
    pooling_dir = os.path.join(output_dir, '1_Pooling')
    os.makedirs(pooling_dir, exist_ok=True)
    pooling_config = {
        "word_embedding_dimension": embed_dim,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False
    }
    with open(os.path.join(pooling_dir, 'config.json'), 'w') as f:
        json.dump(pooling_config, f, indent=2)

    # 2_Normalize
    if has_normalize:
        normalize_dir = os.path.join(output_dir, '2_Normalize')
        os.makedirs(normalize_dir, exist_ok=True)

    # sentence_bert_config.json
    st_config = {
        "max_seq_length": orig_max_seq_length,
        "do_lower_case": False
    }
    with open(os.path.join(output_dir, 'sentence_bert_config.json'), 'w') as f:
        json.dump(st_config, f, indent=2)

    print(f"(Added SentenceTransformer configuration, Normalize={has_normalize}, max_seq_length={orig_max_seq_length})")

    # Japanese test
    print("\nJAPANESE Tokenization test:")
    for word in ["こんにちは", "機械学習", "人工知能", "東京"]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        decoded = tokenizer.decode(ids).replace(' ', '')
        status = "✓" if decoded == word else "✗"
        print(f"  {status} '{word}' → {ids}")

    print("\n" + "=" * 60)
    print("Finish!")
    print("=" * 60)

    return model, tokenizer


def _verify_tokenization_ja(tokenizer, original_model_name: str):

    tokenizer_orig = AutoTokenizer.from_pretrained(original_model_name, use_fast=False)

    test_texts = ["hello", "world", "machine", "learning", "Hello world"]

    print("English Tokenization:")
    all_ok = True
    for text in test_texts:
        ids_orig = tokenizer_orig.encode(text, add_special_tokens=False)
        ids_new = tokenizer.encode(text, add_special_tokens=False)

        if ids_orig == ids_new:
            print(f"'{text}'")
        else:
            print(f"'{text}': {ids_orig} → {ids_new}")
            all_ok = False

    if all_ok:
        print("  English tokenization remain unchanged")
    else:
        print("  English tokenization is broken!")


def _verify_model_output_ja(model, original_model_name: str, tokenizer, device: str):

    model_orig = AutoModel.from_pretrained(original_model_name, trust_remote_code=True)
    model_orig.to(device)
    model_orig.eval()
    tokenizer_orig = AutoTokenizer.from_pretrained(original_model_name, use_fast=False)
    test_texts = ["Hello world", "Machine learning is great"]
    print("Model Output:")

    with torch.no_grad():
        for text in test_texts:

            inputs = tokenizer_orig(text, return_tensors='pt').to(device)

            out_orig = model_orig(**inputs).last_hidden_state
            out_new = model(**inputs).last_hidden_state

            if out_orig.shape != out_new.shape:
                print(f"'{text}' Shape changed")
                continue

            diff = (out_orig - out_new).abs().max().item()
            status = "✓" if diff < 1e-5 else "✗"
            print(f"{status} '{text}' (diff={diff:.2e})")




def greenplm_load_ja(model_dir: str, device: str = 'cuda', use_fast: bool = True):

    print(f"Loading model: {model_dir}")
    print(f"Use {'Fast' if use_fast else 'Slow'} Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=use_fast)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    model.eval()
    print(f"Vocab size: {len(tokenizer)}")

    return model, tokenizer


def greenplm_create_st_ja(
        model_name: str,
        dict_path: str,
        output_dir: str = './greenplm_output_ja',
        device: str = 'cuda',
        verify: bool = True
):

    from sentence_transformers import SentenceTransformer
    print("=" * 60)
    print("GreenPLM Japanese model creation (SentenceTransformer)")
    print("=" * 60)
    print("Loading model...")
    model = SentenceTransformer(model_name, device=device)
    transformer = model[0]
    auto_model = transformer.auto_model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    transformer.tokenizer = tokenizer
    print(f"  Tokenizer: {type(tokenizer).__name__} (Slow)")
    print(f"  Original vocab size: {len(tokenizer)}")

    # Get embedding
    embed_layer = get_embedding_layer(auto_model)
    original_vocab_size, embed_dim = embed_layer.weight.shape
    original_weight = embed_layer.weight.data.clone()
    original_dtype = original_weight.dtype
    original_device_tensor = original_weight.device
    ja_to_en = load_bilingual_dict_ja(dict_path)

    # LWM
    print("Lexicon Walk Mapping...")
    new_vocab = lexicon_walk_mapping_ja(ja_to_en, tokenizer, original_weight)

    # Expand vocab size
    print("Expanding vocab size...")
    words_to_add, embeds_to_add = filter_new_tokens_ja(tokenizer, new_vocab)
    print(f"  Need to add: {len(words_to_add)} tokens")

    if len(words_to_add) == 0:
        print("  No new words are needed")
        return model

    num_added = tokenizer.add_tokens(words_to_add)
    new_vocab_size = len(tokenizer)
    print(f"  Vocab size: {original_vocab_size} → {new_vocab_size}")

    # Update embedding
    print("Updating embedding...")

    new_weight = torch.zeros(new_vocab_size, embed_dim, dtype=original_dtype, device=original_device_tensor)
    new_weight[:original_vocab_size] = original_weight

    for word, embed in zip(words_to_add, embeds_to_add):
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id >= original_vocab_size:
            new_weight[token_id] = embed.to(dtype=original_dtype, device=original_device_tensor)

    new_embedding = nn.Embedding(
        new_vocab_size, embed_dim,
        padding_idx=embed_layer.padding_idx,
        dtype=original_dtype,
        device=original_device_tensor
    )
    new_embedding.weight.data = new_weight

    set_embedding_layer(auto_model, new_embedding)
    auto_model.config.vocab_size = new_vocab_size

    diff = (original_weight - get_embedding_layer(auto_model).weight.data[:original_vocab_size]).abs().max().item()
    print(f"  The embedding diff: {diff:.2e}")

    # Verify
    if verify:
        print("Verifying...")
        model_orig = SentenceTransformer(model_name, device=device)

        test_sentences = ["Hello world", "Machine learning is great"]
        emb_orig = model_orig.encode(test_sentences, convert_to_tensor=True)
        emb_new = model.encode(test_sentences, convert_to_tensor=True)

        print("English Embedding:")
        for i, text in enumerate(test_sentences):
            diff = (emb_orig[i] - emb_new[i]).abs().max().item()
            status = "✓" if diff < 1e-4 else "✗"
            print(f"{status} '{text}' (diff={diff:.2e})")

    # Save
    print(f"Saved into: {output_dir}")
    model.save(output_dir)

    # Japanese test
    print("Japanese test:")
    for word in ["こんにちは", "機械学習", "人工知能"]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        decoded = tokenizer.decode(ids).replace(' ', '')
        status = "✓" if decoded == word else "✗"
        print(f"  {status} '{word}' → {ids}")

    # Embedding test
    print("Japanese Embedding test:")
    test_ja = ["こんにちは世界", "機械学習は面白い"]
    embeddings = model.encode(test_ja, convert_to_tensor=True)
    print(f"  Input: {test_ja}")
    print(f"  Output shape: {embeddings.shape}")

    print("\nFinish!")
    return model


def greenplm_load_st_ja(model_dir: str, device: str = 'cuda'):

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_dir, device=device)
    print(f"Finish loading: {model_dir}")

    return model



if __name__ == "__main__":

    model, tokenizer = greenplm_create_ja(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        dict_path="./ja-en.txt",
        output_dir='./aligned_greenplm_all_MiniLM_L12_v2_japanese',
        device='cuda',
        verify=True
    )
