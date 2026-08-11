# Chapter 2: Tokenizer

The tokenizer is fundamental to LLMs—it converts raw text into the discrete tokens that the model processes. A good tokenizer balances vocabulary size, compression efficiency, and multilingual coverage.

## 2.1 Why Tokenization Matters

Language models operate on discrete tokens, not raw characters or bytes. Tokenization determines:
- **Vocabulary size**: Too small = long sequences, poor efficiency. Too large = sparse embeddings, wasted parameters
- **Coverage**: Handling rare words, code, math, and multiple languages
- **Efficiency**: Tokens per word ratio affects both training and inference cost

## 2.2 Byte Pair Encoding (BPE)

BPE is the dominant tokenization algorithm for modern LLMs. It builds a vocabulary by iteratively merging the most frequent pairs of bytes/characters.

### 2.2.1 Algorithm

1. Start with a base vocabulary of all unique bytes/characters
2. Count all adjacent pairs in the training corpus
3. Merge the most frequent pair into a new token
4. Repeat until desired vocabulary size is reached

### 2.2.2 BPE Variants

- **Original BPE**: Merges based on frequency
- **Byte-level BPE (BBPE)**: Operates on bytes, ensuring any text can be tokenized (no UNK tokens)
- **SentencePiece**: Google's implementation supporting BPE and Unigram models with lossless tokenization
- **Tiktoken**: OpenAI's fast BPE implementation used in GPT-4

## 2.3 Tokenizer Design Choices

### 2.3.1 Vocabulary Size

Typical ranges for modern LLMs:
- Small models: 32K-50K tokens
- Large models: 100K-256K tokens
- Multilingual models: 128K+ tokens to cover multiple scripts

### 2.3.2 Special Tokens

Special tokens serve as control signals:
- `<BOS>` / `<EOS>`: Beginning/End of Sequence
- `<PAD>`: Padding token
- `<UNK>`: Unknown token (avoided in BBPE)
- `<SEP>`: Separator
- Custom: `<|user|>`, `<|assistant|>`, `<|system|>` for chat formats

### 2.3.3 Multilingual Considerations

- Need sufficient coverage across all target languages
- CJK (Chinese/Japanese/Korean) characters pose challenges due to large character sets
- Some languages may be over-tokenized (many tokens per word), reducing efficiency

## 2.4 Training a BPE Tokenizer

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# Configure trainer
trainer = trainers.BpeTrainer(
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
)

# Train on corpus
tokenizer.train(files=["corpus.txt"], trainer=trainer)
```

## 2.5 Tokenization Issues and Solutions

### 2.5.1 Common Problems

- **Number tokenization**: "123" might split as "12" + "3" or "1" + "23"
- **Code tokenization**: Variable names and indentation can fragment awkwardly
- **Trailing whitespace**: Tokens with trailing spaces vs. without can cause issues

### 2.5.2 Solutions

- Use byte-level BPE to handle arbitrary inputs
- Add domain-specific pre-tokenization rules (e.g., for numbers and code)
- Experiment with vocabulary size for your specific use case
- Consider sentencepiece for lossless tokenization

## 2.6 Key Takeaways

1. BPE is the standard tokenization algorithm for modern LLMs
2. Byte-level BPE ensures any input can be tokenized
3. Vocabulary size is a critical hyperparameter affecting efficiency and coverage
4. Multilingual tokenization requires careful design for fair language representation
5. Special tokens define the interaction format between users and models
