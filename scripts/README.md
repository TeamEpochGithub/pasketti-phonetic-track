# KenLM Phoneme Language Model for CTC Decoding

## Why?

The CTC acoustic model (wav2vec2) sometimes gets stuck in repetitive loops, producing outputs like `ktktktkt`, `mmmmmmmm`, or `ɚːɪŋɚːɪŋɹɚɹɚɹɚ`. A KenLM language model trained on real phonetic transcripts penalizes these impossible sequences during beam search, steering the decoder toward valid phoneme patterns.

## How it works

The wav2vec2 model outputs 55 CTC labels (individual phonemes like `ʔ`, `æ`, `p`, `ɫ`).

pyctcdecode uses the **space label** as a word boundary — it groups consecutive non-space phonemes into "words". For example, the CTC output `ʔ æ p ɫ [space] h æ t` becomes two "words": `ʔæpɫ` and `hæt`.

KenLM then scores these phoneme-cluster words using n-gram probabilities learned from training data. When the model loops (`ktktktkt`), that "word" is unknown → heavy penalty → beam gets pruned.

**Key distinction:**
- 55 = CTC output labels (individual phonemes)
- ~45K = LM vocabulary (phoneme-cluster "words" between spaces, e.g. `ʔæpɫ`, `mama`, `ðə`)

This is analogous to an English character-level CTC model having 27 labels (`a-z` + space) but using a 100K-word English dictionary LM.

## Files

| File | Description |
|------|-------------|
| `scripts/build_phoneme_corpus.py` | Extracts phonetic transcripts from training JSONL → corpus + unigram list |
| `phoneme_corpus.txt` | Raw phonetic transcripts, one utterance per line |
| `phoneme_unigrams.txt` | Unique word list (one per line, sorted by frequency) |
| `phoneme_lm.arpa` | KenLM 3-gram language model in ARPA format |

## Usage

### 1. Build the corpus

```bash
uv run python scripts/build_phoneme_corpus.py
```

Options:
- `--jsonl-files`: paths to training JSONL files (default: both ultrasuite + talkbank)
- `--output`: corpus output path (default: `phoneme_corpus.txt`)
- `--unigrams-output`: unigram list path (default: `phoneme_unigrams.txt`)

### 2. Build the KenLM ARPA model

```bash
# Install KenLM (one-time)
sudo apt-get install -y build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev
git clone https://github.com/kpu/kenlm.git /tmp/kenlm
cd /tmp/kenlm && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
sudo cp bin/lmplz bin/build_binary /usr/local/bin/

# Build 3-gram model (run from the project root)
lmplz -o 3 --discount_fallback < phoneme_corpus.txt 2>/dev/null > phoneme_lm.arpa
```

A 3-gram is recommended over 5-gram — with ~45K unique words and ~153K training sentences, higher-order models are too sparse to generalize well.

### 3. Decoder config

In `configs/decoder/beam_search.yaml`:

```yaml
_target_: src.utils.decoder.PyCTCDecoder
beam_width: 15
kenlm_model_path: phoneme_lm.arpa
unigrams_path: phoneme_unigrams.txt
alpha: 0.3    # LM weight
beta: 1.0     # word-insertion bonus
```

## Tuning for children's speech

Since this is children's speech data with frequent mispronunciations and short (~2s) utterances:

- **`alpha` (LM weight):** Keep low (0.2–0.5). Too high will over-correct valid mispronunciations toward "standard" phoneme sequences.
- **`beta` (word-insertion bonus):** Keep moderate (0.5–1.5). Too high inflates output length for short utterances.
- **Repetition squashing:** The decoder's `_squash_repetitions` only triggers at 4+ consecutive repeats to preserve legitimate patterns like `mama`, `baba`, `nɑ nɑ`.

Set `kenlm_model_path: null` in the config to disable the LM and fall back to pure beam search.
