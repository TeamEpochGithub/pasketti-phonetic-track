import torch
import re
from typing import List
from pathlib import Path
import numpy as np

class TorchaudioCTCDecoder:
    def __init__(
        self,
        tokenizer: "PhonemeTokenizer",
        beam_width: int = 15,
        kenlm_model_path: str = None,
        alpha: float = 0.3,
        beta: float = 1.0,
    ):
        """
        Fast C++ CTC Beam Search using torchaudio (Flashlight).
        
        Natively supports lexicon-free decoding, meaning it scores individual 
        phonemes directly against the KenLM without forcing them into "words".
        """
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        
        try:
            from torchaudio.models.decoder import ctc_decoder
        except ImportError:
            raise ImportError("Please install torchaudio to use this decoder: pip install torchaudio")

        # 1. Prepare tokens list (indices MUST exactly match the model's logits)
        tokens = []
        for i in range(tokenizer.vocab_size):
            if i == tokenizer.blank_token_id:
                tokens.append("<blank>")
            else:
                char = tokenizer.decode([i])
                if not char or char == " ": 
                    char = f"<special_{i}>"
                tokens.append(char)

        # 2. Build the C++ Flashlight decoder
        if kenlm_model_path:
            # Note: Torchaudio heavily prefers .binary KenLM models over .arpa
            kenlm_model_path = str(Path(kenlm_model_path).resolve())
            print(f"[TorchaudioDecoder] Loading Lexicon-Free KenLM from {kenlm_model_path} (alpha={alpha}, beta={beta})")
            
            self.decoder = ctc_decoder(
                lexicon=None,  # Crucial: Tells the decoder to score individual tokens via LM
                tokens=tokens,
                lm=kenlm_model_path,
                blank_token="<blank>",
                beam_size=beam_width,
                lm_weight=alpha,
                word_score=beta, # Acts as token insertion penalty in lexicon-free mode
            )
        else:
            print(f"[TorchaudioDecoder] Building pure Beam Search (no LM)")
            self.decoder = ctc_decoder(
                lexicon=None,
                tokens=tokens,
                blank_token="<blank>",
                beam_size=beam_width,
            )

    def _iter_log_probs(self, logits):
        """Yields single-sequence log_softmax tensors to avoid padding hallucinations."""
        if isinstance(logits, (list, tuple)):
            for item in logits:
                if isinstance(item, np.ndarray):
                    item = torch.from_numpy(item)
                # torchaudio C++ backend strictly requires float32
                yield torch.nn.functional.log_softmax(item.cpu().float(), dim=-1)
        else:
            if isinstance(logits, np.ndarray):
                logits = torch.from_numpy(logits)
            log_probs = torch.nn.functional.log_softmax(logits.cpu().float(), dim=-1)
            for b in range(log_probs.shape[0]):
                yield log_probs[b]

    def _squash_repetitions(self, text: str, min_repeat: int = 4) -> str:
        """Post-processing to squash EXTREME repeating patterns only."""
        squashed = re.sub(r'(.+?)\1{' + str(min_repeat - 1) + r',}', r'\1', text)
        return squashed

    def __call__(self, logits) -> List[str]:
        """
        Args:
            logits: [B, T, vocab_size] tensor or list of [T, vocab_size] tensors.
        """
        decoded = []
        
        for emissions in self._iter_log_probs(logits):
            # Flashlight expects [B, T, C], so we unsqueeze to batch size of 1
            emissions_batched = emissions.unsqueeze(0)
            
            # Decode the single sequence
            results = self.decoder(emissions_batched)
            
            # Grab the best hypothesis for this sequence (beam 0)
            best_hyp = results[0][0]
            
            # Flashlight perfectly collapses blanks and repeats into clean token IDs
            token_ids = best_hyp.tokens.tolist()
            
            # Decode using your exact tokenizer logic
            text = self.tokenizer.decode(token_ids)
            
            # Clean up and apply your squasher
            text = re.sub(r'\s+', ' ', text).strip()
            text = self._squash_repetitions(text)
            
            decoded.append(text)

        return decoded