from typing import List, Optional, Sequence

import torch


def levenshtein_distance(seq1: Sequence, seq2: Sequence) -> int:
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    prev_row = list(range(m + 1))
    for i in range(1, n + 1):
        curr_row = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,
                curr_row[j - 1] + 1,
                prev_row[j - 1] + cost,
            )
        prev_row = curr_row
    return prev_row[m]


def cer(pred_texts: List[str], gt_texts: List[str]) -> float:
    total_dist = 0
    total_chars = 0
    for pred, gt in zip(pred_texts, gt_texts):
        total_dist += levenshtein_distance(pred, gt)
        total_chars += len(gt)
    if total_chars == 0:
        return 0.0
    return total_dist / total_chars


def wer(pred_texts: List[str], gt_texts: List[str]) -> float:
    total_dist = 0
    total_words = 0
    for pred, gt in zip(pred_texts, gt_texts):
        pred_words = pred.split()
        gt_words = gt.split()
        total_dist += levenshtein_distance(pred_words, gt_words)
        total_words += len(gt_words)
    if total_words == 0:
        return 0.0
    return total_dist / total_words


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    blank_idx: int = 0,
    input_lengths: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    """Decode CTC output with greedy strategy.

    Args:
        log_probs: [T, B, C] log probabilities.
    """
    pred_ids = torch.argmax(log_probs, dim=2)  # [T, B]
    pred_ids = pred_ids.transpose(0, 1).contiguous()  # [B, T]

    if input_lengths is None:
        lengths = [pred_ids.size(1)] * pred_ids.size(0)
    else:
        lengths = [int(x) for x in input_lengths.tolist()]

    decoded: List[List[int]] = []
    for b, seq in enumerate(pred_ids):
        seq_list = seq[:lengths[b]].tolist()
        prev = None
        out: List[int] = []
        for idx in seq_list:
            if idx != blank_idx and idx != prev:
                out.append(idx)
            prev = idx
        decoded.append(out)

    return decoded
