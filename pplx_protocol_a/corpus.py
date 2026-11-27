"""Corpus loading, exact UTF-8 lexical-atom partitioning, and frozen manifests.

Design invariants (addressing known bugs in prior black-box BPB harnesses):
  1. Exact UTF-8 bytes are preserved everywhere: NO `.lower()` and NO
     punctuation stripping (fixing BPBench `sample_echo.py`'s downward bias).
  2. Byte denominators are always measured on raw UTF-8 text bytes, never on
     decoded SentencePiece/BPE token strings.
  3. Every document is partitioned into contiguous, non-overlapping lexical
     atoms whose concatenation equals the exact UTF-8 text byte-for-byte,
     aligned to leading-space BPE word boundaries so open-weight proxy models
     and closed-source APIs share exact evaluation spans.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Exact-cover regex: splits text into contiguous atoms where leading ASCII
# spaces stay attached to the following word (matching modern BPE pre-tokenizers
# like o200k_base / cl100k_base / Llama-3 / Qwen / DeepSeek), while trailing
# punctuation forms its own separate atom. Every character is covered once.
_ATOM_RE = re.compile(r" ?[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*| ?[^\sA-Za-z0-9]+|\s+")


@dataclass(frozen=True)
class PositionSpec:
    """One frozen evaluation position inside a document."""

    position_id: str
    slice_name: str
    doc_id: str
    pos_index: int
    frac: float
    prefix_text: str
    anchor_text: str
    target_text: str
    remaining_text: str
    prefix_utf8_bytes: int
    target_utf8_bytes: int
    anchor_utf8_bytes: int
    doc_utf8_bytes: int
    prefix_sha256: str
    target_hex: str


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_exact_atoms(text: str) -> list[tuple[int, int, str]]:
    """Partition `text` into contiguous (char_start, char_end, atom_str) spans.

    Invariant: `"".join(a for _, _, a in spans) == text`.
    """
    spans = [(m.start(), m.end(), m.group(0)) for m in _ATOM_RE.finditer(text)]
    if "".join(a for _, _, a in spans) != text:
        raise ValueError("Internal error: atom partition does not exactly reconstruct text")
    return spans


def _is_word_atom(atom: str) -> bool:
    """True when atom is a lexical word (optional single leading space + alnum)."""
    stripped = atom[1:] if atom.startswith(" ") else atom
    return bool(stripped) and stripped[0].isalnum()


def extract_anchor(prefix_text: str, anchor_words: int = 8) -> str:
    """Extract the exact verbatim trailing `anchor_words` tokens of `prefix_text`.

    Preserves exact internal whitespace, casing, and punctuation — does NOT
    normalize newlines or strip trailing punctuation inside the anchor.
    Trims only trailing whitespace at the very right edge of `prefix_text`
    (which belongs to the next word's leading space in BPE).
    """
    if anchor_words < 1:
        raise ValueError("anchor_words must be >= 1")
    trimmed = prefix_text.rstrip(" ")
    matches = list(re.finditer(r"\S+", trimmed))
    if len(matches) < anchor_words:
        raise ValueError(f"prefix has fewer than {anchor_words} non-whitespace tokens")
    start = matches[-anchor_words].start()
    return trimmed[start:]


def load_jsonl_corpus(
    slice_paths: Sequence[tuple[str, Path]],
    *,
    text_field: str = "text",
    id_field: str = "id",
) -> list[dict[str, str]]:
    """Load documents from one or more `(slice_name, jsonl_path)` pairs."""
    docs: list[dict[str, str]] = []
    for slice_name, path in slice_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle):
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                text = obj.get(text_field)
                if not isinstance(text, str) or not text.strip():
                    continue
                # Normalize Windows CRLF -> LF so raw file line-endings do not
                # cause artificial echo mismatches against model LF output.
                text = text.replace("\r\n", "\n").replace("\r", "\n")
                doc_id = str(obj.get(id_field, f"{slice_name}:{line_no}"))
                docs.append({"slice_name": slice_name, "doc_id": doc_id, "text": text})
    return docs


def build_position_manifest(
    docs: Sequence[dict[str, str]],
    *,
    fracs: Sequence[float] = (0.30, 0.55, 0.80),
    anchor_words: int = 8,
    min_atoms: int = 40,
    max_prefix_chars: int = 16000,
    words_only: bool = False,
) -> list[PositionSpec]:
    """Select deterministic evaluation positions per document.

    For each document and each fraction in `fracs`:
      - Picks the atom index closest to `frac * len(atoms)` (optionally
        snapping to the nearest word atom if `words_only=True`, though keeping
        all non-whitespace atoms preserves unbiased full-text BPB!).
      - Guarantees the prefix ends right before the target atom (no trailing
        space split off from a leading-space target atom).
      - Avoids splitting inside a pure-whitespace atom (snaps forward 1 atom).
    """
    positions: list[PositionSpec] = []
    for doc in docs:
        slice_name = doc["slice_name"]
        doc_id = doc["doc_id"]
        text = doc["text"]
        doc_bytes = len(text.encode("utf-8"))
        atoms = split_exact_atoms(text)
        if len(atoms) < min_atoms:
            continue

        used_indices: set[int] = set()
        for frac in fracs:
            idx = max(anchor_words + 6, int(len(atoms) * frac))
            # Snap forward past pure-whitespace atoms (or non-word atoms if requested)
            while idx < len(atoms) - 2:
                atom_str = atoms[idx][2]
                if not atom_str.strip():
                    idx += 1
                    continue
                if words_only and not _is_word_atom(atom_str):
                    idx += 1
                    continue
                if idx in used_indices:
                    idx += 1
                    continue
                break
            if idx >= len(atoms) - 2 or idx in used_indices:
                continue

            char_start, char_end, target_text = atoms[idx]
            prefix_text = text[:char_start]
            if max_prefix_chars and len(prefix_text) > max_prefix_chars:
                # Trim from the left at a whitespace boundary so the right edge
                # (including the anchor and exact target boundary) is untouched.
                cut = prefix_text[-max_prefix_chars:]
                ws = cut.find(" ")
                prefix_text = cut[ws + 1 :] if ws != -1 else cut

            # Ensure prefix doesn't end with an isolated trailing space when target
            # does not start with one; keep exact boundary invariant:
            # prefix_text + target_text == text_suffix_ending_at_char_end.
            try:
                anchor_text = extract_anchor(prefix_text, anchor_words=anchor_words)
            except ValueError:
                continue
            # If prefix had trailing spaces stripped by extract_anchor, fold them
            # onto target_text so `anchor_text` is the EXACT suffix of `prefix_text`
            # and `prefix_text + target_text` remains byte-identical to the source.
            trailing_spaces = len(prefix_text) - len(prefix_text.rstrip(" "))
            if trailing_spaces > 0:
                prefix_text = prefix_text[:-trailing_spaces]
                target_text = (" " * trailing_spaces) + target_text
                char_start -= trailing_spaces

            remaining_text = text[char_start : min(len(text), char_start + 128)]
            if not prefix_text.endswith(anchor_text):
                continue

            target_bytes = target_text.encode("utf-8")
            used_indices.add(idx)
            pos_id = f"{slice_name}__{doc_id}__p{idx}"
            positions.append(
                PositionSpec(
                    position_id=pos_id,
                    slice_name=slice_name,
                    doc_id=doc_id,
                    pos_index=idx,
                    frac=float(frac),
                    prefix_text=prefix_text,
                    anchor_text=anchor_text,
                    target_text=target_text,
                    remaining_text=remaining_text,
                    prefix_utf8_bytes=len(prefix_text.encode("utf-8")),
                    target_utf8_bytes=len(target_bytes),
                    anchor_utf8_bytes=len(anchor_text.encode("utf-8")),
                    doc_utf8_bytes=doc_bytes,
                    prefix_sha256=sha256_hex(prefix_text.encode("utf-8")),
                    target_hex=target_bytes.hex(),
                )
            )
    return positions


def save_manifest(path: Path, positions: Sequence[PositionSpec], metadata: dict[str, Any]) -> str:
    """Write a frozen manifest JSON with SHA-256 content binding."""
    payload = {
        "schema_version": "protocol_a_manifest_v1",
        "metadata": metadata,
        "count": len(positions),
        "positions": [asdict(p) for p in positions],
    }
    raw = canonical_json_bytes(payload)
    digest = sha256_hex(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return digest


def load_manifest(path: Path) -> tuple[list[PositionSpec], dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_hex(raw)
    data = json.loads(raw.decode("utf-8"))
    positions = [PositionSpec(**item) for item in data["positions"]]
    return positions, data.get("metadata", {}), digest
