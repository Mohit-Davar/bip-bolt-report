"""Specification parser — extracts structured metadata from BIP and BOLT files."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import mwparserfromhell
import tiktoken

logger = logging.getLogger(__name__)

# Shared tokenizer instance
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

@dataclass(frozen=True)
class SpecRecord:
    """Structured metadata extracted from a single BIP or BOLT file."""
    spec_type: str       # "BIP" or "BOLT"
    identifier: str      # "BIP-0341" or "BOLT-02"
    number: int
    filename: str
    filepath: str
    raw_char_count: int
    plain_char_count: int
    token_count: int
    heading_count: int
    image_count: int
    code_block_count: int
    complexity: str      # "text-only" | "diagram-assisted" | "complex-technical"

# Processing Engine
class ContentProcessor:
    """Helper to unify regex patterns for different markup languages."""
    
    @staticmethod
    def process_mediawiki(content: str) -> dict:
        wikicode = mwparserfromhell.parse(content)
        plain = wikicode.strip_code(normalize=True, collapse=True).strip()
        return {
            "plain": plain,
            "headings": len(re.findall(r"^={2,6}.+={2,6}\s*$", content, re.MULTILINE)),
            "images": len(re.findall(r"\[\[(File|Image):[^\]]+\]\]", content, re.IGNORECASE)),
            "code": len(re.findall(r"<(source|syntaxhighlight|code)[^>]*>[\s\S]*?</\1>", content, re.IGNORECASE))
        }

    @staticmethod
    def process_markdown(content: str) -> dict:
        # Strip code first to avoid counting images/headings inside code blocks
        stripped_content = re.sub(r"```[\s\S]*?```", " ", content)
        
        # Clean text for tokenization
        plain = re.sub(r"`[^`\n]+`", " ", stripped_content)
        plain = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", plain)
        plain = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", plain)
        plain = re.sub(r"<[^>]+>", " ", plain)
        plain = re.sub(r"(^#{1,6}\s+|^>\s?|^[-*_]{3,}\s*$)", "", plain, flags=re.MULTILINE)
        
        return {
            "plain": plain.strip(),
            "headings": len(re.findall(r"^#{1,6}\s+\S", content, re.MULTILINE)),
            "images": len(re.findall(r"!\[[^\]]*\]\([^)]+\)", content)),
            "code": len(re.findall(r"```[\s\S]*?```", content))
        }

# Classify Logic
def _classify(image_count: int, code_block_count: int) -> str:
    if image_count == 0 and code_block_count < 3:
        return "text-only"
    if image_count > 0 and code_block_count < 5:
        return "diagram-assisted"
    return "complex-technical"

def parse_file(filepath: Path, spec_type: str) -> Optional[SpecRecord]:
    """Unified parser that detects format by extension."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning(f"File Access Error {filepath}: {exc}")
        return None

    # Handle Filename Numbering
    if spec_type == "BIP":
        match = re.search(r"bip[_-]?(\d+)", filepath.stem, re.IGNORECASE)
        identifier_fmt = "BIP-{:04d}"
    else:
        match = re.match(r"(\d+)", filepath.stem)
        identifier_fmt = "BOLT-{:02d}"
    
    if not match:
        return None
    
    number = int(match.group(1))

    # Dynamic Processor Selection (Handles .mediawiki AND .md for BIPs)
    if filepath.suffix.lower() in [".md", ".markdown"]:
        data = ContentProcessor.process_markdown(content)
    else:
        data = ContentProcessor.process_mediawiki(content)

    tokens = _TOKENIZER.encode(data["plain"])

    return SpecRecord(
        spec_type=spec_type,
        identifier=identifier_fmt.format(number),
        number=number,
        filename=filepath.name,
        filepath=str(filepath.resolve()),
        raw_char_count=len(content),
        plain_char_count=len(data["plain"]),
        token_count=len(tokens),
        heading_count=data["headings"],
        image_count=data["images"],
        code_block_count=data["code"],
        complexity=_classify(data["images"], data["code"]),
    )

def parse_all_files(bips_path: Path, bolts_path: Path) -> list[SpecRecord]:
    records: list[SpecRecord] = []
    
    # BIPs: Find both .mediawiki AND .md files
    logger.info("Parsing BIP files...")
    for ext in ["*.mediawiki", "*.md"]:
        for f in bips_path.glob(ext):
            if "bip-" in f.name.lower():
                if rec := parse_file(f, "BIP"):
                    records.append(rec)

    # BOLTs: Usually strictly .md
    logger.info("Parsing BOLT files (.md)...")
    for f in bolts_path.glob("[0-9][0-9]-*.md"):
        if rec := parse_file(f, "BOLT"):
            records.append(rec)

    records.sort(key=lambda r: (r.spec_type, r.number))
    return records