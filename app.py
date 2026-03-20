"""
SummaScope — AI-Powered Document Analyzer
==========================================
Extracts summaries, named entities, and key phrases from any text
using pre-trained transformer models.

Models:
  - Summarization : sshleifer/distilbart-cnn-12-6  (DistilBART)
  - NER           : dslim/bert-base-NER             (BERT)
  - Key phrases   : YAKE  (unsupervised, no model required)

Deployed on Hugging Face Spaces with a Gradio interface.
"""

import json
import re
import textwrap

import gradio as gr
import yake
from transformers import pipeline


# ═══════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════

print("Loading summarization model …")
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    tokenizer="sshleifer/distilbart-cnn-12-6",
)

print("Loading NER model …")
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    tokenizer="dslim/bert-base-NER",
    aggregation_strategy="simple",
)

print("All models loaded ✓")


# ═══════════════════════════════════════════════════════════════════
#  Utility Helpers
# ═══════════════════════════════════════════════════════════════════

# YAKE keyword extractor (language-agnostic, fast)
kw_extractor = yake.KeywordExtractor(
    lan="en", n=2, dedupLim=0.9, top=10, features=None
)


def _clean(text: str) -> str:
    """Normalise whitespace and strip boilerplate."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _chunk_text(text: str, max_tokens: int = 900) -> list[str]:
    """Split long text into chunks the summarizer can handle."""
    words = text.split()
    chunks, current = [], []
    for w in words:
        current.append(w)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def _reading_time(word_count: int) -> str:
    minutes = word_count / 238
    if minutes < 1:
        return "< 1 min"
    return f"~{minutes:.0f} min"


# ═══════════════════════════════════════════════════════════════════
#  Core Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze(text: str, summary_ratio: int = 30) -> tuple[str, str, str, str]:
    """
    Run the full analysis pipeline.

    Parameters
    ----------
    text : str
        The document / article text.
    summary_ratio : int
        Approximate target length for the summary as a percentage
        of the original (10–50).

    Returns
    -------
    summary : str
    entities_json : str   (JSON array of {entity, type, score})
    key_phrases : str     (comma-separated)
    stats : str           (plain-text stats)
    """
    text = _clean(text)
    if len(text.split()) < 30:
        return (
            "Please provide at least 30 words of text for meaningful analysis.",
            "[]",
            "",
            "",
        )

    # ── 1. Summarization ──────────────────────────────────────────
    word_count = len(text.split())
    target_len = max(40, int(word_count * summary_ratio / 100))
    target_len = min(target_len, 300)
    min_len = max(20, target_len // 3)

    chunks = _chunk_text(text, max_tokens=900)
    summaries = []
    for chunk in chunks:
        chunk_words = len(chunk.split())
        if chunk_words < 30:
            continue
        c_max = max(40, min(target_len, chunk_words - 10))
        c_min = max(10, min(min_len, c_max - 5))
        out = summarizer(chunk, max_length=c_max, min_length=c_min, do_sample=False)
        summaries.append(out[0]["summary_text"].strip())
    summary = " ".join(summaries) if summaries else "Could not generate summary."

    # ── 2. Named Entity Recognition ───────────────────────────────
    # BERT NER has a 512-token limit; process in windows
    ner_input = text[:4000]  # keep it fast
    raw_entities = ner_pipeline(ner_input)

    seen, entities = set(), []
    LABEL_MAP = {
        "PER": "Person",
        "ORG": "Organization",
        "LOC": "Location",
        "MISC": "Miscellaneous",
    }
    for ent in raw_entities:
        name = ent["word"].strip()
        label = LABEL_MAP.get(ent["entity_group"], ent["entity_group"])
        key = (name.lower(), label)
        if key not in seen and len(name) > 1 and ent["score"] > 0.75:
            seen.add(key)
            entities.append(
                {"entity": name, "type": label, "score": round(float(ent["score"]), 3)}
            )
    entities.sort(key=lambda e: e["score"], reverse=True)
    entities_json = json.dumps(entities[:20], ensure_ascii=False)

    # ── 3. Key Phrases (YAKE) ─────────────────────────────────────
    keywords = kw_extractor.extract_keywords(text[:5000])
    key_phrases = ", ".join(kw for kw, _score in keywords[:10])

    # ── 4. Text Statistics ────────────────────────────────────────
    sentence_count = len(re.split(r"[.!?]+", text))
    stats = (
        f"{word_count:,} words · {sentence_count} sentences · "
        f"{_reading_time(word_count)} read"
    )

    return summary, entities_json, key_phrases, stats


# ═══════════════════════════════════════════════════════════════════
#  Gradio Interface
# ═══════════════════════════════════════════════════════════════════

EXAMPLES = [
    [
        textwrap.dedent("""\
        The Transformer architecture, introduced in the 2017 paper "Attention Is All You Need" by \
        Vaswani et al. at Google, fundamentally changed the landscape of natural language processing. \
        Unlike recurrent neural networks that process sequences step by step, Transformers use a \
        mechanism called self-attention to process all positions in a sequence simultaneously. This \
        parallelism enables much faster training on modern hardware. The key innovation is the \
        multi-head attention mechanism, which allows the model to jointly attend to information from \
        different representation subspaces at different positions. The architecture consists of an \
        encoder-decoder structure, though many successful variants use only the encoder (BERT) or \
        only the decoder (GPT). Since its introduction, the Transformer has become the foundation \
        for virtually all state-of-the-art language models, including GPT-4, Claude, PaLM, and \
        LLaMA. Its influence extends beyond NLP into computer vision (Vision Transformers), \
        protein structure prediction (AlphaFold), and audio processing. The scaling properties of \
        Transformers have driven the large language model revolution, with models now containing \
        hundreds of billions of parameters trained on trillions of tokens of text data."""),
        30,
    ],
    [
        textwrap.dedent("""\
        CRISPR-Cas9 is a revolutionary gene-editing technology that allows scientists to make \
        precise changes to DNA sequences in living organisms. Discovered as part of the bacterial \
        immune system, CRISPR (Clustered Regularly Interspaced Short Palindromic Repeats) works \
        by using a guide RNA to direct the Cas9 enzyme to a specific location in the genome, \
        where it cuts the DNA. The cell's natural repair mechanisms then fix the break, either \
        disabling the gene or inserting new genetic material. Jennifer Doudna and Emmanuelle \
        Charpentier were awarded the 2020 Nobel Prize in Chemistry for developing this technology. \
        CRISPR has enormous potential in medicine, agriculture, and biotechnology. In medicine, \
        it is being explored for treating genetic disorders like sickle cell disease, muscular \
        dystrophy, and certain cancers. Clinical trials are underway for several CRISPR-based \
        therapies. In agriculture, CRISPR is used to develop disease-resistant crops and improve \
        nutritional content. However, the technology also raises significant ethical concerns, \
        particularly regarding germline editing — changes that can be inherited by future \
        generations. The case of He Jiankui, who controversially created the first gene-edited \
        babies in 2018, highlighted the urgent need for international regulatory frameworks."""),
        30,
    ],
    [
        textwrap.dedent("""\
        The Apollo 11 mission, launched on July 16, 1969, was the spaceflight that first landed \
        humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed \
        the American crew that landed the Apollo Lunar Module Eagle on July 20, 1969. Armstrong \
        became the first person to step onto the lunar surface six hours and 39 minutes later; \
        Aldrin joined him 19 minutes after that. They spent about two and a quarter hours together \
        exploring the site they had named Tranquility Base upon landing. Michael Collins piloted \
        the command module Columbia alone in lunar orbit while they were on the Moon's surface. \
        The mission fulfilled President John F. Kennedy's 1961 goal of landing a man on the Moon \
        and returning him safely to Earth before the end of the decade. Armstrong's first step \
        onto the lunar surface was broadcast on live TV to a worldwide audience estimated at 600 \
        million people. The astronauts returned to Earth and splashed down in the Pacific Ocean on \
        July 24 after more than eight days in space. The mission carried several scientific \
        experiments and collected 47.5 pounds of lunar material for return to Earth for analysis. \
        Apollo 11 effectively ended the Space Race and is widely regarded as one of humanity's \
        greatest achievements."""),
        30,
    ],
]

with gr.Blocks(title="SummaScope — Document Analyzer") as demo:
    gr.Markdown("# SummaScope — AI Document Analyzer")
    gr.Markdown(
        "Paste any article or document text below to get an abstractive summary, "
        "named entities, and key phrases — powered by DistilBART and BERT."
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Document Text",
                placeholder="Paste your article or document text here …",
                lines=10,
            )
            summary_slider = gr.Slider(
                minimum=10,
                maximum=50,
                value=30,
                step=5,
                label="Summary Length (%)",
                info="Target summary length as a percentage of the original text.",
            )
            analyze_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=2):
            summary_output = gr.Textbox(label="Summary", lines=6, interactive=False)
            entities_output = gr.JSON(label="Named Entities")
            phrases_output = gr.Textbox(
                label="Key Phrases", interactive=False
            )
            stats_output = gr.Textbox(label="Text Statistics", interactive=False)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[text_input, summary_slider],
        label="Example Documents",
    )

    analyze_btn.click(
        fn=analyze,
        inputs=[text_input, summary_slider],
        outputs=[summary_output, entities_output, phrases_output, stats_output],
        api_name="predict",
    )

demo.launch()
