# Synthetic Speech Data Pipeline (S.S.D.P.)
### Egyptian Arabic · End-to-End

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Stage 1 — Text Generation](#stage-1--text-generation)
4. [Stage 2 — TTS Synthesis](#stage-2--tts-synthesis)
5. [Stage 3 — Review](#stage-3--review)
6. [Stage 4 — Training-Ready Export](#stage-4--training-ready-export)
7. [Extended Contribution: Real Speech Pipeline](#extended-contribution-real-speech-pipeline)
8. [Synthetic vs Real Data: A Practical Comparison](#synthetic-vs-real-data-a-practical-comparison)
9. [Why Synthetic Data Can Mislead a Downstream STT Model](#why-synthetic-data-can-mislead-a-downstream-stt-model)
10. [Deliverables — Sample Dataset](#deliverables--sample-dataset)
11. [Output Format](#output-format)
12. [Observed Quality Issues](#observed-quality-issues)
13. [Trade-offs and Limitations](#trade-offs-and-limitations)
14. [Running the Pipeline](#running-the-pipeline)

---

## Overview

This project delivers a complete data pipeline for producing training-ready Egyptian Arabic speech data. It has two complementary tracks:

**Track A — Synthetic pipeline (S.S.D.P.):** Generates Egyptian Arabic text using a large language model, synthesizes audio with a dialect-specific TTS model, scores every (text, audio) pair automatically using STT round-trip evaluation, and exports accepted samples in a HuggingFace-compatible format.

**Track B — Real speech pipeline (bonus contribution):** A previously built pipeline that collected approximately 90 hours of authentic Egyptian Arabic audio, segmented and cleaned it using `inaSpeechSegmenter`, and exported it in the same training-ready format.

Both tracks produce the same output schema so they can be combined directly. The motivation for including both is grounded in a practical reality: synthetic data and real data have complementary failure modes, and mixing them is the correct production strategy for fine-tuning a robust Egyptian Arabic STT model.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SSDP — Track A                           │
│                                                                 │
│  Stage 1           Stage 2           Stage 3        Stage 4     │
│  Text Gen    ──►   TTS Synthesis ──► Review    ──►  Export      │
│  (Gemma-4)         (NAMAA-TTS)       (Whisper)      (HF format) │
│                                                                 │
│  egyptian_dataset  namaa/            reviewed/      dataset/    │
│  .jsonl            manifest.jsonl    namaa_reviewed  metadata   │
│                    + *.wav           .jsonl          .csv       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  Real Speech Pipeline — Track B                 │
│                                                                 │
│  Collection      Cleaning            STT              Export    │
│  YouTube    ──►  inaSpeechSegmenter ──► Whisper   ──►  HF       │
│  Web sounds      (diarization,          large-v3       schema   │
│  Speech repos    silence removal,       auto-                   │
│  (~90 hrs)       music detection)       transcription           │
└─────────────────────────────────────────────────────────────────┘
```

Every long-running stage writes a JSONL manifest as it goes. If the process is interrupted, re-running the notebook resumes from the last completed sample — no work is lost.

---

## Stage 1 — Text Generation

### Model choice

**Gemma-4-E2B-IT** (via KaggleHub) was chosen as the text generator for the following reasons:

- Strong multilingual capability with good coverage of Arabic script
- Instruction-tuned variant responds reliably to structured Arabic prompts
- Small enough (2B) to run on a single GPU without quantization, keeping the pipeline self-contained
- Returns clean JSON arrays when instructed to, which makes output parsing robust

### Prompt design

The system prompt enforces Egyptian colloquial Arabic (العامية المصرية) exclusively, bans Modern Standard Arabic (الفصحى), and explicitly permits natural Arabic–English code-switching — for example, "انت شايل ال laptop معاك؟" — because this is how Egyptians actually speak. Responses are required to be JSON arrays only, which makes extraction deterministic.

Nine topic categories were defined to ensure phonetic and lexical diversity:

| Category | Rationale |
|---|---|
| Greetings and farewells | High-frequency, short utterances; tests basic prosody |
| Food and orders | Practical vocabulary; includes numbers and quantities |
| Directions and places | Spatial language; longer, more complex sentences |
| Numbers and prices | Critical for STT — numbers are a known weak point |
| Family and daily life | Core conversational vocabulary |
| Weather and time | Common triggers for code-switching (e.g. "ال weather النهارده") |
| Work and business | More formal register within colloquial speech |
| Phone conversations | Turn-taking patterns; incomplete sentences |
| Emotions and reactions | Expressive, informal; tests prosody variation |

### Generation parameters

The following sampling parameters were used for all text generation calls:

| Parameter | Value | Rationale |
|---|---|---|
| `temperature` | 1.0 | Full sampling temperature — encourages lexical variety across sentences rather than collapsing to the most probable phrasing |
| `top_p` | 0.85 | Nucleus sampling keeps the output within a high-probability mass while still allowing diverse word choices |
| `do_sample` | True | Enables stochastic sampling; without this, the model would produce near-identical sentences across the same category |
| `repetition_penalty` | 1.2 | Discourages the model from repeating the same words or phrases within a generation, which is a common failure mode when generating multiple sentences in a single prompt |

Together these parameters push the model toward diverse, natural-sounding output while the repetition penalty specifically guards against the corpus becoming phonetically or lexically monotonous — which would narrow the acoustic coverage of the synthesized audio.

### Validation

Each generated sentence is accepted only if it contains a minimum of 3 words and at least 4 Arabic Unicode characters. This filters out hallucinated English-only responses or near-empty outputs without over-restricting code-switched sentences.

---

## Stage 2 — TTS Synthesis

### Model choice

**NAMAA-Egyptian-TTS** (`NAMAA-Space/NAMAA-Egyptian-TTS`) was chosen because it is the only publicly available TTS model trained specifically on Egyptian Arabic. All general-purpose Arabic TTS models (e.g. those built on MSA corpora) produce Modern Standard Arabic prosody that sounds unnatural to Egyptian ears and introduces a prosodic distribution mismatch that would mislead a downstream STT model trained on this data.

### Why not use the fine-tuned Qwen TTS model?

A personal Qwen TTS model fine-tuned on Egyptian Arabic speech was considered but rejected: the fine-tune was trained on Upper Egyptian dialect (الصعيدي) speech data, not Cairene Egyptian Arabic. Using it would mean the synthesized audio carries the phonology and prosody of the wrong dialect, which would mislead a downstream STT model into learning Upper Egyptian acoustic patterns while the transcriptions are Cairene text. NAMAA-TTS, despite its limitations, is the correct choice here because dialect match in the synthesis stage is more important than model familiarity.

### Known limitations of NAMAA-TTS

- **Clean studio conditions**: no background noise, no channel effects. The model will likely under-perform on phone audio or noisy environments unless real data is also included.
- **Prosody on long sentences**: sentences over approximately 15 words can produce unnatural pauses or stress patterns.
- **Code-switched segments**: the model occasionally mispronounces embedded English words. These samples will be caught by the WER filter in Stage 3.

### Resumability

The synthesis stage writes to `namaa/manifest.jsonl` after each sample. On restart, `get_completed_ids()` reads completed IDs and skips them. This means a 450-sample synthesis run interrupted at sample 200 resumes from sample 201.

---

## Stage 3 — Review

### Approach: automated STT round-trip scoring

Each synthesized WAV is transcribed by a fine-tuned Whisper model and the transcription is compared against the original text using Word Error Rate (WER) and Character Error Rate (CER).

### Why this Whisper model?

**`itshamdi404/Egy_Arabic_whisper-small`** is a Whisper-small model fine-tuned specifically on Egyptian Arabic. Using a general Whisper model for review would introduce a dialect mismatch in the evaluator itself — a general model might legitimately transcribe an Egyptian word differently from the TTS prompt text, flagging valid audio as erroneous. Using an Egyptian-fine-tuned evaluator makes the WER/CER scores reflect genuine synthesis quality rather than dialect mismatch between evaluator and synthesizer.



### Arabic normalization

Before computing WER and CER, both the reference text and the transcription pass through `normalize_arabic()`, which:

- Collapses all alef variants (إ أ آ ٱ) to bare alef (ا) — different encodings of the same sound
- Converts alef maqsoura (ى) to ya (ي)
- Converts taa marbuta (ة) to ha (ه)
- Strips all diacritics (tashkeel)
- Strips punctuation
- Collapses whitespace

Without this normalization, a WER score of 0.5 might be entirely due to encoding differences rather than synthesis errors. This is tested with six unit-test assertions covering every normalization rule.

### Thresholds

| Label | Condition | Action |
|---|---|---|
| Accept | WER ≤ 0.15 AND CER ≤ 0.10 | Exported to training set |
| Flag | WER > 0.15 OR CER > 0.10 (and not rejected) | Held for human review |
| Reject | WER > 0.30 OR CER > 0.20 | Excluded |

### Manual review
 
Flagged samples are held in `reviewed/namaa_reviewed.jsonl` with label `"flag"`. The review manifest is a plain JSONL file — each line contains the audio path, original text, WER, CER, and label — so flagged samples can be inspected directly by opening the file, locating the audio path, and listening to the WAV alongside the transcript.
 
The export stage (Stage 4) currently exports only samples labelled `"accept"`. To promote a flagged sample after manual listening, change its label from `"flag"` to `"accept"` in `namaa_reviewed.jsonl` and re-run the export cell — it will be included in the output dataset.
---

## Stage 4 — Training-Ready Export

### Output format: HuggingFace AudioFolder

```
dataset/
├── metadata.csv          # tab-separated: file_name, transcription, category, wer, cer, model
├── data/
│   ├── prompt_00001.wav
│   └── ...
└── dataset_info.json     # schema + pipeline provenance
```

This format was chosen because:

- It loads directly with `datasets.load_dataset("audiofolder", data_dir="dataset/")` with no preprocessing
- `metadata.csv` is human-readable and inspectable with any spreadsheet tool
- The schema matches the format expected by most Whisper and wav2vec2 fine-tuning scripts
- Adding the real-data track requires only appending rows to `metadata.csv` and copying WAVs into `data/` — no schema changes

### Loading the dataset

```python
from datasets import load_dataset
ds = load_dataset("audiofolder", data_dir="dataset/", split="train")
print(ds[0])
# {'audio': {'path': '...', 'array': array([...]), 'sampling_rate': 16000},
#  'transcription': 'إزيك يا محمد، عامل إيه النهارده؟',
#  'category': 'greetings_and_farewells',
#  'wer': 0.083,
#  'cer': 0.042}
```

### Sample rate

All exported WAVs are 16 kHz mono PCM, which is the standard input rate for Whisper and most open-source STT fine-tuning pipelines.

---

## Extended Contribution: Real Speech Pipeline

In parallel with the synthetic pipeline, I previously built a pipeline to collect and prepare approximately 90 hours of authentic Egyptian Arabic audio. This section documents that pipeline so the two tracks can be understood and evaluated together.

### Collection

Audio was collected from three source types:

- **YouTube** — Egyptian Arabic video content covering news, vlogs, interviews, and entertainment
- **Web sounds** — publicly accessible audio files and speech recordings from the web
- **Speech repositories** — open datasets and corpora containing Egyptian Arabic speech

Sources were chosen to represent Cairene dialect specifically. Raw audio ranged from clean studio recordings to phone-quality and ambient-noise conditions, providing the acoustic diversity that the synthetic pipeline fundamentally cannot replicate.

### Processing with inaSpeechSegmenter

Each recording was segmented using `inaSpeechSegmenter`, which performs:

- Speaker diarization (separating different speakers)
- Music and jingle detection and removal
- Silence and noise segment detection

Only segments classified as `"speech"` by the segmenter were retained. Segments shorter than 1 second or longer than 30 seconds were discarded, targeting the utterance lengths most useful for STT fine-tuning.

### Transcription and alignment

Retained segments were transcribed automatically using **Whisper large-v3**. This is an important detail about the real-data samples: unlike the synthetic track where the original LLM-generated text is the ground-truth transcript (it was never spoken, only synthesized from it), the real-data transcripts are machine-generated and carry their own error rate.

Whisper large-v3 performs reasonably well on Egyptian Arabic, but dialectal speech — especially informal, fast, or code-switched utterances — is harder for it than MSA. Estimated word error rate on the automatic transcriptions is in the range of 10–15% before any manual review. This means some transcripts in the real-data sample will contain errors: wrong words, skipped words, or occasional MSA substitutions for colloquial terms (e.g. transcribing "مش" as "ليس").

For the 30 real-data samples included in this submission, the transcripts are the raw Whisper large-v3 output — not manually corrected. They should be treated as high-quality automatic transcriptions, not verified ground truth. Downstream users who require clean transcriptions should budget for manual spot-checking on a random sample before training.

### Export

The real-data track exports to the same HuggingFace AudioFolder schema as the synthetic track, with an additional `source` field set to `"real"` (vs `"synthetic"` in the synthetic track). This makes it trivial to train on either track independently, or combine both and filter by source.

---

## Synthetic vs Real Data: A Practical Comparison

| Dimension | Synthetic (NAMAA-TTS) | Real (collected) |
|---|---|---|
| Scale | ~450 samples (~30 min) | ~90 hours |
| Transcription accuracy | Near-perfect (text is ground truth) | High, but requires verification |
| Speaker diversity | Single speaker | Many speakers |
| Acoustic diversity | Clean, studio-like | Phone, ambient, broadcast |
| Dialect consistency | Controlled by prompt | Varies — mostly Cairene but not guaranteed |
| Prosody naturalness | TTS prosody (slightly mechanical) | Fully natural |
| Code-switching | Controlled and deliberate | Spontaneous and varied |
| Production cost | Low (compute only) | High (collection, cleaning, review) |
| Risk of distribution mismatch | High — model may over-fit to TTS patterns | Low — matches real deployment conditions |

### Where synthetic data works

- **Bootstrapping**: when no real data exists, synthetic data gets a model off the ground.
- **Rare vocabulary**: numbers, named entities, technical terms can be injected into the text corpus and guaranteed to appear in audio.
- **Controlled dialect**: prompts enforce Cairene Egyptian Arabic consistently, whereas real audio may drift toward other varieties.
- **Transcription ground truth**: no transcription errors — the text is the source, so alignment is perfect.

### Where synthetic data fails

- **Prosody**: TTS audio has characteristic unnatural rhythm and stress on longer sentences. A model trained only on TTS data will be less robust to natural speech.
- **Acoustic mismatch**: all synthetic audio is clean. Real Egyptian Arabic is often recorded on phones, in noisy cafes, or over compressed VoIP. A model trained purely on clean TTS audio will degrade significantly in real deployment conditions.
- **Speaker diversity**: a single TTS voice means the model learns one speaker's phonetic realization. Egyptian Arabic has significant speaker-to-speaker variation that a single-voice dataset cannot cover.
- **Informal register**: even with careful prompting, LLM-generated text is somewhat more regular and grammatical than authentic spontaneous speech, which contains false starts, fillers (يعني، بقى، إيه), and truncated utterances.

### The right strategy: mix both

A model fine-tuned on synthetic data alone will perform well on clean, standard utterances and fail on noisy, fast, informal speech. A model fine-tuned on real data alone may be limited by the scale and transcription quality of what was collected. The optimal approach is:

1. Use real data as the primary training signal — it reflects actual deployment conditions.
2. Use synthetic data to augment coverage of rare vocabulary, specific topic domains, and controlled dialect examples.
3. Apply a mixing ratio (e.g. 80% real, 20% synthetic) and validate on a held-out real-speech test set, not a synthetic one.
4. Never evaluate a synthetic-data-trained model on synthetic test data — this will produce optimistic metrics that do not reflect real-world performance.

---

## Why Synthetic Data Can Mislead a Downstream STT Model

This is worth stating plainly because it is easy to overlook.

**The core risk** is that a model trained on synthetic data learns to recognise TTS audio patterns rather than human speech patterns. NAMAA-TTS, like all neural TTS systems, has characteristic artifacts: consistent phoneme durations, uniform background silence, and a specific prosodic contour. If the training set consists only of this model's output, the STT model will overfit to these characteristics and degrade significantly on real audio.

**The WER metric can be misleading.** If you evaluate the STT model on a synthetic test set (audio from the same TTS model), you will get low WER scores that do not reflect real-world performance. Always hold out a real-speech evaluation set.

**Dialect contamination is subtle.** If the TTS model has any exposure to MSA during training, it may produce prosody that is slightly more formal than authentic Cairene speech. The STT model will then learn associations between acoustic features and transcriptions that are slightly off for real Egyptian Arabic.

**Single-speaker over-fitting.** With one TTS voice, the model will implicitly learn speaker-specific features as if they were dialect features. It may generalise poorly to speakers whose voice characteristics differ from the TTS voice, which is every real speaker.

**How this pipeline addresses it:**

- The review stage uses a dialect-specific Whisper model to catch samples where TTS quality has degraded — these are the samples most likely to introduce noise.
- The real-data track provides acoustic diversity and naturalness that the synthetic track cannot.
- The `source` field in the export schema allows training scripts to apply differential weighting or to ablate each track independently.
- This README explicitly warns against evaluating on a synthetic test set.

---

## Observed Quality Issues

The following issues were observed during development and should be factored into downstream use of this dataset:

**Code-switched segments with embedded English.** NAMAA-TTS occasionally mispronounces English words embedded in otherwise Arabic sentences. For example, "شايل ال laptop معاك" may produce an awkward English phoneme sequence. The WER filter catches most of these, but borderline cases may pass through.

**Long-sentence prosody degradation.** Sentences above approximately 15 words produced unnatural stress patterns in some cases. Prompts were constrained to 8–15 words to mitigate this, but some generated sentences exceeded this range.

**Number pronunciation inconsistency.** Egyptian Arabic numbers can be pronounced in multiple valid ways (Eastern Arabic numerals vs. Western, masculine vs. feminine agreement). The TTS model does not always match the script. The WER filter accounts for this via Arabic normalization, but the transcription field in the exported dataset reflects the original written text, not the spoken form.

**LLM output drift.** In some batches, Gemma-4 produced sentences that were more formal (closer to MSA) than intended despite the system prompt. These are syntactically valid but stylistically wrong. They pass the character count filter but would benefit from a dialect classifier as an additional review signal.

**Single-speaker acoustic monotony.** All synthetic audio comes from one voice. This is the most significant limitation for downstream training and is addressed by the real-data track.

---

## Deliverables — Sample Dataset

This submission includes 60 sample (audio, transcript) pairs split across both tracks.

### 30 synthetic samples (`synthetic_audio/` + `synthetic_metadata.csv`)

Selected as the 30 best-scoring samples from the full synthetic run, ranked by combined WER + CER. These represent the upper end of synthesis quality — clean audio, accurate transcriptions, consistent Cairene dialect. Each file is named by prompt ID (e.g. `prompt_00012.wav`) and the matching transcript is in `30_sampled_metadata.csv`.

The transcript for each synthetic sample is the **original LLM-generated text** — the exact string that was fed into NAMAA-TTS. It is ground truth by construction: the TTS synthesized audio from it, so there is no transcription error on the text side. Any WER score in the metadata reflects how well the Whisper reviewer could recover that text from the audio, not a transcription error.

### 30 real samples (`real_samples/` + `real_samples_metadata.csv`)

Drawn from the 90-hour real-speech collection. Each segment is a natural utterance from a real speaker recorded in authentic conditions.

The transcript for each real sample is the **Whisper large-v3 automatic transcription** — not manually verified ground truth. This is an important distinction:

- For the synthetic samples: text → audio (text is the source of truth)
- For the real samples: audio → text (text is a model prediction, with an estimated 10–15% WER)

Some transcripts may contain errors — wrong words, skipped words, or MSA substitutions for colloquial Egyptian terms. The audio is authentic; the transcript is a best-effort automatic annotation. If you spot an obvious error in a transcript while listening, that is expected and reflects the transcription limitation described in the Trade-offs section, not a problem with the audio.



---



**Synthetic track**

- Scale is limited by compute time. At approximately 4 seconds per synthesis call on a single GPU, 450 samples takes around 30 minutes. Scaling to 10,000 samples is feasible but requires batching or parallel synthesis jobs.
- The pipeline depends on NAMAA-TTS remaining publicly available on HuggingFace. If the model is removed or updated, synthesis results will differ.
- Prompt-based text generation cannot guarantee phonetic coverage (e.g. all Arabic phonemes appearing in sufficient frequency). A phoneme-balanced generation strategy would require a constrained generation approach.

**Real-data track**

- Transcription quality depends on Whisper large-v3, which is not perfect on dialectal Arabic. Estimated word error rate on the automatic transcriptions is 10–15% before manual correction. The 30 real-data samples in this submission use raw Whisper output — not verified ground truth — and this should be factored in when assessing them.
- Copyright and licensing of source audio is a concern that must be addressed before this data is used in production or shared publicly.
- `inaSpeechSegmenter` occasionally misclassifies short speech segments as noise. A small percentage of valid speech is discarded.

**Review stage**

- Automated WER/CER scoring is a proxy for quality, not a direct measure. A synthesis failure that produces fluent but semantically wrong audio (e.g. wrong word pronunciation) will be scored as high-error and rejected — which is correct. But a synthesis failure that is subtle (e.g. slightly wrong vowel) may pass the threshold.
- The review Whisper model is fine-tuned on Egyptian Arabic but is still a relatively small model. Its own error rate contributes noise to the WER/CER scores.

---


## Running the Pipeline
 
### Requirements
 
Use the provided `requirements.txt` for exact versions:
 
```bash
pip install -r requirements.txt 
```
 

 
### Execution
 
Run the notebook cells in order. Each stage can be re-run independently — it will resume from where it left off.
 
```
Stage 1  →  egyptian_dataset.jsonl              (text prompts, checkpointed per category)
Stage 2  →  namaa/manifest.jsonl + namaa/*.wav  (synthesis, resumable)
Stage 3  →  reviewed/namaa_reviewed.jsonl       (WER/CER scores, resumable)
Stage 4  →  dataset/metadata.csv               (training-ready, 16 kHz mono WAVs)
            dataset/data/*.wav
            dataset/dataset_info.json
```
 

 
*Pipeline by Hamdi Mohamed — submitted as part of the Olimi AI case study.*
 
