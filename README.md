# CRYSTAL MARK II

Successor to CRYSTAL MARK I
Original Repository: [https://github.com/vatdut8994/Crystal3.0](https://github.com/vatdut8994/Crystal3.0)
Superceding version: [CRYSTAL-R1](https://github.com/vatsaldutt/CRYSTAL-R1)

---

## Overview

CRYSTAL MARK II is the stage where CRYSTAL transitions from an embodied multimodal assistant into a locally autonomous, perception-grounded cognitive system. While MARK I proved that vision, audio, embodiment, and reasoning could coexist in a single always-on agent, MARK II aggressively removes external dependencies and re-centers the system around live environment understanding and self-contained reasoning.

This version emphasizes *locality*, *situated awareness*, and *continuous world-state construction*. All perception, transcription, reasoning, and grounding run on-device, with no reliance on paid APIs or cloud inference. MARK II is less about reacting to user input and more about maintaining an ongoing internal model of the world and querying it intelligently.

---

## System Architecture (Always-On Cognitive Loop)

CRYSTAL MARK II operates as a multi-threaded, continuously running system where perception, live data, and reasoning never halt. Rather than discrete request–response cycles, the system maintains a rolling internal context that is constantly updated and injected into the reasoning engine.

At a high level, the system consists of:

* a real-time visual perception pipeline
* a local audio transcription and speaker identification pipeline
* autonomous live data ingestion (time, weather)
* a self-directed web grounding engine
* a locally trained LLM acting as the reasoning core
* a visualization loop for live inspection and debugging

All subsystems synchronize through shared state and lightweight file-based IPC, enabling concurrent execution without blocking the main reasoning loop.

---

## Visual Perception & Environment Understanding (CircumSpect)

A central advancement in MARK II is dense visual grounding via *CircumSpect*, CRYSTAL’s dedicated environment perception module. Instead of treating vision as a peripheral feature, MARK II treats visual understanding as first-class cognitive input.

CircumSpect continuously processes live camera input and converts it into structured, language-ready descriptions of the environment. This includes dense captioning and semantic interpretation of what CRYSTAL is seeing, not just object detection. The output is a textual world description that captures context, spatial cues, visible text, faces, and user attention signals, alongside an annotated video stream for real-time visualization.

Key capabilities grouped together:

* Dense image captioning for holistic scene understanding
* OCR for extracting readable text from the environment
* Face recognition and identity association
* Gaze detection to infer attention and engagement
* Scene annotation and visual overlays

These outputs are streamed directly into the reasoning context, allowing CRYSTAL to answer questions like *“what’s in front of me?”* or *“what am I looking at right now?”* using its own perception rather than user-provided descriptions.

---

## Audio Perception, Transcription, and Speaker Identity (SoundScribe)

CRYSTAL MARK II uses a fully local, Whisper-based speech transcription pipeline for continuous listening and real-time transcription. This replaces earlier cloud-based approaches and ensures that all audio perception remains on-device.

SoundScribe handles:

* Continuous live speech transcription (no push-to-talk)
* Local Whisper inference
* Speaker identification and attribution
* User identity resolution via recorded voice signatures

Recognized speech is streamed directly into the reasoning loop while preserving speaker identity, enabling CRYSTAL to reason not just about *what* was said, but *who* said it. This preserves identity-aware interaction without any external STT services.

---

## Live World-State Construction

Rather than responding to isolated inputs, MARK II continuously constructs a live situational state that represents CRYSTAL’s current understanding of its surroundings.

This world-state includes:

* Current time and date
* Real-time weather conditions (scraped directly from search results)
* Dense visual environment descriptions from CircumSpect
* Most recent recognized speech input

This information is continuously refreshed, written to disk, and injected into the reasoning prompt, ensuring that every response is grounded in the current physical and informational context.

---

## Autonomous Web Grounding & Information Retrieval

To avoid reliance on static knowledge or external APIs, MARK II introduces a self-directed web grounding system capable of live information retrieval and parsing.

When a query requires external knowledge, CRYSTAL autonomously:

* Generates search queries
* Performs Google searches
* Scrapes pages using headless Chrome (Selenium)
* Parses HTML via BeautifulSoup
* Extracts clean article text using Trafilatura

The system includes multiple fallback strategies for definitions, Wikipedia summaries, translations, and even YouTube transcript extraction when relevant. Rather than blindly returning scraped text, the system selects and condenses information before injecting it into the reasoning context.

This allows CRYSTAL to reason over current, real-world information, not just what was present during training.

---

## Local LLM & Custom Training Pipeline

At the core of CRYSTAL MARK II is a locally trained large language model, designed to operate without dependence on proprietary inference APIs. This model serves as the system’s reasoning engine, synthesizing perception, memory, and web-grounded data into coherent responses.

The LLM was trained using custom pipelines that:

* Aggregate large-scale text data from the web
* Normalize and clean conversational structures
* Incorporate environment-grounded descriptions
* Continuously append validated interactions to a growing corpus

This approach allowed CRYSTAL to reason over its own perceptions and experiences, not just abstract text. The model is invoked with dynamically constructed prompts that include live world-state data, perception outputs, user identity, and tunable inference parameters.

---

## Real-Time Reasoning & Streaming Output

CRYSTAL MARK II streams model outputs token-by-token, writing responses live to disk. This enables:

* Real-time UI display
* Immediate TTS consumption
* External system hooks
* Continuous feedback during inference

Rather than waiting for a complete response, CRYSTAL can react and adapt mid-generation, reinforcing the system’s agent-like behavior.

---

## What MARK II Actually Changed

MARK II is not a cosmetic upgrade over MARK I. It fundamentally reframes CRYSTAL as:

* a **world-modeling system**, not just an assistant
* a **local-first autonomous agent**, not an API client
* a **perception-grounded reasoner**, not a text-only LLM wrapper

This version laid the architectural groundwork for CRYSTAL-R1, where explicit world models, cleaner agent loops, and safer autonomy become primary design goals.

---

## Author

**Vatsal Dutt**
Creator of CRYSTAL
