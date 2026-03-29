# Real-Time Speaker Verification System (ECAPA-TDNN)

![Status](https://img.shields.io/badge/status-active-success)
![Domain](https://img.shields.io/badge/domain-audio%20biometrics-blue)
![Model](https://img.shields.io/badge/model-ECAPA--TDNN-informational)
![Framework](https://img.shields.io/badge/framework-PyTorch-red)

Production-oriented audio biometric authentication system designed for **low-latency speaker verification**, **liveness detection**, and **robust performance across real-world audio conditions**.

---

## System Summary

| Component         | Details                                        |
| ----------------- | ---------------------------------------------- |
| Task              | Speaker Verification                           |
| Model             | ECAPA-TDNN                                     |
| Embedding Size    | 192-dim                                        |
| Input Features    | Mel-Spectrogram (64 FBanks, 25ms window, 8kHz) |
| Scoring           | Cosine Similarity                              |
| Normalization     | z-norm / s-norm                                |
| Deployment Target | Real-time (Web / Mobile)                       |

---

## Architecture Overview

```
Audio Input
   ↓
Preprocessing (Resampling, Framing, Mel-Spectrogram)
   ↓
Embedding Extraction (ECAPA-TDNN)
   ↓
Similarity Scoring (Cosine / PLDA)
   ↓
Score Normalization (z-norm / s-norm)
   ↓
Decision Thresholding
   ↓
Liveness Verification Layer
```

---

## Core System Components

### 1. Data Pipeline

* Audio resampled to 8 kHz
* Feature extraction via Mel-spectrograms
* Structured preprocessing for consistent inference

---

### 2. Embedding Model

* ECAPA-TDNN architecture
* 192-dimensional speaker embeddings
* Fine-tuned for:

  * Channel variability
  * Noise robustness
  * Speaker discrimination

---

### 3. Scoring & Calibration

* Cosine similarity-based scoring
* Score normalization:

  * z-norm
  * s-norm
* Stabilizes performance across devices and environments

---

### 4. Liveness Detection

* Dynamic challenge-response prompts
* Prevents replay and pre-recorded spoofing attacks

---

### 5. Inference Layer

* Designed for real-time execution
* Compatible with API-based deployment (FastAPI)
* Supports low-latency verification workflows

---

## Performance Snapshot

| Metric                 | Value                |
| ---------------------- | -------------------- |
| Equal Error Rate (EER) | ~2–5%                |
| Inference Latency      | Sub-second           |
| Robustness             | Cross-channel stable |

---

## Experimental Focus

* Cross-channel speaker verification (telephony vs microphone)
* Impact of normalization techniques on EER
* FAR vs FRR trade-off tuning
* Embedding stability under noisy conditions

---

## Tech Stack

**Modeling**

* PyTorch
* ECAPA-TDNN
* pyannote.audio

**Data Processing**

* Librosa
* NumPy
* Pandas

**Deployment**

* FastAPI
* Docker

---

## Repository Structure

```
├── data/
├── models/
├── inference/
├── evaluation/
├── utils/
├── scripts/
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/Haseeb-Sultan-NU/real-time-speaker-verification-ecapa.git
cd real-time-speaker-verification-ecapa
pip install -r requirements.txt
python inference/run_inference.py --input sample.wav
```

---

## Roadmap

* Real-time streaming inference
* Edge/mobile deployment optimization
* Advanced anti-spoofing (deepfake detection)
* Batch inference optimization (GPU)

---

## Contact

* GitHub: https://github.com/Haseeb-Sultan-NU
* LinkedIn: [Add link]
