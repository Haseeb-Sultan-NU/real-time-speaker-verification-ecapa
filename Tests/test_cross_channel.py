import sys
import os
import torch
import torchaudio
import soundfile as sf
import numpy as np

# 1. MONKEYPATCHES
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

def custom_audio_load(filepath, channels_first=True, **kwargs):
    data, samplerate = sf.read(filepath, dtype='float32')
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()
    if not channels_first:
        tensor = tensor.t()
    return tensor, samplerate

torchaudio.load = custom_audio_load

# 2. PATH FIX
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.verification.ecapa_engine import EcapaVerifier
from src.audio.simulator import TelephonySimulator

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    # --- THE FIX: Load your custom weights and new threshold! ---
    FINETUNED_MODEL_PATH = "models/best_urdu_triplet_ecapa.pth"
    OPTIMAL_THRESHOLD = 0.2393

    print("🚀 Booting up Multi-Condition Gatekeeper...")
    verifier = EcapaVerifier(finetuned_weights_path=FINETUNED_MODEL_PATH)
    simulator = TelephonySimulator(target_sr=8000)
    
    clean_file = "data/samples/Haris_Sample.wav"
    telephony_file = "data/samples/SPK_000_clean_1.wav"

    if not os.path.exists(clean_file) or not os.path.exists(telephony_file):
        print(f"[!] Error: Missing files. Ensure {clean_file} and {telephony_file} exist.")
        return

    print(f"\n[RUNNING MULTI-CONDITION VERIFICATION]")
    print(f"Target Voice (Enrollment): {clean_file}")
    print(f"Incoming Live Call (Verification): {telephony_file}")

    # --- STEP 1: Multi-Condition Enrollment ---
    wav, sr = torchaudio.load(clean_file)
    
    # Profile A: Clean
    emb_clean = verifier.extract_embedding(clean_file)
    
    # Profile B: Filtered (8kHz + Bandpass)
    wav_bp = simulator.apply_telephony_filter(simulator.resample(wav, sr))
    torchaudio.save("temp_bp.wav", wav_bp, 8000)
    emb_bp = verifier.extract_embedding("temp_bp.wav")
    
    # Profile C: Noisy (8kHz + Bandpass + White Noise)
    wav_noisy, sim_sr = simulator.process(wav, sr, snr_db=15)
    torchaudio.save("temp_noisy.wav", wav_noisy, sim_sr)
    emb_noisy = verifier.extract_embedding("temp_noisy.wav")

    # --- STEP 2: Extract Live Call Embedding ---
    emb_live = verifier.extract_embedding(telephony_file)

    # --- STEP 3: Compare ---
    scores = [
        cosine_sim(emb_clean, emb_live),
        cosine_sim(emb_bp, emb_live),
        cosine_sim(emb_noisy, emb_live)
    ]
    
    best_score = max(scores)
    is_match = best_score >= OPTIMAL_THRESHOLD

    print(f"\n[RESULTS]")
    print(f"Clean Similarity:    {scores[0]:.4f}")
    print(f"Filtered Similarity: {scores[1]:.4f}")
    print(f"Noisy Similarity:    {scores[2]:.4f}")
    print(f"----------------------------------------")
    print(f"🏆 FINAL MATCH SCORE: {best_score:.4f} (Required: {OPTIMAL_THRESHOLD})")
    print(f"🔒 DECISION: {'✅ ACCESS GRANTED' if is_match else '❌ ACCESS DENIED'}")

    # Cleanup temp files
    for temp_file in ["temp_bp.wav", "temp_noisy.wav"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    main()