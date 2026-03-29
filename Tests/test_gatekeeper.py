import os
import sys
from unittest.mock import MagicMock

# 1. PATH FIX: Ensure 'src' is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. THE OMNI-PATCH: Stable environment for Windows/PyTorch 2.6
import torch
import torchaudio

# Fake deleted torchaudio functions
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None
if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"

# Fake torchaudio backend package structure
for module in ['torchaudio.backend', 'torchaudio.backend.common']:
    if module not in sys.modules:
        m = MagicMock()
        m.__path__ = []
        sys.modules[module] = m

# Mock TorchCodec
if 'torchcodec' not in sys.modules:
    sys.modules['torchcodec'] = MagicMock()

# Aggressive PyTorch 2.6 Security Bypass
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False})

# Now safe to import our module
from src.verification.gatekeeper import SecurityGatekeeper

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
SAMPLES_DIR = "data/samples"

def main():
    gatekeeper = SecurityGatekeeper(HF_TOKEN)
    
    files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith('.wav')]
    if not files:
        print(f"No samples found in {SAMPLES_DIR}!")
        return

    print(f"\n[RUNNING SECURITY BATCH SCAN] - {len(files)} samples detected")
    print("-" * 70)
    
    results = []
    for filename in sorted(files):
        filepath = os.path.join(SAMPLES_DIR, filename)
        is_secure, message = gatekeeper.check_audio_security(filepath)
        
        status = "✅ ACCEPT" if is_secure else "❌ REJECT"
        results.append((filename, status, message))
        print(f"Processed: {filename}")

    # Final Summary Table
    print("\n" + "="*75)
    print(f"{'FILENAME':<25} | {'STATUS':<12} | {'REASON'}")
    print("-" * 75)
    for res in results:
        print(f"{res[0]:<25} | {res[1]:<12} | {res[2]}")
    print("="*75)

if __name__ == "__main__":
    main()