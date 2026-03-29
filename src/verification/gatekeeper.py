# ============================================================================
# 🚨 MUST BE SET BEFORE IMPORTING TORCH
# ============================================================================
import os
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"

import sys
import torch
import tempfile
from unittest.mock import MagicMock

# ============================================================================
# 🚨 TORCHAUDIO GLOBAL MONKEYPATCH (Windows / Backend Fix)
# ============================================================================
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None

if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"

if "torchaudio.backend" not in sys.modules:
    mock_backend = MagicMock()
    mock_backend.__path__ = []
    sys.modules["torchaudio.backend"] = mock_backend

if "torchaudio.backend.common" not in sys.modules:
    sys.modules["torchaudio.backend.common"] = MagicMock()

# ============================================================================
# 🚨 PyTorch 2.6 SAFE GLOBALS FIX FOR PYANNOTE
# ============================================================================
try:
    from pyannote.audio.core.task import Specifications
    from pyannote.audio.core.model import Model
    from pyannote.audio.core.pipeline import Pipeline as PyannotePipeline

    torch.serialization.add_safe_globals([
        torch.torch_version.TorchVersion,
        Specifications,
        Model,
        PyannotePipeline
    ])
except Exception:
    pass
# ============================================================================

from pyannote.audio import Pipeline
import soundfile as sf


class SecurityGatekeeper:
    def __init__(self, hf_token):
        print("Initializing Security Gatekeeper (pyannote.audio)...")

        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
        except Exception as e:
            print(f"[!] Failed to load pipeline. Error: {e}")
            self.pipeline = None

    def check_audio_security(self, audio_path):

        if not self.pipeline:
            return False, "Gatekeeper offline."

        if not os.path.exists(audio_path):
            return False, "Audio file not found."

        temp_path = None

        try:
            # --------------------------------------------------------------
            # Load audio
            # --------------------------------------------------------------
            data, samplerate = sf.read(audio_path, dtype="float32")
            waveform = torch.from_numpy(data)

            # --------------------------------------------------------------
            # Ensure mono format (1, num_samples)
            # --------------------------------------------------------------
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            elif waveform.ndim == 2:
                # Convert stereo to mono
                if waveform.shape[1] == 2:
                    waveform = torch.mean(waveform, dim=1, keepdim=True).t()
                else:
                    waveform = waveform.t()

            # --------------------------------------------------------------
            # Pad to minimum 4 seconds (pyannote stability requirement)
            # --------------------------------------------------------------
            duration_sec = waveform.shape[1] / samplerate

            if duration_sec < 4.0:
                pad_frames = int((4.0 - duration_sec) * samplerate)
                waveform = torch.nn.functional.pad(waveform, (0, pad_frames))

            # --------------------------------------------------------------
            # Save to temporary WAV (pyannote expects file path)
            # --------------------------------------------------------------
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_path = temp_wav.name

            sf.write(temp_path, waveform.squeeze().numpy(), samplerate)

            # --------------------------------------------------------------
            # Run diarization
            # --------------------------------------------------------------
            diarization = self.pipeline(temp_path)

        except Exception as e:
            return False, f"Failed to process audio: {e}"

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

        # --------------------------------------------------------------
        # Extract unique speakers
        # --------------------------------------------------------------
        speakers = set()

        try:
            if hasattr(diarization, "labels"):
                speakers = set(diarization.labels())
            else:
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speakers.add(speaker)
        except Exception:
            try:
                for track in diarization.tracks():
                    speakers.add(track.label)
            except Exception:
                pass

        num_speakers = len(speakers)

        # --------------------------------------------------------------
        # Decision Logic
        # --------------------------------------------------------------
        if num_speakers == 0:
            return False, "REJECT: No human speech detected."

        elif num_speakers > 1:
            return False, f"REJECT: Multiple speakers ({num_speakers}) detected. Potential coercion."

        else:
            return True, "ACCEPT: Single, isolated speaker verified."


if __name__ == "__main__":
    print("Security Gatekeeper module ready.")