"""Optional local synthetic narration. This is not a student's recorded voice."""

from pathlib import Path
import hashlib
import json

MODEL_SHA = "ae315a79b623f244700e4afb9246c46a26066782e049ba174bf3ba433970ee9c"
VOICES_SHA = "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"


class Narrator:
    def __init__(self, root):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        root = Path(root)
        model = root / "cache/tts/kokoro-v1.0.int8.onnx"
        voices = root / "cache/tts/voices-v1.0.bin"
        for path, expected in [(model, MODEL_SHA), (voices, VOICES_SHA)]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("TTS model checksum mismatch: " + str(path))
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.model = Kokoro.from_session(session, str(voices))
        self.spec = {
            "engine": "kokoro-onnx 0.6.1",
            "model_sha256": MODEL_SHA,
            "voices_sha256": VOICES_SHA,
            "voice": "af_sarah",
            "speed": 1.0,
            "language": "en-us",
            "synthetic_voice": True,
            "source": "https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1",
            "model_license": "Apache-2.0",
            "wrapper_license": "MIT",
        }
        (root / "walkthrough/narration_metadata.json").write_text(
            json.dumps(self.spec, indent=2) + "\n"
        )

    def create(self, text, output):
        from scipy.io import wavfile
        import numpy as np

        output = Path(output)
        key = hashlib.sha256(
            json.dumps({"text": text, **self.spec}, sort_keys=True).encode()
        ).hexdigest()
        marker = output.with_suffix(".sha256")
        if output.exists() and marker.exists() and marker.read_text() == key:
            return
        samples, rate = self.model.create(
            text, voice=self.spec["voice"], speed=self.spec["speed"], lang="en-us"
        )
        if len(samples) < rate or not np.isfinite(samples).all():
            raise ValueError("Invalid synthesized audio")
        wavfile.write(output, rate, (np.clip(samples, -1, 1) * 32767).astype(np.int16))
        marker.write_text(key)
