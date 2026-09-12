import shutil
import subprocess
from pathlib import Path

BIN_PATH = Path(__file__).resolve().parent / "bin" / "espeak-ng"


def _find_espeak_ng() -> str:
    if BIN_PATH.exists():
        return str(BIN_PATH)
    return shutil.which("espeak-ng") or ""  # pragma: no cover


def text_to_speech(text: str, output_path: str | Path, voice: str = "es-mx") -> Path:
    binary = _find_espeak_ng()
    if not binary:
        raise FileNotFoundError("No se encontró espeak-ng")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [binary, "-w", str(output), "-v", voice, text],
        check=True,
        capture_output=True,
    )

    return output


def play_audio(audio_path: str | Path) -> None:
    aplay = shutil.which("aplay")
    if not aplay:
        print("Advertencia: aplay no está disponible; no se pudo reproducir el audio")
        return
    subprocess.run([aplay, "-q", str(audio_path)], check=False)