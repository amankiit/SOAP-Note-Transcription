#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from shutil import which

from ollama import generate
import whisper


SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Convert the provided medical transcript into a "
    "structured SOAP note with exactly these top-level keys: Subjective, Objective, Assessment, Plan. "
    "Return valid JSON only, with each key mapped to a concise but clinically useful string. "
    "Do not include markdown, commentary, code fences, or extra keys. "
    "Style requirements: "
    "Subjective should summarize patient-reported symptoms/history (include onset, severity, modifiers, relevant negatives if stated). "
    "Objective should include only observed exam findings, measurements, and test/imaging results. "
    "Assessment should be a one-line clinical impression/diagnosis. "
    "Plan should be concrete treatment and follow-up steps with timelines/contingencies when stated. "
    "Use compact medical phrasing and avoid conversational wording."
)


DEFAULT_PROMPT_FILE = Path("prompts/soap_prompt.txt")


def load_prompt_template(prompt_file: Path) -> str:
    if not prompt_file.exists():
        raise RuntimeError(f"Prompt file not found: {prompt_file}")
    content = prompt_file.read_text(encoding="utf-8").strip()
    if "{transcript}" not in content:
        raise RuntimeError("Prompt file must include '{transcript}' placeholder.")
    return content



def check_ffmpeg() -> None:
    if not which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is not installed or not available in PATH. Install it first, e.g. on macOS: brew install ffmpeg"
        )



def transcribe_audio(audio_path: Path, model_name: str) -> str:
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    text = (result or {}).get("text", "").strip()
    if not text:
        raise RuntimeError("Local Whisper transcription returned no text.")
    return text



def call_ollama(transcript: str, model: str, prompt_template: str) -> str:
    prompt = prompt_template.format(transcript=transcript)
    response = generate(
        model=model,
        prompt=prompt,
        system=SYSTEM_PROMPT,
        format="json",
    )
    content = response["response"].strip()
    if not content:
        raise RuntimeError("Ollama returned an empty response.")
    return content



def parse_soap_json(raw_text: str) -> dict:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise RuntimeError("Could not parse SOAP note JSON from Llama output.")
        parsed = json.loads(match.group(0))

    expected_keys = ["Subjective", "Objective", "Assessment", "Plan"]
    soap = {}
    for key in expected_keys:
        value = parsed.get(key, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        soap[key] = str(value).strip()
    return soap



def print_output(transcript: str, soap_note: dict) -> None:
    print("\n=== Transcript ===\n")
    print(transcript)
    print("\n=== SOAP Note ===\n")
    for section in ["Subjective", "Objective", "Assessment", "Plan"]:
        print(f"{section}:\n{soap_note.get(section, '')}\n")



def save_output(audio_file: Path, transcript: str, soap_note: dict, output_file: Path) -> None:
    result = {
        "audio_file": str(audio_file),
        "transcript": transcript,
        "soap_note": soap_note,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe medical dictation audio locally and generate a SOAP note.")
    parser.add_argument("audio_file", nargs="?", default="data/audio.mp3", help="Path to input audio file.")
    parser.add_argument("--output", default="output.json", help="Path to output JSON file.")
    parser.add_argument(
        "--whisper-model",
        default=os.getenv("WHISPER_MODEL", "base"),
        help="Local Whisper model size: tiny, base, small, medium, large, turbo.",
    )
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), help="Ollama model name.")
    parser.add_argument(
        "--prompt-file",
        default=os.getenv("SOAP_PROMPT_FILE", str(DEFAULT_PROMPT_FILE)),
        help="Path to SOAP prompt template file (must include {transcript}).",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio_file)
    output_path = Path(args.output)
    prompt_file = Path(args.prompt_file)

    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        return 1

    try:
        check_ffmpeg()
        transcript = transcribe_audio(audio_path, args.whisper_model)
        prompt_template = load_prompt_template(prompt_file)
        raw_soap = call_ollama(transcript, args.ollama_model, prompt_template)
        soap_note = parse_soap_json(raw_soap)
        print_output(transcript, soap_note)
        save_output(audio_path, transcript, soap_note, output_path)
        print(f"Saved JSON output to: {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
