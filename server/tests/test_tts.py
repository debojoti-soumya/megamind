from __future__ import annotations

import asyncio
import struct
from types import SimpleNamespace

from megamind.agent.tts import OpenAITTS, split_sentences
from megamind.config import Settings


def test_split_sentences_keeps_short_fragments_together():
    assert split_sentences("On the counter.") == ["On the counter."]
    assert split_sentences("Yes. It was on the kitchen counter next to the coffee machine.") == [
        "Yes. It was on the kitchen counter next to the coffee machine."
    ]
    two = "Your keys are on the kitchen counter. You left them there about twenty minutes ago."
    assert split_sentences(two) == [
        "Your keys are on the kitchen counter.",
        "You left them there about twenty minutes ago.",
    ]
    assert split_sentences("The total was $23.47 at checkout.") == ["The total was $23.47 at checkout."]


async def test_openai_tts_renders_sentences_concurrently_into_one_wav(tmp_path):
    active, peak, inputs = 0, 0, []

    async def create(*, model, voice, input, response_format):  # noqa: A002
        nonlocal active, peak
        assert response_format == "pcm"
        inputs.append(input)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return SimpleNamespace(content=b"\x01\x00" * 100)

    tts = OpenAITTS(Settings(_env_file=None, openai_api_key="k", audio_dir=tmp_path))
    tts.client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))
    url = await tts.synthesize("Your keys are on the kitchen counter. You left them there twenty minutes ago.", "a1")
    assert url == "/v1/audio/a1.wav" and peak == 2 and len(inputs) == 2
    wav = (tmp_path / "a1.wav").read_bytes()
    assert wav[:4] == b"RIFF" and struct.unpack("<I", wav[24:28])[0] == 24000
    assert struct.unpack("<I", wav[40:44])[0] == len(wav) - 44 == 200 * 2 + 7200
