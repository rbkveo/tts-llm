import asyncio
import os
from silero_vad import load_silero_vad
import edge_tts

async def test_tts():
    print("Testing TTS...")
    text = "This is a test of the text to speech system."
    communicate = edge_tts.Communicate(text, "en-US-AndrewNeural")
    await communicate.save("test_tts_output.mp3")
    if os.path.exists("test_tts_output.mp3"):
        print("TTS Success: test_tts_output.mp3 created.")
        os.remove("test_tts_output.mp3")
    else:
        print("TTS Failed.")

def test_vad_loading():
    print("Testing VAD model loading...")
    try:
        load_silero_vad()
        print("VAD Model loaded successfully.")
    except Exception as e:
        print(f"VAD Loading Failed: {e}")

if __name__ == "__main__":
    test_vad_loading()
    asyncio.run(test_tts())
