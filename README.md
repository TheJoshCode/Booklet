CURRENTLY A WIP BUT WORKING GREAT ON UB 22.04

# MULTI-WORKER SETUP ALLOWS FOR 5X NOMRAL POCKET-TTS PERFORMANCE

# HOW RUN
```bash
git clone https://github.com/TheJoshCode/pocket-tts-multicore
uv sync
uv add ./pocket-tts-multicore
uvicorn server:app --reload
```

