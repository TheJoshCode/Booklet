gcc booklet.c -o booklet -lzmq -ljson-c -lpthread

pip install zmq numpy pydub numpy onnxruntime sentencepiece scipy soundfile

./booklet --text_file voicesnstuff/kjv_clean.txt --speaker_wav voicesnstuff/mf.wav --workers 8 --max_chunk_size 500 --output audiobook.wav --precision int8 --temperature 0.7 --lsd_steps 5

./booklet --text_file voicesnstuff/kjv_clean.txt --speaker_wav voicesnstuff/frieren.wav --output frierenkjv.mp3 --workers 2 --max_chunk_size 500 --precision int8 --temperature 0.7 --lsd_steps 5
