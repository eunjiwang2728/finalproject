import pyaudio
import wave
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import time

inference_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="iic/emotion2vec_plus_seed"
)

# audio data processing
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 2

def recognize_emotion(audio_data):
    # save audio data to a wav file
    wf = wave.open('temp.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(audio_data)
    wf.close()

    # use emotion2vec model to recognize emotion
    rec_result = inference_pipeline('temp.wav', granularity="frame", extract_embedding=False)
    return rec_result

def get_highest_emotion(rec_result):
    labels = rec_result[0]['labels']
    scores = rec_result[0]['scores']
    highest_index = np.argmax(scores)
    return labels[highest_index]

def continuous_record_and_recognize(callback, audio_data_queue):
    while True:
        audio_data = audio_data_queue.get()
        if audio_data is None:
            break
        
        rec_result = recognize_emotion(audio_data)
        emotion = get_highest_emotion(rec_result)

        if callback:
            callback(emotion)

