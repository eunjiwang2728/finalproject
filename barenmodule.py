import pyaudio
import wave
import speech_recognition as sr
from bareunpy import Tagger
import threading
import time
import os


API_KEY = "koba-7Y2RALI-3KEECJQ-XDOEAJY-S3PXVUY" # Replace this with your own API key, which can be obtained from bareun.ai
tagger = Tagger(API_KEY, 'localhost')

class EnhancedVoiceAnalysis:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.stream = None
        self.p = None
        self.recognizer = sr.Recognizer()
        self.callback = None

    def start_recording(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=44100,
                                  input=True,
                                  frames_per_buffer=1024)
        self.recording = True
        self.record_thread = threading.Thread(target=self.record)
        self.record_thread.start()

    def stop_recording(self):
        self.recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        if self.callback:
            self.callback('Recording stopped')

    def record(self):
        while self.recording:
            data = self.stream.read(44100 * 5, exception_on_overflow=False)  # Record for 5 seconds
            self.frames.append(data)
            if len(self.frames) > 0:
                self.process_data(b''.join(self.frames))
                self.frames = []

    def process_data(self, data):
        try:
            temp_filename = 'temp_audio.wav'
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(data)
            wf.close()
            
            with sr.AudioFile(temp_filename) as source:
                audio = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio, language="ko-KR")
            if self.callback:
                self.callback(f'Recognized text: {text}')
            self.analyze_morpheme(text)
            os.remove(temp_filename)
        except sr.UnknownValueError:
            if self.callback:
                self.callback(" ")
        except sr.RequestError as e:
            if self.callback:
                self.callback(f"Recognition error: {e}")
        except Exception as e:
            if self.callback:
                self.callback(f"Error: {e}")

    def analyze_morpheme(self, text):

        with wave.open('temp_audio.wav', 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
        
        analysis_result = tagger.pos(text)
        # print(analysis_result) # Uncomment this line to see the result of morpheme analysis
        ic_detected = any(tag == 'IC' for _, tag in analysis_result)
        words_count = len(text)  # Directly count all characters in the recognized text
        words_per_second = words_count / duration

        feedback = ""
        if ic_detected:
            feedback += "필러 감지!\n"
        if words_per_second < 4.7:
            feedback += "빠르게\n"
        elif words_per_second > 6.7:
            feedback += "천천히"

        if self.callback:
            self.callback(f'Morpheme Analysis: \n {feedback}')

    def set_callback(self, callback):
        self.callback = callback