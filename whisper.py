#!/usr/bin/env python3

"""
whisper_realtime_recognizer.py is a ROS node that captures audio in real-time,
processes it using the Whisper model, and publishes the transcription.

Parameters:
  ~lm - (Not required for Whisper)
  ~dict - (Not required for Whisper)
  ~hmm - (Not required for Whisper)
  ~mic_name - name of the microphone device (optional)

Publications:
  ~output (std_msgs/String) - transcribed text

Services:
  ~start (std_srvs/Empty) - start transcription
  ~stop (std_srvs/Empty) - stop transcription
"""

import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

import pyaudio
import torch
import threading
import queue

from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperRealTimeRecognizer:
    def __init__(self):
        rospy.init_node("whisper_realtime_recognizer")

        # ROS Parameters
        self.mic_name = rospy.get_param("~mic_name", None)  # Optional: specify microphone name

        # Initialize ROS Publisher and Services
        self.pub = rospy.Publisher('~output', String, queue_size=10)
        rospy.Service("~start", Empty, self.start)
        rospy.Service("~stop", Empty, self.stop)

        # Initialize Whisper Model
        rospy.loginfo("Loading Whisper model...")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        self.language = "french"  # Set your desired language

        # Audio Parameters
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_size = 1024     # Number of frames per buffer
        self.format = pyaudio.paInt16
        self.channels = 1

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None

        # Threading
        self.audio_queue = queue.Queue()
        self.running = False
        self.thread = None

        rospy.on_shutdown(self.shutdown)

    def start(self, req):
        if not self.running:
            rospy.loginfo("Starting Whisper Real-Time Recognizer...")
            self.running = True
            self.thread = threading.Thread(target=self.run)
            self.thread.start()
        return EmptyResponse()

    def stop(self, req):
        if self.running:
            rospy.loginfo("Stopping Whisper Real-Time Recognizer...")
            self.running = False
            if self.thread is not None:
                self.thread.join()
        return EmptyResponse()

    def run(self):
        # Open audio stream
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.sample_rate,
                                  input=True,
                                  frames_per_buffer=self.chunk_size,
                                  input_device_index=self.get_device_index())

        rospy.loginfo("Audio stream opened.")

        buffer = []

        while self.running and not rospy.is_shutdown():
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                buffer.append(data)

                # Process every 5 seconds of audio
                if len(buffer) * self.chunk_size / self.sample_rate >= 5:
                    audio_data = b''.join(buffer)
                    buffer = []

                    # Convert byte data to numpy array
                    audio_np = self.byte_to_numpy(audio_data)

                    # Prepare input for Whisper
                    input_features = self.processor(audio_np, sampling_rate=self.sample_rate, return_tensors="pt").input_features

                    # Generate transcription
                    with torch.no_grad():
                        predicted_ids = self.model.generate(input_features, 
                                                            forced_decoder_ids=self.processor.get_decoder_prompt_ids(language=self.language, task="transcribe"))
                    
                    transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                    if transcription:
                        msg = String()
                        msg.data = transcription.lower()
                        rospy.loginfo(f"Transcription: {msg.data}")
                        self.pub.publish(msg)

            except Exception as e:
                rospy.logerr(f"Error in audio processing: {e}")
                self.running = False

        # Close stream
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        rospy.loginfo("Audio stream closed.")

    def byte_to_numpy(self, byte_data):
        import numpy as np
        audio_np = np.frombuffer(byte_data, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_np

    def get_device_index(self):
        if self.mic_name is None:
            return None  # Use default input device

        device_count = self.p.get_device_count()
        for i in range(device_count):
            device_info = self.p.get_device_info_by_index(i)
            if self.mic_name in device_info['name']:
                rospy.loginfo(f"Using microphone: {device_info['name']} (Index {i})")
                return i
        rospy.logwarn(f"Microphone '{self.mic_name}' not found. Using default device.")
        return None

    def shutdown(self):
        rospy.loginfo("Shutting down Whisper Real-Time Recognizer...")
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

if __name__ == "__main__":
    recognizer = WhisperRealTimeRecognizer()
    rospy.spin()
