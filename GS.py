#!/usr/bin/env python3

"""
whisper_realtime_recognizer_gst.py is a ROS node that captures real-time audio using GStreamer,
processes it with the Whisper model, and publishes the transcription results.

Parameters:
  ~mic_name - Name of the microphone device (optional)

Publications:
  ~output (std_msgs/String) - Transcribed text

Services:
  ~start (std_srvs/Empty) - Start transcription
  ~stop (std_srvs/Empty) - Stop transcription
"""

import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import threading
import queue

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

Gst.init(None)

import sys

class WhisperRealTimeRecognizerGst:
    def __init__(self):
        rospy.init_node("whisper_realtime_recognizer_gst")

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

        # GStreamer Pipeline Configuration
        self.launch_config = self.build_gstreamer_pipeline()
        rospy.loginfo(f"GStreamer pipeline configuration: {self.launch_config}")

        self.pipeline = Gst.parse_launch(self.launch_config)
        self.asr = self.pipeline.get_by_name('asr')
        self.asr.set_property('dsratio', 1)

        # Configure Whisper model properties
        # Whisper model does not require lm, dict, hmm properties

        # Set up bus message handling
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus_id = self.bus.connect('message::element', self.element_message)

        # Control variables
        self.running = False
        self.thread = None

        # Audio buffer
        self.audio_queue = queue.Queue()

        rospy.on_shutdown(self.shutdown)

    def build_gstreamer_pipeline(self):
        """
        Build the GStreamer pipeline string to capture audio and push it to the application.
        """
        if self.mic_name:
            # Find the pulse audio source based on mic_name
            launch_config = f'pulsesrc device="{self.mic_name}" ! ' \
                            f'audioconvert ! audioresample ! ' \
                            f'capsfilter caps="audio/x-raw, format=S16LE, channels=1, rate=16000" ! ' \
                            f'appsrc name=appsrc ! queue ! wavenc ! appsink name=appsink'
        else:
            # Use the default audio source
            launch_config = f'autoaudiosrc ! ' \
                            f'audioconvert ! audioresample ! ' \
                            f'capsfilter caps="audio/x-raw, format=S16LE, channels=1, rate=16000" ! ' \
                            f'appsrc name=appsrc ! queue ! wavenc ! appsink name=appsink'

        return launch_config

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
        """
        Start the GStreamer pipeline and process audio data.
        """
        # Get the appsink element
        self.appsink = self.pipeline.get_by_name('appsink')
        self.appsink.set_property('emit-signals', False)
        self.appsink.set_property('sync', False)

        # Start the pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        rospy.loginfo("GStreamer pipeline started, capturing audio...")

        # Audio buffer
        buffer = b''

        while self.running and not rospy.is_shutdown():
            # Pull sample from appsink
            sample = self.appsink.emit('pull-sample')
            if sample:
                buf = sample.get_buffer()
                data = buf.extract_dup(0, buf.get_size())
                buffer += data

                # Minimum audio length required by Whisper (e.g., 5 seconds)
                if len(buffer) >= 16000 * 2 * 5:  # 16000 Hz * 2 bytes * 5 seconds
                    audio_np = self.buffer_to_numpy(buffer)
                    buffer = b''

                    # Prepare Whisper input
                    input_features = self.processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features

                    # Generate transcription
                    with torch.no_grad():
                        predicted_ids = self.model.generate(
                            input_features,
                            forced_decoder_ids=self.processor.get_decoder_prompt_ids(language=self.language, task="transcribe")
                        )

                    transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                    if transcription:
                        msg = String()
                        msg.data = transcription.lower()
                        rospy.loginfo(f"Transcription: {msg.data}")
                        self.pub.publish(msg)

        # Stop the pipeline
        self.pipeline.set_state(Gst.State.NULL)
        rospy.loginfo("GStreamer pipeline stopped.")

    def buffer_to_numpy(self, buffer):
        """
        Convert audio buffer byte data to a NumPy array.
        """
        import numpy as np
        audio_np = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_np

    def element_message(self, bus, msg):
        """
        Handle GStreamer bus messages (optional, useful for debugging).
        """
        if msg.type == Gst.MessageType.ELEMENT:
            struct = msg.get_structure()
            if struct:
                rospy.logdebug(f"GStreamer message: {struct.to_string()}")

    def shutdown(self):
        """
        Clean up resources.
        """
        rospy.loginfo("Shutting down Whisper Real-Time Recognizer node...")
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.pipeline.set_state(Gst.State.NULL)
        rospy.loginfo("Node has been shut down.")

if __name__ == "__main__":
    recognizer = WhisperRealTimeRecognizerGst()
    rospy.spin()
