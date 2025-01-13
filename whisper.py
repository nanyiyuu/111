#!/usr/bin/env python3

"""
recognizer.py is a wrapper for Whisper-based speech recognition.
  parameters:
    ~mic_name - set the pulsesrc device name for the microphone input.
                e.g. a Logitech G35 Headset has the following device name: alsa_input.usb-Logitech_Logitech_G35_Headset-00-Headset_1.analog-mono
                To list audio device info on your machine, in a terminal type: pacmd list-sources
  publications:
    ~output (std_msgs/String) - text output
  services:
    ~start (std_srvs/Empty) - start speech recognition
    ~stop (std_srvs/Empty) - stop speech recognition
"""

import rospy

from gi import pygtkcompat
import gi
gi.require_version('Gst', '1.0')

from gi.repository import GObject, Gst
Gst.init(None)
gst = Gst

pygtkcompat.enable()
pygtkcompat.enable_gtk(version='3.0')

import gtk

from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

import os
import numpy as np

# Import Whisper dependencies
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

class recognizer(object):
    """ Whisper-based speech recognizer. """

    def __init__(self):
        # Start node
        rospy.init_node("recognizer")

        self._device_name_param = "~mic_name"  # Find the name of your microphone by typing pacmd list-sources in the terminal

        # Configure mics with GStreamer launch config
        if rospy.has_param(self._device_name_param):
            self.device_name = rospy.get_param(self._device_name_param)
            self.device_index = self.pulse_index_from_name(self.device_name)
            self.launch_config = "pulsesrc device=" + str(self.device_index)
            rospy.loginfo("Using: pulsesrc device=%s name=%s", self.device_index, self.device_name)
        elif rospy.has_param('~source'):
            # common sources: 'alsasrc'
            self.launch_config = rospy.get_param('~source')
        else:
            self.launch_config = 'autoaudiosrc'

        rospy.loginfo("Launch config: %s", self.launch_config)

        # Configure GStreamer pipeline to output raw audio to appsink
        self.launch_config += (
            " ! audioconvert ! audioresample "
            "! audio/x-raw,format=S16LE,channels=1,rate=16000 "
            "! appsink name=asr emit-signals=true sync=false max-buffers=1 drop=true"
        )

        # Configure ROS settings
        self.started = False
        rospy.on_shutdown(self.shutdown)
        self.pub = rospy.Publisher('~output', String, queue_size=10)
        rospy.Service("~start", Empty, self.start)
        rospy.Service("~stop", Empty, self.stop)

        # Initialize Whisper model and processor
        rospy.loginfo("Loading Whisper model...")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')
            rospy.loginfo("Whisper model loaded on CUDA.")
        else:
            rospy.loginfo("Whisper model loaded on CPU.")

    def start_recognizer(self):
        rospy.loginfo("Starting recognizer... ")

        self.pipeline = gst.parse_launch(self.launch_config)
        self.appsink = self.pipeline.get_by_name('asr')
        self.appsink.connect('new-sample', self.on_new_sample)

        self.pipeline.set_state(gst.State.PLAYING)
        self.started = True
        rospy.loginfo("Recognizer started and pipeline is PLAYING.")

    def pulse_index_from_name(self, name):
        output = os.popen(
            "pacmd list-sources | grep -B 1 'name: <" + name + ">' | grep -o -P '(?<=index: )[0-9]*'"
        ).read().strip()

        if output.isdigit():
            return int(output)
        else:
            raise Exception("Error. Pulse index doesn't exist for name: " + name)

    def stop_recognizer(self):
        if self.started:
            self.pipeline.set_state(gst.State.NULL)
            self.pipeline = None
            self.appsink = None
            self.started = False
            rospy.loginfo("Recognizer stopped and pipeline is NULL.")

    def shutdown(self):
        """ Delete any remaining parameters so they don't affect next launch """
        for param in [self._device_name_param]:
            if rospy.has_param(param):
                rospy.delete_param(param)

        """ Shutdown the GTK thread. """
        gtk.main_quit()

    def start(self, req):
        if not self.started:
            self.start_recognizer()
            rospy.loginfo("Recognizer started via service.")
        else:
            rospy.loginfo("Recognizer is already running.")
        return EmptyResponse()

    def stop(self, req):
        if self.started:
            self.stop_recognizer()
            rospy.loginfo("Recognizer stopped via service.")
        else:
            rospy.loginfo("Recognizer is not running.")
        return EmptyResponse()

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        # Extract audio data from buffer
        array = self.buffer_to_array(buf, caps)
        if array is not None:
            transcription = self.transcribe(array)
            if transcription:
                msg = String()
                msg.data = transcription
                rospy.loginfo("Transcription: %s", msg.data)
                self.pub.publish(msg)
        return Gst.FlowReturn.OK

    def buffer_to_array(self, buf, caps):
        # Get buffer data
        result, map_info = buf.map(Gst.MapFlags.READ)
        if not result:
            rospy.logwarn("Failed to map buffer data.")
            return None

        # Extract audio format info
        structure = caps.get_structure(0)
        rate = structure.get_value('rate')
        channels = structure.get_value('channels')
        format = structure.get_value('format')

        # Assuming S16LE format
        if format != 'S16LE' or channels != 1 or rate != 16000:
            rospy.logwarn("Unexpected audio format: %s, channels: %d, rate: %d", format, channels, rate)
            buf.unmap(map_info)
            return None

        # Convert buffer to numpy array
        audio_data = np.frombuffer(map_info.data, dtype=np.int16).astype(np.float32) / 32768.0
        buf.unmap(map_info)
        return audio_data

    def transcribe(self, audio_array):
        try:
            # Prepare input for Whisper
            input_features = self.processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
            if torch.cuda.is_available():
                input_features = input_features.to('cuda')

            # Generate transcription
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription
        except Exception as e:
            rospy.logerr("Error during transcription: %s", str(e))
            return None

if __name__ == "__main__":
    start = recognizer()
    gtk.main()
