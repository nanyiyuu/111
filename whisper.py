#!/usr/bin/env python3

"""
recognizer.py is a wrapper for Google Speech-to-Text API.
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

# Import Google Cloud Speech dependencies
from google.cloud import speech
import queue
import threading

class recognizer(object):
    """ Google Speech-to-Text based speech recognizer. """

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

        # Initialize Google Speech client
        rospy.loginfo("Initializing Google Speech-to-Text client...")
        self.client = speech.SpeechClient()

        # Prepare audio buffer and threading
        self.audio_queue = queue.Queue()
        self.streaming_thread = None
        self.streaming = False

    def start_recognizer(self):
        rospy.loginfo("Starting recognizer... ")

        self.pipeline = gst.parse_launch(self.launch_config)
        self.appsink = self.pipeline.get_by_name('asr')
        self.appsink.connect('new-sample', self.on_new_sample)

        self.pipeline.set_state(gst.State.PLAYING)
        self.started = True
        rospy.loginfo("Recognizer started and pipeline is PLAYING.")

        # Start the streaming recognition in a separate thread
        self.streaming = True
        self.streaming_thread = threading.Thread(target=self.stream_recognition)
        self.streaming_thread.start()

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
            self.streaming = False
            if self.streaming_thread is not None:
                self.streaming_thread.join()
                self.streaming_thread = None

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
            self.audio_queue.put(array)
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

    def stream_recognition(self):
        """ Stream audio data to Google Speech-to-Text and handle responses """
        def generator():
            while self.streaming:
                try:
                    audio_chunk = self.audio_queue.get(timeout=1)
                    # Convert float32 back to int16
                    int_audio = (audio_chunk * 32768).astype(np.int16).tobytes()
                    yield speech.StreamingRecognizeRequest(audio_content=int_audio)
                except queue.Empty:
                    continue

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="zh-CN",  # 根据需要修改语言代码
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )

        try:
            requests = generator()
            responses = self.client.streaming_recognize(streaming_config, requests)

            for response in responses:
                if not self.streaming:
                    break
                for result in response.results:
                    if result.is_final:
                        transcription = result.alternatives[0].transcript.strip()
                        if transcription:
                            msg = String()
                            msg.data = transcription
                            rospy.loginfo("Transcription: %s", msg.data)
                            self.pub.publish(msg)
        except Exception as e:
            rospy.logerr("Error during Google Speech-to-Text streaming: %s", str(e))

if __name__ == "__main__":
    start = recognizer()
    gtk.main()
