#!/usr/bin/env python3 

"""
recognizer.py 是基于 Whisper 的语音识别封装。
  参数:
    ~mic_name - 设置用于麦克风输入的 pulsesrc 设备名称。
               例如，一个 Logitech G35 耳机的设备名称为: alsa_input.usb-Logitech_Logitech_G35_Headset-00-Headset_1.analog-mono
               要在你的机器上列出音频设备信息，请在终端中输入: pacmd list-sources
  发布:
    ~output (std_msgs/String) - 文本输出
  服务:
    ~start (std_srvs/Empty) - 开始语音识别
    ~stop (std_srvs/Empty) - 停止语音识别
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

# 导入 Whisper 依赖
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

class Recognizer(object):
    """ 基于 Whisper 的语音识别器。 """

    def __init__(self):
        # 启动节点
        rospy.init_node("recognizer")
        self._device_name_param = "~mic_name"  # 通过 pacmd list-sources 查找你的麦克风名称
        # 使用 GStreamer 启动配置配置麦克风
        if rospy.has_param(self._device_name_param):
            self.device_name = rospy.get_param(self._device_name_param)
            self.device_index = self.pulse_index_from_name(self.device_name)
            self.launch_config = "pulsesrc device=" + str(self.device_index)
            rospy.loginfo("Using: pulsesrc device=%s name=%s", self.device_index, self.device_name)
        elif rospy.has_param('~source'):
            # 常用源: 'alsasrc'
            self.launch_config = rospy.get_param('~source')
        else:
            self.launch_config = 'autoaudiosrc'
        rospy.loginfo("Launch config: %s", self.launch_config)
        # 配置 GStreamer 管道以将原始音频输出到 appsink
        self.launch_config += (
            " ! audioconvert ! audioresample "
            "! audio/x-raw,format=S16LE,channels=1,rate=16000 "
            "! appsink name=asr emit-signals=true sync=false max-buffers=1 drop=true"
        )
        # 配置 ROS 设置
        self.started = False
        rospy.on_shutdown(self.shutdown)
        self.pub = rospy.Publisher('~output', String, queue_size=10)
        rospy.Service("~start", Empty, self.start)
        rospy.Service("~stop", Empty, self.stop)

        # 初始化 Whisper 模型和处理器
        rospy.loginfo("Loading Whisper model...")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')
            rospy.loginfo("Whisper model loaded on CUDA.")
        else:
            rospy.loginfo("Whisper model loaded on CPU.")
        
        # 初始化音频缓冲区
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_lock = GObject.MainContext()
        self.max_buffer_size = 16000 * 5  # 5秒的音频数据

        self.start_recognizer()

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
        """ 删除任何剩余的参数以免影响下次启动 """
        for param in [self._device_name_param]:
            if rospy.has_param(param):
                rospy.delete_param(param)

        """ 关闭 GTK 线程。 """
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
        # 从缓冲区提取音频数据
        array = self.buffer_to_array(buf, caps)
        if array is not None:
            # 累积音频数据
            self.audio_buffer = np.concatenate((self.audio_buffer, array))
            rospy.logdebug("Accumulated audio buffer size: %d", len(self.audio_buffer))
            # 检查是否已经累积了足够的音频数据
            if len(self.audio_buffer) >= self.max_buffer_size:
                transcription = self.transcribe(self.audio_buffer)
                if transcription:
                    msg = String()
                    msg.data = transcription
                    rospy.loginfo("Transcription: %s", msg.data)
                    self.pub.publish(msg)
                # 清空缓冲区
                self.audio_buffer = np.array([], dtype=np.float32)
        return Gst.FlowReturn.OK

    def buffer_to_array(self, buf, caps):
        # 获取缓冲区数据
        result, map_info = buf.map(Gst.MapFlags.READ)
        if not result:
            rospy.logwarn("Failed to map buffer data.")
            return None
        # 提取音频格式信息
        structure = caps.get_structure(0)
        rate = structure.get_value('rate')
        channels = structure.get_value('channels')
        format = structure.get_value('format')
        # 假设 S16LE 格式
        if format != 'S16LE' or channels != 1 or rate != 16000:
            rospy.logwarn("Unexpected audio format: %s, channels: %d, rate: %d", format, channels, rate)
            buf.unmap(map_info)
            return None
        # 将缓冲区转换为 numpy 数组
        audio_data = np.frombuffer(map_info.data, dtype=np.int16).astype(np.float32) / 32768.0
        buf.unmap(map_info)
        return audio_data

    def transcribe(self, audio_array):
        try:
            # 准备 Whisper 的输入
            inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt", language="en", task="transcribe")
            input_features = inputs.input_features
            if torch.cuda.is_available():
                input_features = input_features.to('cuda')
            # 生成转录
            predicted_ids = self.model.generate(input_features, max_length=448)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription
        except Exception as e:
            rospy.logerr("Error during transcription: %s", str(e))
            return None

if __name__ == "__main__":
    recognizer = Recognizer()
    gtk.main()
