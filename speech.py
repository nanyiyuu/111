#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# 2020/2/19   修改人员：monster water

"""
recognizer.py 是科大讯飞语音识别的ROS节点包装器。
  参数:
    ~mic_name - 设置麦克风输入的pulsesrc设备名称。
                例如，Logitech G35耳机的设备名称为：alsa_input.usb-Logitech_Logitech_G35_Headset-00-Headset_1.analog-mono
                要在终端中列出音频设备信息，请输入：pacmd list-sources
    ~appid - 您的科大讯飞应用ID
    ~api_key - 您的科大讯飞API Key
    ~api_secret - 您的科大讯飞API Secret
  发布:
    ~output (std_msgs/String) - 文字输出
  服务:
    ~start (std_srvs/Empty) - 启动语音识别
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
import json
import base64
import hashlib
import time
import hmac
import requests
import websocket
import threading
from urllib.parse import quote

# 配置日志
import logging
logging.basicConfig()

# 科大讯飞 WebSocket 端点
base_url = "ws://rtasr.xfyun.cn/v1/ws"

# 结束标志
end_tag = "{\"end\": true}"

class IFLYTEKClient:
    """ 科大讯飞语音识别客户端。 """

    def __init__(self, app_id, api_key, api_secret, publish_callback):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.publish_callback = publish_callback  # 回调函数用于发布识别结果

        self.ws = None
        self.thread = None
        self.stop_event = threading.Event()

    def connect(self):
        """ 连接到科大讯飞的 WebSocket 服务 """
        ts = str(int(time.time()))
        tmp = self.app_id + ts
        hl = hashlib.md5()
        hl.update(tmp.encode('utf-8'))
        h2 = hl.hexdigest()
        apikey = self.api_key.encode('utf-8')
        h2 = h2.encode('utf-8')
        my_sign = hmac.new(apikey, h2, hashlib.sha1).digest()
        signa = base64.b64encode(my_sign).decode('utf-8')

        ws_url = f"{base_url}?appid={self.app_id}&ts={ts}&signa={quote(signa)}"

        # 创建WebSocket连接
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        # 启动WebSocket线程
        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.start()

    def on_open(self, ws):
        rospy.loginfo("WebSocket连接已打开。")
        # 发送配置参数
        config = {
            "common": {
                "app_id": self.app_id,
                "engine_type": "sms16k",  # 引擎类型，根据需求选择
                "aue": "raw"
            },
            "business": {
                "language": "zh_cn",
                "domain": "iat",
                "accent": "mandarin",
                "vinfo": 1,
                "vad_eos": 1000
            },
            "data": {
                "status": 0,
                "format": "audio/L16;rate=16000",
                "encoding": "raw"
            }
        }
        ws.send(json.dumps(config))

    def on_message(self, ws, message):
        """ 处理服务器返回的消息 """
        try:
            result_dict = json.loads(message)
        except json.JSONDecodeError:
            rospy.logwarn(f"无法解析的消息: {message}")
            return

        # 解析结果
        if result_dict.get("action") == "started":
            rospy.loginfo("握手成功，开始识别。")

        elif result_dict.get("action") == "result":
            if "data" in result_dict:
                result_data = result_dict["data"]
                if "result" in result_data:
                    # 解析识别结果
                    ws_result = json.loads(result_data)
                    if 'cn' in ws_result and 'st' in ws_result['cn']:
                        st = ws_result['cn']['st']
                        for item in st:
                            if 'rt' in item and len(item['rt']) > 0:
                                ws_ws = item['rt'][0].get('ws', [])
                                str_result = []
                                for ws_item in ws_ws:
                                    cw = ws_item['cw'][0]['w']
                                    str_result.append(cw)
                                transcription = ''.join(str_result)
                                if transcription:
                                    rospy.loginfo(f"识别结果: {transcription}")
                                    self.publish_callback(transcription)

        elif result_dict.get("action") == "error":
            rospy.logerr(f"识别错误: {message}")
            self.close()

    def on_error(self, ws, error):
        rospy.logerr(f"WebSocket错误: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        rospy.loginfo("WebSocket连接已关闭。")

    def send_audio(self, audio_data):
        """ 发送音频数据到WebSocket """
        if self.ws and self.ws.sock and self.ws.sock.connected:
            try:
                self.ws.send(audio_data, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception as e:
                rospy.logerr(f"发送音频数据失败: {e}")
        else:
            rospy.logwarn("WebSocket未连接，无法发送音频数据。")

    def send_end_tag(self):
        """ 发送结束标志 """
        if self.ws and self.ws.sock and self.ws.sock.connected:
            try:
                self.ws.send(end_tag.encode('utf-8'), opcode=websocket.ABNF.OPCODE_TEXT)
                rospy.loginfo("发送结束标志成功。")
            except Exception as e:
                rospy.logerr(f"发送结束标志失败: {e}")

    def close(self):
        """ 关闭WebSocket连接 """
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join()
        rospy.loginfo("WebSocket连接已关闭。")

class RecognizerNode:
    """ ROS节点，使用科大讯飞的语音识别API进行语音转文字。 """

    def __init__(self):
        # 初始化ROS节点
        rospy.init_node("recognizer")

        # 参数名称
        self._device_name_param = "~mic_name"
        self._appid_param = "~appid"
        self._api_key_param = "~api_key"
        self._api_secret_param = "~api_secret"

        # 获取AppID、API Key和API Secret
        if rospy.has_param(self._appid_param) and rospy.has_param(self._api_key_param) and rospy.has_param(self._api_secret_param):
            self.appid = rospy.get_param(self._appid_param)
            self.api_key = rospy.get_param(self._api_key_param)
            self.api_secret = rospy.get_param(self._api_secret_param)
        else:
            rospy.logerr("AppID、API Key和API Secret参数未设置。请设置~appid、~api_key和~api_secret参数。")
            raise Exception("AppID、API Key和API Secret参数未设置。")

        # 配置麦克风
        if rospy.has_param(self._device_name_param):
            self.device_name = rospy.get_param(self._device_name_param)
            self.device_index = self.pulse_index_from_name(self.device_name)
            self.launch_config = f"pulsesrc device={self.device_index}"
            rospy.loginfo(f"使用麦克风设备: pulsesrc device={self.device_index} name={self.device_name}")
        elif rospy.has_param('~source'):
            # 常见源: 'alsasrc'
            self.launch_config = rospy.get_param('~source')
        else:
            self.launch_config = 'autoaudiosrc'

        rospy.loginfo(f"GStreamer配置: {self.launch_config}")

        # 配置GStreamer管道以输出原始音频到appsink
        self.launch_config += (
            " ! audioconvert ! audioresample "
            "! audio/x-raw,format=S16LE,channels=1,rate=16000 "
            "! appsink name=asr emit-signals=true sync=false max-buffers=1 drop=true"
        )

        # 配置ROS设置
        self.started = False
        rospy.on_shutdown(self.shutdown)
        self.pub = rospy.Publisher('~output', String, queue_size=10)
        rospy.Service("~start", Empty, self.start)
        rospy.Service("~stop", Empty, self.stop)

        # 初始化科大讯飞语音识别
        rospy.loginfo("初始化科大讯飞语音识别客户端...")
        self.xfyun_recognizer = IFLYTEKClient(
            self.appid,
            self.api_key,
            self.api_secret,
            self.publish_result  # 回调函数
        )
        self.ws_thread = None

    def publish_result(self, transcription):
        """ 发布识别结果到ROS话题 """
        msg = String()
        msg.data = transcription
        rospy.loginfo(f"发布识别结果: {msg.data}")
        self.pub.publish(msg)

    def start_recognizer(self):
        rospy.loginfo("启动识别器... ")

        self.pipeline = gst.parse_launch(self.launch_config)
        self.appsink = self.pipeline.get_by_name('asr')
        self.appsink.connect('new-sample', self.on_new_sample)

        self.pipeline.set_state(gst.State.PLAYING)
        self.started = True
        rospy.loginfo("识别器已启动，管道正在播放。")

        # 连接科大讯飞WebSocket
        self.xfyun_recognizer.connect()

    def pulse_index_from_name(self, name):
        """ 根据设备名称获取Pulse index """
        try:
            output = os.popen(
                f"pacmd list-sources | grep -B 1 'name: <{name}>' | grep -o -P '(?<=index: )[0-9]*'"
            ).read().strip()
            if output.isdigit():
                return int(output)
            else:
                raise Exception(f"错误。该名称的Pulse index不存在: {name}")
        except Exception as e:
            rospy.logerr(str(e))
            raise

    def stop_recognizer(self):
        if self.started:
            # 发送结束标志
            self.xfyun_recognizer.send_end_tag()
            # 关闭WebSocket连接
            self.xfyun_recognizer.close()

            # 关闭GStreamer管道
            self.pipeline.set_state(gst.State.NULL)
            self.pipeline = None
            self.appsink = None
            self.started = False
            rospy.loginfo("识别器已停止，管道已关闭。")

    def shutdown(self):
        """ 删除任何剩余的参数以避免影响下次启动 """
        for param in [self._device_name_param, self._appid_param, self._api_key_param, self._api_secret_param]:
            if rospy.has_param(param):
                rospy.delete_param(param)

        """ 关闭GTK线程。 """
        gtk.main_quit()

    def start(self, req):
        if not self.started:
            self.start_recognizer()
            rospy.loginfo("通过服务启动识别器。")
        else:
            rospy.loginfo("识别器已在运行。")
        return EmptyResponse()

    def stop(self, req):
        if self.started:
            self.stop_recognizer()
            rospy.loginfo("通过服务停止识别器。")
        else:
            rospy.loginfo("识别器未在运行。")
        return EmptyResponse()

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        # 提取音频数据
        array = self.buffer_to_array(buf, caps)
        if array is not None:
            # 将音频数据发送到科大讯飞语音识别
            audio_bytes = (array * 32768).astype(np.int16).tobytes()
            self.xfyun_recognizer.send_audio(audio_bytes)
        return Gst.FlowReturn.OK

    def buffer_to_array(self, buf, caps):
        # 获取缓冲区数据
        result, map_info = buf.map(Gst.MapFlags.READ)
        if not result:
            rospy.logwarn("无法映射缓冲区数据。")
            return None

        # 提取音频格式信息
        structure = caps.get_structure(0)
        rate = structure.get_value('rate')
        channels = structure.get_value('channels')
        format = structure.get_value('format')

        # 假设为S16LE格式
        if format != 'S16LE' or channels != 1 or rate != 16000:
            rospy.logwarn(f"意外的音频格式: {format}, 通道数: {channels}, 采样率: {rate}")
            buf.unmap(map_info)
            return None

        # 将缓冲区转换为numpy数组
        audio_data = np.frombuffer(map_info.data, dtype=np.int16).astype(np.float32) / 32768.0
        buf.unmap(map_info)
        return audio_data

if __name__ == "__main__":
    try:
        node = RecognizerNode()
        gtk.main()
    except rospy.ROSInterruptException:
        pass
