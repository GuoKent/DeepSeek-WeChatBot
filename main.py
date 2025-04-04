#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import signal
from argparse import ArgumentParser

from configs.robot_config import Config
from robot import Robot, __version__
from wcferry import Wcf

def main():
    config = Config()
    wcf = Wcf(debug=True)

    def handler(sig, frame):
        wcf.cleanup()  # 退出前清理环境
        exit(0)

    signal.signal(signal.SIGINT, handler)

    robot = Robot(config, wcf, model_name="deepseek")
    robot.LOG.info(f"WeChatRobot {__version__} 成功启动···")

    # 机器人启动发送测试消息
    robot.sendTextMsg(msg="机器人启动成功！", receiver="filehelper")

    # 接收消息
    # robot.enableRecvMsg()     # 可能会丢消息？
    robot.enableReceivingMsg()  # 加队列

    # 让机器人一直跑
    robot.keepRunningAndBlockProcess()


if __name__ == "__main__":
    main()
