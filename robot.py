# -*- coding: utf-8 -*-
import logging
import re
import time
import xml.etree.ElementTree as ET
from queue import Empty
from threading import Thread

from utils.utils import load_user_config
from wcferry import Wcf, WxMsg
from model.deepseek import DeepSeek
from configs.robot_config import Config
from job_mgmt import Job
# from utils.func_news import News
from utils.utils import load_model_config, load_preinfo

__version__ = "39.2.4.0"

class Robot(Job):
    def __init__(self, config: Config, wcf: Wcf, model_name: str = None) -> None:
        self.wcf = wcf
        self.config = config
        self.LOG = logging.getLogger("Robot")
        self.wxid = self.wcf.get_self_wxid()
        self.wx_info = wcf.get_user_info()  # {'wxid': xxx, 'name': xxx, 'home': root_path}
        self.allContacts = self.getAllContacts()
        self.model_name = model_name
        self.user_default_config = load_user_config()
        self.user = {}  # {'user_name': {'config': {...}, 'history': [...]}}
        self.rag = True

        if model_name is not None:
            model_config = load_model_config(model_name)
            self.model : DeepSeek = DeepSeek(model_config)

        self.LOG.info(f"初始化模型: {model_name}")

    @staticmethod
    def value_check(args: dict) -> bool:
        if args:
            return all(value is not None for key, value in args.items() if key != 'proxy')
        return False

    def toAt(self, msg: WxMsg, pre_info: str=None) -> bool:
        """
        处理被 @ 消息
        :param msg: 微信消息结构
        :return: 处理状态，`True` 成功，`False` 失败
        """
        return self.toChitchat(msg, pre_info)
    
    def mask_think(self, response: str) -> str:
        begin = response.find("<think>")
        end = response.find("</think>")
        return response[end+10:]

    def toChitchat(self, msg: WxMsg, pre_info: str=None) -> bool:
        """
        闲聊模式
        """
        if self.model_name:
            # wait_msg = f"思考ing...\n请等待, 前面还有 {self.wcf.msgQ.qsize()} 人"
            # self.replyTextMsg(wait_msg, msg)
            # 初始化用户配置 & 处理用户历史记录
            if msg.sender not in self.user:
                self.user[msg.sender] = {}
                self.user[msg.sender]['config'] = self.user_default_config
                self.user[msg.sender]['history'] = self.model.default_messages + [{"role": "user", "content": msg.content}]
            else:
                self.user[msg.sender]['history'].append({"role": "user", "content": msg.content})
                self.user[msg.sender]['history'] = self.model.clean_history_messages(self.user[msg.sender]['history'], history=5)
            # 使用RAG, 添加前置知识
            if self.rag and pre_info is not None:
                self.user[msg.sender]['history'][0]['content'] = pre_info + self.user[msg.sender]['history'][0]['content']
            # 生成回复
            outputs = self.model.generate(self.user[msg.sender]['history'])
            self.user[msg.sender]['history'].append({"role": "assistant", "content": outputs['response']})
            if self.user[msg.sender]['config']['mask_think']:
                outputs['response'] = self.mask_think(outputs['response'])
            rsp = \
            f"{outputs['response']}\n" + \
            f"| Cost time: {outputs['cost_time']:.2f}s |\n" + \
            f"| Token nums: {outputs['token_num']} |\n" + \
            f"| Token speed: {outputs['token_speed']:.2f} token/s |"
        else:  # 自动回复
            rsp = "模型未启动喵~这里是自动回复喵~"

        if rsp:  # 回复信息
            self.replyTextMsg(rsp, msg)
            return True
        else:
            rsp = "目前人数较多，服务器繁忙，请稍后再试喵~"
            self.replyTextMsg(rsp, msg)
            self.LOG.error(f"目前人数较多，服务器繁忙，请稍后再试")
            return False

    def processMsg(self, msg: WxMsg) -> None:
        """
        当接收到消息的时候，会调用本方法。如果不实现本方法，则打印原始消息。
        此处可进行自定义发送的内容,如通过 msg.content 关键字自动获取当前天气信息，并发送到对应的群组@发送者
        群号: msg.roomid  微信ID: msg.sender  消息内容: msg.content
        content = "xx天气信息为: "
        receivers = msg.roomid
        self.sendTextMsg(content, receivers, msg.sender)
        """

        if msg.content == '/clean':
            self.user[msg.sender]['history'] = self.model.clean_history_messages(self.user[msg.sender]['history'], history=0)
            self.sendTextMsg(f"{msg.sender}的历史记录清理完毕, 当前列表长度: {len(self.user[msg.sender]['history'])-1}", msg.sender)
            return

        # 群聊消息
        if msg.from_group():
            # 如果在群里被 @
            if msg.roomid not in self.config.GROUPS:  # 不在配置的响应的群列表里，忽略
                return

            if msg.is_at(self.wxid):  # 被@
                # msg.content = str(msg.content[len(self.wx_info['name'])+2:])
                # print(msg.content)
                if self.rag:
                    pre_info = load_preinfo(msg.roomid)
                self.toAt(msg, pre_info)
            return  # 处理完群聊信息，后面就不需要处理了

        # 非群聊信息，按消息类型进行处理
        if msg.type == 37:  # 好友请求
            self.autoAcceptFriendRequest(msg)

        elif msg.type == 10000:  # 系统信息
            self.sayHiToNewFriend(msg)

        elif msg.type == 0x01:  # 文本消息
            # 让配置加载更灵活，自己可以更新配置。也可以利用定时任务更新。
            if msg.from_self():
                if msg.content == "^更新$":
                    self.config.reload()
                    self.LOG.info("已更新")
            else:
                self.toChitchat(msg)  # 私聊

    def onMsg(self, msg: WxMsg) -> int:
        try:
            self.LOG.info(msg)  # 打印信息
            self.processMsg(msg)
        except Exception as e:
            self.LOG.error(e)

        return 0

    def enableRecvMsg(self) -> None:
        self.wcf.enable_recv_msg(self.onMsg)

    def enableReceivingMsg(self) -> None:
        def innerProcessMsg(wcf: Wcf):
            while wcf.is_receiving_msg():
                try:
                    msg = wcf.get_msg()
                    self.LOG.info(msg)
                    self.processMsg(msg)
                except Empty:
                    continue  # Empty message
                except Exception as e:
                    self.LOG.error(f"Receiving message error: {e}")

        self.wcf.enable_receiving_msg()
        Thread(target=innerProcessMsg, name="GetMessage", args=(self.wcf,), daemon=True).start()

    def sendTextMsg(self, msg: str, receiver: str, at_list: str = "") -> None:
        """ 发送消息
        :param msg: 消息字符串
        :param receiver: 接收人wxid或者群id
        :param at_list: 要@的wxid, @所有人的wxid为: notify@all
        """
        # msg 中需要有 @ 名单中一样数量的 @
        ats = ""
        if at_list:
            if at_list == "notify@all":  # @所有人
                ats = " @所有人"
            else:
                wxids = at_list.split(",")
                for wxid in wxids:
                    # 根据 wxid 查找群昵称
                    ats += f" @{self.wcf.get_alias_in_chatroom(wxid, receiver)}"

        # {msg}{ats} 表示要发送的消息内容后面紧跟@，例如 北京天气情况为：xxx @张三
        if ats == "":
            self.LOG.info(f"To {receiver}: {msg}")
            self.wcf.send_text(f"{msg}", receiver, at_list)
        else:
            self.LOG.info(f"To {receiver}: {ats}\r{msg}")
            self.wcf.send_text(f"{ats}\n\n{msg}", receiver, at_list)

    def replyTextMsg(self, rsp: str, msg: WxMsg) -> None:
        '''
        回复消息
        '''
        if msg.from_group():
            self.sendTextMsg(rsp, msg.roomid, msg.sender)
        else:
            self.sendTextMsg(rsp, msg.sender)

    def getAllContacts(self) -> dict:
        """
        获取联系人（包括好友、公众号、服务号、群成员……）
        格式: {"wxid": "NickName"}
        """
        contacts = self.wcf.query_sql("MicroMsg.db", "SELECT UserName, NickName FROM Contact;")
        return {contact["UserName"]: contact["NickName"] for contact in contacts}

    def keepRunningAndBlockProcess(self) -> None:
        """
        保持机器人运行，不让进程退出
        """
        while True:
            self.runPendingJobs()
            time.sleep(1)

    def autoAcceptFriendRequest(self, msg: WxMsg) -> None:
        try:
            xml = ET.fromstring(msg.content)
            v3 = xml.attrib["encryptusername"]
            v4 = xml.attrib["ticket"]
            scene = int(xml.attrib["scene"])
            self.wcf.accept_new_friend(v3, v4, scene)

        except Exception as e:
            self.LOG.error(f"同意好友出错：{e}")

    def sayHiToNewFriend(self, msg: WxMsg) -> None:
        nickName = re.findall(r"你已添加了(.*)，现在可以开始聊天了。", msg.content)
        if nickName:
            # 添加了好友，更新好友列表
            self.allContacts[msg.sender] = nickName[0]
            self.sendTextMsg(f"Hi {nickName[0]}，我自动通过了你的好友请求。", msg.sender)

    # def newsReport(self) -> None:
    #     receivers = self.config.NEWS
    #     if not receivers:
    #         return

    #     news = News().get_important_news()
    #     for r in receivers:
    #         self.sendTextMsg(news, r)
