#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Version  : 1.0
# @Author   : QQ736592720
# @Datetime : 2022/4/9 18:04
# @Project  : 简答题399___baidu_ocr.py
# @File     : 简答题457___轻量级服务器lighthouse搭建.py
import traceback
import text_recognition
from bottle import route, run, request
from json import dumps

@route('/recognition', method='POST')
def recognition():
    uploadfile = request.files.get('img_avatar')  # 获取上传的文件
    uploadfile.save("./upload/1.png", overwrite=True)  # overwrite参数是指覆盖同名文件
    rec_res=text_recognition.recognition("./upload/1.png")
    return "\n"+str(rec_res)

@route('/hello', method='GET')
def hello():
    return "update"


if __name__ == '__main__':
    # email(key) phone comment
    run(host='0.0.0.0', port=8888, debug=False)  # 记得服务器开启8888端口