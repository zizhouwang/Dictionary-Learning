import copy
from baidubce import bce_base_client
from baidubce.auth import bce_credentials
from baidubce.auth import bce_v1_signer
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_methods
from baidubce import bce_client_configuration

# 手写文字识别 Python示例代码
# class ApiCenterClient(bce_base_client.BceBaseClient):
#
#     def __init__(self, config=None):
#         self.service_id = 'apiexplorer'
#         self.region_supported = True
#         self.config = copy.deepcopy(bce_client_configuration.DEFAULT_CONFIG)
#
#         if config is not None:
#             self.config.merge_non_none_values(config)
#
#     def _merge_config(self, config=None):
#         if config is None:
#             return self.config
#         else:
#             new_config = copy.copy(self.config)
#             new_config.merge_non_none_values(config)
#             return new_config
#
#     def _send_request(self, http_method, path,
#                       body=None, headers=None, params=None,
#                       config=None, body_parser=None):
#         config = self._merge_config(config)
#         if body_parser is None:
#             body_parser = handler.parse_json
#
#         return bce_http_client.send_request(
#             config, bce_v1_signer.sign, [handler.parse_error, body_parser],
#             http_method, path, body, headers, params)
#
#     def demo(self):
#         path = b'/rest/2.0/ocr/v1/handwriting'
#         headers = {}
#         headers[b'Content-Type'] = 'application/x-www-form-urlencoded;charset=UTF-8'
#
#         params = {}
#
#         params['access_token'] = '24.b165c77c66f0c916aca18cb13d8bcd2b.2592000.1655274843.282335-26242205'
#
#         body = 'url=https://baidu-ai.bj.bcebos.com/ocr/ocr.jpg&probability=false'
#         return self._send_request(http_methods.POST, path, body, headers, params)

# if __name__ == '__main__':
#     endpoint = 'https://aip.baidubce.com'
#     ak = ''
#     sk = ''
#     config = bce_client_configuration.BceClientConfiguration(credentials=bce_credentials.BceCredentials(ak, sk),
#                                                              endpoint=endpoint)
#     client = ApiCenterClient(config)
#     res = client.demo()
#     print(res.__dict__['raw_data'])
#     aa=1

from paddleocr import PaddleOCR
import os
import scipy.io as scio
from cnocr import CnOcr
import easyocr
from dbnet.dbnet_infer import DBNET
import onnxruntime as rt
import numpy as np
import os
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
import ddddocr

standard=0.9
paddleOCR_res={}
cnOCR_res={}
easyOCR_res={}
def is_Chinese(word):
    for ch in word:
        if '\u4e00' > ch or word > '\u9fff':
            return False
    return True
all_res=[]



import sys
import io

paddleOCR = PaddleOCR(use_angle_cls=True, lang='ch')  # need to run only once to download and load model into memory
cnocr = CnOcr()
reader = easyocr.Reader(['ch_sim','en'])
ddd_ocr = ddddocr.DdddOcr()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
def recognition(img_path):
    max_res = 0.0
    res_res = 0
    if os.path.exists(img_path):
        result = reader.readtext(img_path)
        res = cnocr.ocr(img_path)
        for line in result:
            all_res.append((line[1], line[2]))
            if line[2] > max_res:
                max_res = line[2]
                res_res = line[1]
        if len(res) > 0:
            for line in res:
                if len(line[0]) == 0:
                    continue
                all_res.append((line[0][0], line[1]))
                if line[1] > max_res:
                    max_res = line[1]
                    for ocr_res in line[0]:
                        if is_Chinese(ocr_res):
                            res_res = ocr_res
                            break
        easy_ress = paddleOCR.ocr(img_path, cls=True)
        if len(easy_ress) > 0:
            for easy_res in easy_ress:
                all_res.append(easy_res[1])
                if easy_res[1][1] > max_res:
                    max_res = easy_res[1][1]
                    res_res = easy_res[1][0]
        if max_res > standard:
            return res_res
        else:
            with open(img_path, 'rb') as f:
                image = f.read()
            f.close()
            res = ddd_ocr.classification(image)
            if len(res) > 0 and is_Chinese(res):
                return res
            else:
                res = ddd_ocr.classification(image)
                if len(res) > 0 and is_Chinese(res):
                    return res
                else:
                    return res_res
    # !/usr/bin/env python
    # -*- coding: utf-8 -*-




if __name__ == '__main__':
    img_path=sys.argv[1]
    # img_path="jpg/25.jpg"
    print(recognition(img_path))
    sys.stdout.flush()
    # for i in range(10263):
    #     print(i)
    #     img_path = './jpg/'+str(i+1)+'.jpg'
    #     sys.stdout.flush()
    #     if os.path.exists(img_path):
    #         pass
    #         result = reader.readtext(img_path)
    #         res = cnocr.ocr(img_path)
    #         if len(result)>1:
    #             aa=1
    #         if len(res)>1:
    #             aa=1
    #         for line in result:
    #             all_res.append((line[1],line[2]))
    #             # if line[1][1]>standard and is_Chinese(line[1][0]):
    #             #     paddleOCR_res[i+1]=line[1][0]
    #         if len(res)>0:
    #             for line in res:
    #                 if len(line[0])==0:
    #                     continue
    #                 all_res.append((line[0][0],line[1]))
    #                 if line[1]>standard and is_Chinese(line[0][0]):
    #                     cnOCR_res[i+1]=line[0][0]
    #         easy_ress = ocr.ocr(img_path, cls=True)
    #         if len(easy_ress)>0:
    #             for easy_res in easy_ress:
    #                 all_res.append(easy_res[1])
    #                 if easy_res[1][1]>standard:
    #                     easyOCR_res[i + 1] = easy_res[1][0]
    #     text_handle = DBNET(model_path)
    #     boxes_list, score_list = self.text_handle.process(np.asarray(img).astype(np.uint8), short_size=short_size)
    #     result = crnnRecWithBox(np.array(img), boxes_list, score_list)
    # print(paddleOCR_res)
    # scio.savemat('paddleOCR_res.mat', {'paddleOCR_res': paddleOCR_res})
    # print(cnOCR_res)
    # scio.savemat('cnOCR_res.mat', {'cnOCR_res': cnOCR_res})