import requests
import base64

'''
#植物识别
'''

request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/plant"

path='【图片地址】'
# 二进制方式打开图片文件
f = open(path, 'rb')
img = base64.b64encode(f.read())

params = {"image":img}
access_token = '【请求的access_token】'
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
response = requests.post(request_url, data=params, headers=headers)
if response:
    print (response.json())
    
