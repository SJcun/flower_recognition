# encoding:utf-8
import requests 

# client_id 为官网获取的AK， client_secret 为官网获取的SK
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=VUPQKszA6SZFPzrLX53XSMFL&client_secret=kpNHiA9saiGsw3E4wmCeq6iVHMSKdbxq'
response = requests.get(host)
if response:
    print(response.json()['access_token']) #['access_token']