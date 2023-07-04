import base64
import json
import argparse
import os, os.path as osp
import cv2
import requests
from kubernetes import client, config

# kubectl port-forward pod_name -n namespace 8081:8080
# python request.py 20230219_160832_2.jpg --kserve

NAMESPACE = 'pipeline'
SVC_NAME =  'kserve-torchserve'
NODE_PORT = "8081"
NODE_ADDRESS = 'http://localhost'

INFERENCE_PORT = '8095'
MODEL_NAME = 'pipeline'
URL = f"http://localhost:{INFERENCE_PORT}/predictions/{NAMESPACE}"
ACCEPT_MB = 6


def get_isvc_url():
    group = 'serving.kserve.io' 
    version = 'v1beta1'
    namespace = NAMESPACE
    inferenceservice_name = SVC_NAME

    config.load_kube_config()
    api_instance = client.CustomObjectsApi()

    try:
        inference_service = api_instance.get_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,
            plural='inferenceservices',
            name=inferenceservice_name
        )
    except:
        raise RuntimeError(f"InferenceService:{inferenceservice_name} is not Running")
        
    url = None
    host = None
    for svc_key, svc_val in inference_service['status'].items():
        if svc_key == 'address':
            url = svc_val['url']
        if svc_key == 'url':
            host = svc_val
   
    url = url.split(host)[-1].replace(SVC_NAME, MODEL_NAME)
    kserve_url = f"{NODE_ADDRESS}:{NODE_PORT}{url}"
    kserve_headers = dict(Host = host) 
    return kserve_url, kserve_headers
        
        
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="converts image to bytes array",
                        type=str)
    parser.add_argument("--kserve", help="converts image to bytes array",
                        action='store_true')
    
    args = parser.parse_args()
    return args

def resize_image(file_name, scale):    
    file_path = osp.join(os.getcwd(), file_name)
    if not osp.isfile(file_path):
        raise OSError(f"The file does not exist! \n file path: {file_path}")
    
    org_img = cv2.imread(file_path)
    h, w, c = org_img.shape
    
    re_h, re_w = int(h*scale), int(w*scale)
    re_img = cv2.resize(org_img, (re_w, re_h))
    
    print(f"Resizing from [{h}, {w}, {c}] to [{re_h}, {re_w}, {c}]")
    
    return re_img


def show_license_plate(sub_text, main_text, type_text):
    if type_text == "r_board":
        line_top = '┏' + '━'*20 + '┓' + '\n'
        line_bottom = '┗' + '━'*20 + '┛' + '\n'
        line_sub = '┃' + ' '*4 + f'{sub_text[0]}   {sub_text[1]}      {sub_text[2]}'  + ' '*4 + '┃' + '\n'
        line_main = '┃' + ' '*2 + f'{main_text[0]}    {main_text[1]}    {main_text[2]}    {main_text[3]}'  + ' '*2 + '┃' + '\n'
        License_plate = line_top + line_sub + line_main + line_bottom
    elif type_text == "l_board":
        line_top = '┏' + '━'*30 + '┓' + '\n'
        line_bottom = '┗' + '━'*30 + '┛' + '\n'
        line_sub = '┃' + ' '*3 + f'{sub_text[0]}  {sub_text[1]}   {sub_text[2]}'  + ' '*6 
        line_main = f'{main_text[0]}  {main_text[1]}  {main_text[2]}  {main_text[3]}'  + ' '*3 + '┃' + '\n'
        License_plate = line_top + line_sub + line_main + line_bottom
    else:
        raise TypeError(f"The plate type is not vaild.")
        
    print(License_plate)
    

def send_request_get_response(request, kserve = False):        
    if kserve:                      # if serving by torchserve with kserve
        kserve_url, kserve_headers = get_isvc_url()
        response = requests.post(kserve_url, json=request, headers=kserve_headers)
        print(f"\n    request: {kserve_url}\n    header: {kserve_headers}")
    else:                           # if serving by only torchserve
        response = requests.post(URL, json=request)
        print(f"Send request to {URL}")
    
    # show response
    if not response.text == '':
        output = response.json() # ['response']
        
        if type(output['response'][0]) == str \
            or output['response'][0] == 'None': 
            print(f"License plate detection failed.")
            exit()

        if list(output.keys()) != ['response']:
            raise RuntimeError(f"The Torchserve is down!!")
        
        print(f"License plate information")
        for response in output['response']:
            for key, value in response.items():
                if key in ['width', 'height', 'board_center_p']: continue
                if key =='sub_text':
                    sub_text = value
                elif key =='main_text':
                    main_text = value
                elif key =='type':
                    type_text = value
        
            # show response(result if infernece and post processing) 
            show_license_plate(sub_text, main_text, type_text)
        
    else:
        print("Request has been denied.")
   

def endecode_image(image_bytes):
    image_64_encode = base64.b64encode(image_bytes)         # encode image to base64
    bytes_array = image_64_encode.decode('utf-8')           # decode base64 to UTF-8 string
    request = {"data": bytes_array}                         # set request
    return request
    
def get_size_reque_mbytes(request):
    # check size of request
    json_string = json.dumps(request)
    json_bytes = json_string.encode('utf-8')
    size_in_bytes = len(json_bytes)
    size_in_mbytes = size_in_bytes/(1024*1024)
    return size_in_mbytes

if __name__ == '__main__':    
    args = parser_args()
        
    file_path = osp.join(os.getcwd(), args.filename)
    if not osp.isfile(file_path):
        raise RuntimeError(f"File dose not exist: {file_path}")
    
    # load image
    image_file = open(file_path, 'rb')          # open binary file in read mode
    image_bytes = image_file.read()             # <class 'bytes'>
    
    
    request = endecode_image(image_bytes)           # decode image to bytes and set request
    size_mbytes = get_size_reque_mbytes(request)    # check size if request
    if size_mbytes > ACCEPT_MB:
        image_np = resize_image(args.filename, 6/size_mbytes*1)       # load image and resizing
        _, buffer = cv2.imencode(".jpg", image_np)          # compression in memory
        
        image_bytes = buffer.tobytes()                      # convert to `bytes`
        request = endecode_image(image_bytes)
        size_mbytes_re = get_size_reque_mbytes(request)
        if size_mbytes_re > ACCEPT_MB:
            raise RuntimeError(f"The size of the request payload too large.\n"
                f"              Size of the request payload: {size_mbytes_re:.3f} MB")

    send_request_get_response(request, args.kserve)

