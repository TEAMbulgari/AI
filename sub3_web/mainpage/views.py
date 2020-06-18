from django.shortcuts import render
from django.http import HttpResponseRedirect
from cv2 import cv2
import os, io
import numpy as np
import face_recognition as fr
from matplotlib import pyplot as plt
from PIL import Image
import base64
from datetime import datetime
from sub3_ai import label_image as limg
from model_project import pytest, human_pytest

# Create your views here.

modelFullPath = 'model_project/model/model_5000.pb'                             # 읽어들일 graph 파일 경로
labelsFullPath = 'model_project/model/labels_5000.txt'                               # 읽어들일 labels 파일 경로
modelFullPath_human = 'model_project/model/human_model6_30000.pb'                             # 읽어들일 graph 파일 경로
labelsFullPath_human = 'model_project/model/labels_human6_30000.txt' 

def home(request):
    return render(request,'home.html')

def resultPage(request):
    if request.method == "POST":
        url=request.POST.get("imgurl")         
        if url:
            target=preprocessing(url)
            if target:
                result = human_pytest.run_inference_on_image_human(target, modelFullPath_human , labelsFullPath_human)
                target = os.path.join('media/',os.path.basename(target))
                return render(request, 'resultPage.html', {"url" : url, "result1": result[0]['class'], "result2" :result[1]['class'], "result3": result[2]['class'],
                                                        "percent1":round(result[0]['score']*100,2), "percent2":round(result[1]['score']*100,2), "percent3":round(result[2]['score']*100,2), "target":target, "sampleimg":result[0]['path']})
    return HttpResponseRedirect('upload')

def resultPage2(request):
    if request.method == "POST":
        url=request.POST.get("imgurl")         
        if url:
            target=preprocessing(url)
            if target:
                result = pytest.run_inference_on_image(target, modelFullPath, labelsFullPath)
                target = os.path.join('media/',os.path.basename(target))
                return render(request, 'resultPage.html', {"url" : url, "result1": result[0]['class'], "result2" :result[1]['class'], "result3": result[2]['class'],
                                                        "percent1":round(result[0]['score']*100,2), "percent2":round(result[1]['score']*100,2), "percent3":round(result[2]['score']*100,2), "target":target, "sampleimg":result[0]['path']})
    return HttpResponseRedirect('uploads')

def comparePage(request, category):
    print("콘솔"+request.method)
    print(category)
    return render(request,'resultPage.html',{'category':category})

def test(request):
    return render(request,'test.html')


def preprocessing(img):
    head, body = img.split(',')
    nparr = np.fromstring(base64.b64decode(body), np.uint8)
    imageBGR = cv2.imdecode(nparr,cv2.IMREAD_ANYCOLOR)
    image = cv2.cvtColor(imageBGR , cv2.COLOR_BGR2RGB)
    if fr.face_locations(image):    
        filename = str(datetime.now())
        filename = filename.replace(":","")
        filename = filename.replace(" ","")
        filename = filename.replace(".","")
        filename = "media/"+filename+".jpg"
        im = Image.fromarray(image)
        im.save(filename)
        return os.path.abspath(filename)
    return ""
