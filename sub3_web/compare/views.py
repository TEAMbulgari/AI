#https://riptutorial.com/ko/django/example/24805/%EC%9E%A5%EA%B3%A0-%EC%96%91%EC%8B%9D%EC%9C%BC%EB%A1%9C-%ED%8C%8C%EC%9D%BC-%EC%97%85%EB%A1%9C%EB%93%9C

from django.shortcuts import render
# from django.http import HttpResponseRedirect
# import cv2, os
# import face_recognition as fr
# from matplotlib import pyplot as plt
# from PIL import Image
# import base64
# Create your views here.


def cam(request): 
    return render(request, 'cam.html')


def upload(request):
    return render(request, 'upload.html')

def up(request):
    return render(request, 'upload_file.html')

# def process(request):
#     if request.method == "POST":
#         url=request.POST.get("imgurl")         
#         if url:
#             preprocessing(url)
#             return render(request, 'process.html', {"url" : url})
#     return HttpResponseRedirect('home')


# def preprocessing(img):
#     img = base64.b64decode(img)
#     print(img)
#     image = fr.load_image_file(img)
#     top, right, bottom, left = fr.face_locations(image)
#     face_image = image[top:bottom, left:right] #배열로 이미지 전체를 저장
#     im = Image.fromarray(face_image)
#     im.save("facedata.jpg")
