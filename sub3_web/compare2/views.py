#https://riptutorial.com/ko/django/example/24805/%EC%9E%A5%EA%B3%A0-%EC%96%91%EC%8B%9D%EC%9C%BC%EB%A1%9C-%ED%8C%8C%EC%9D%BC-%EC%97%85%EB%A1%9C%EB%93%9C

from django.shortcuts import render

def cam2(request): 
    return render(request, 'cam2.html')

def upload2(request):
    return render(request, 'upload2.html')

def up2(request):
    return render(request, 'upload_file2.html')
