#https://riptutorial.com/ko/django/example/24805/%EC%9E%A5%EA%B3%A0-%EC%96%91%EC%8B%9D%EC%9C%BC%EB%A1%9C-%ED%8C%8C%EC%9D%BC-%EC%97%85%EB%A1%9C%EB%93%9C

from django.shortcuts import render
from .forms import UploadimageForm

# Create your views here.

def upload_img(request):
    form = UploadimageForm()
    if request.method =='POST':
        form = UploadimageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
        return render(request, 'upload.html', locals())