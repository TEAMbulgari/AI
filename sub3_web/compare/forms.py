from django import forms

class UploadimageForm(forms.Form):
    file = forms.FileField()
    