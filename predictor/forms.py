from django import forms

class ImageUploadForm(forms.Form):
    SEX_CHOICES = [
        (0, 'Female'),
        (1, 'Male'),
    ]

    # Este campo recibirá el Base64 de la imagen ya recortada en el navegador
    cropped_image_data = forms.CharField(
        widget=forms.HiddenInput,
        required=True,
        label=''
    )

    sex = forms.ChoiceField(
        choices=SEX_CHOICES,
        label='Sex',
        required=True
    )