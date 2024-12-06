import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .models import ImageUpload
import joblib
from PIL import Image
import torch
from torchvision import transforms

# Load the model
model_path = './ResNet18_CNN_model.joblib'
model = joblib.load(model_path)
model.eval()  # Set the model to evaluation mode

# Define the class names
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']

def preprocess_image(image_path, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict_image_class(image_path):
    image = preprocess_image(image_path, (224, 224))  # Assume the image size is 224x224
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

def image_upload_view(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)
        prediction = predict_image_class(os.path.join('uploads', filename))
        return render(request, 'drclassification/result.html', {
            'uploaded_file_url': uploaded_file_url,
            'prediction': prediction,
        })
    return render(request, 'drclassification/upload.html')
