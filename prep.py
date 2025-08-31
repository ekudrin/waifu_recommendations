import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18

# Подготовка предобученной сети ResNet50
resnet_model = resnet18(weights='DEFAULT')

# Удаляем последний слой классификации, оставляя только слои для экстракции признаков
features_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1])

# Функция предварительной обработки изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Преобразуем размеры изображений
    transforms.ToTensor(),  # Конвертируем в Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализуем
])


# Функция для извлечения эмбеддингов из изображения
def extract_features(image_path):
    with torch.no_grad():
        # Загружаем изображение
        image = Image.open(image_path)

        # Применяем преобразования
        transformed_image = transform(image).unsqueeze(0)

        # Экстрагируем эмбеддинги
        embedding = features_extractor(transformed_image)

        # Приводим размерность к одномерному вектору
        flattened_embedding = embedding.view(-1)

    return flattened_embedding.numpy()


images = os.listdir('./images/')

all_names = []
all_vectors = None
features_extractor.eval()
root = './images/'

for i, file in enumerate(images):
    try:
        flattened_embedding = extract_features(root + file)

        if all_vectors is None:
            all_vectors = flattened_embedding
        else:
            all_vectors = np.vstack((all_vectors, flattened_embedding))
        all_names.append(file)
    except:
        continue

np.save('all_vectors.npy', all_vectors)
np.save('all_names.npy', all_names)
