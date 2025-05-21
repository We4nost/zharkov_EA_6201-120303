import requests
from PIL import Image
from io import BytesIO
import numpy as np
from scipy.ndimage import laplace
from dotenv import load_dotenv
import os

#Получение изображения
load_dotenv()
API_KEY = os.getenv('API_KEY')
headers = {"x-api-key": API_KEY}

# Запрос изображения
params = {"has_breeds" : 1}
response = requests.get("https://api.thecatapi.com/v1/images/search", headers=headers, params=params)
response.raise_for_status()

data1 = response.json()
data = data1[0]
image_url = data["url"]
breed = data["breeds"][0]["name"] if data["breeds"] else "unknown_breed"

print(f"Порода: {breed}")
print(f"Ссылка на изображение: {image_url}")

# Загрузка изображения
img_response = requests.get(image_url)
img_response.raise_for_status()
image = Image.open(BytesIO(img_response.content)).convert("L")
image_np = np.array(image)


#Оператор Лапласа ручная реализация

def manual_laplacian_filter(image_array):
    # Ядро Лапласа 3x3
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])

    # рамка
    padded_image = np.pad(image_array, ((1, 1), (1, 1)), mode='edge')
    output_image = np.zeros_like(image_array)

    height, width = image_array.shape

     # По каждому каналу
    for i in range(height):
        for j in range(width):
                # Извлекаем окно 3x3
            window = padded_image[i:i + 3, j:j + 3]
                # Применяем ядро
            value = np.sum(window * kernel)
                # Ограничиваем значение по диапазону [0, 255]
            output_image[i, j] = np.clip(abs(value), 0, 255)

    return output_image.astype(np.uint8)


#Оператор Лапласа (SciPy реализация)

def scipy_laplacian_filter(image_array):
    filtered_image = np.zeros_like(image_array)
    for c in range(1):  # По каждому каналу
        # Функция laplace сама применяет ядро Лапласа
        channel_filtered = laplace(image_array[:, :])
        # Берём абсолютное значение и ограничиваем по диапазону
        filtered_image[:, :] = np.clip(np.abs(channel_filtered), 0, 255)
    return filtered_image.astype(np.uint8)


#Сохранение котов

# Создаем директорию, если не существует
os.makedirs("results", exist_ok=True)

#Пути для сохранения
breed_clean = breed.replace(' ', '_')
original_path = f"results/{breed_clean}_original.jpg"
manual_laplacian_path = f"results/{breed_clean}_manual_laplacian.jpg"
scipy_laplacian_path = f"results/{breed_clean}_scipy_laplacian.jpg"

# Сохранение изображений
image.save(original_path)
Image.fromarray(manual_laplacian_filter(image_np)).save(manual_laplacian_path)
Image.fromarray(scipy_laplacian_filter(image_np)).save(scipy_laplacian_path)

print("\nИзображения успешно сохранены:")
print(f"- Оригинал: {original_path}")
print(f"- Лаплас вручную: {manual_laplacian_path}")
print(f"- Лаплас SciPy: {scipy_laplacian_path}")