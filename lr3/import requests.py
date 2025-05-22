import requests
from PIL import Image
from io import BytesIO
import numpy as np
from scipy.ndimage import laplace
from dotenv import load_dotenv
import os

class ImageProcessor:

    DEFAULT_BREED = "unknown_breed"

    def __init__(self, image_url=None, breed = None):
        self._image_url = image_url
        self._breed = breed
        self._image = None
        self._image_np = None

    @property
    def image_url(self):
        return self._image_url

    @image_url.setter
    def image_url(self, value):
        self._image_url = value

    @property
    def breed(self):
        return self._breed

    @breed.setter
    def breed(self, value):
        self._breed = value

    @property
    def image(self):
        return self._image

    @property
    def image_np(self):
        return self._image_np

    @classmethod
    def from_api(cls, api_key, limit=1):
        headers = {"x-api-key": api_key}
        params = {"has_breeds": 1, "limit": limit}
        response = requests.get("https://api.thecatapi.com/v1/images/search", 
                              headers=headers, params=params)
        response.raise_for_status()

        processors = []
        for data in response.json():
            image_url = data["url"]
            breed = data["breeds"][0]["name"] if data["breeds"] else cls.DEFAULT_BREED
            processor = cls(image_url, breed)
            processor.load_image()
            processors.append(processor)
        
        return processors if limit > 1 else processors[0]

    def load_image(self):
        img_response = requests.get(self._image_url)
        img_response.raise_for_status()
        self._image = Image.open(BytesIO(img_response.content)).convert("L")
        self._image_np = np.array(self._image)
        return self

    def manual_laplacian_filter(self):
        if self._image_np is None:
            raise ValueError("Изображение не загружено")

        kernel = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])

        padded_image = np.pad(self._image_np, ((1, 1), (1, 1)), mode='edge')
        output_image = np.zeros_like(self._image_np)
        height, width = self._image_np.shape

        for i in range(height):
            for j in range(width):
                window = padded_image[i:i + 3, j:j + 3]
                value = np.sum(window * kernel)
                output_image[i, j] = np.clip(abs(value), 0, 255)

        return output_image.astype(np.uint8)

    def scipy_laplacian_filter(self):
        if self._image_np is None:
            raise ValueError("Изображение не загружено")

        filtered_image = np.zeros_like(self._image_np)
        channel_filtered = laplace(self._image_np[:, :])
        filtered_image[:, :] = np.clip(np.abs(channel_filtered), 0, 255)
        return filtered_image.astype(np.uint8)

    def save_images(self, base_dir="lr3/results", index=0):
        os.makedirs(base_dir, exist_ok=True)
        
        breed_clean = self._breed.replace(' ', '_')
        prefix = f"{index}_{breed_clean}" if index else breed_clean

        original_path = os.path.join(base_dir, f"{prefix}_original.jpg")
        manual_path = os.path.join(base_dir, f"{prefix}_manual_laplacian.jpg")
        scipy_path = os.path.join(base_dir, f"{prefix}_scipy_laplacian.jpg")

        self._image.save(original_path)
        Image.fromarray(self.manual_laplacian_filter()).save(manual_path)
        Image.fromarray(self.scipy_laplacian_filter()).save(scipy_path)

        return {
            "original": original_path,
            "manual_laplacian": manual_path,
            "scipy_laplacian": scipy_path
        }


def main():
    load_dotenv()
    API_KEY = os.getenv('API_KEY')

    # Получаем 3 изображения за один запрос
    processors = ImageProcessor.from_api(API_KEY, limit=3)

    for i, processor in enumerate(processors, start=1):
        print(f"\nОбработка изображения {i}:")
        print(f"Порода: {processor.breed}")
        print(f"Ссылка на изображение: {processor.image_url}")

        paths = processor.save_images(index=i)
        print("\nИзображения успешно сохранены:")
        print(f"- Оригинал: {paths['original']}")
        print(f"- Лаплас вручную: {paths['manual_laplacian']}")
        print(f"- Лаплас SciPy: {paths['scipy_laplacian']}")


if __name__ == "__main__":
    main()