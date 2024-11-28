import io
import os

import numpy as np
import rembg
from PIL import Image


def remove_background(input_path, output_path):
    # Leer la imagen
    with open(input_path, "rb") as file:
        image_data = file.read()

    # Eliminar el fondo usando rembg
    result = rembg.remove(image_data)

    # Convertir el resultado en una imagen de Pillow
    image = Image.open(io.BytesIO(result)).convert("RGBA")

    # Crear un fondo blanco del mismo tamaño que la imagen original
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))

    # Combinar la imagen con el fondo blanco
    final_image = Image.alpha_composite(background, image).convert("RGB")

    # Guardar la imagen procesada en el directorio de salida
    final_image.save(output_path, "PNG")
    print(f"Imagen guardada en: {output_path}")


remove_background(
    "/home/ian/Downloads/WhatsApp Image 2024-11-27 at 12.33.04 AM.jpeg", "./imagen.jpeg"
)
