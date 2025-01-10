import streamlit as st
from ultralytics import YOLO
import cv2
import pytesseract
from transformers import MarianMTModel, MarianTokenizer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Spécifiez le chemin vers l'exécutable Tesseract (pour Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Charger le modèle YOLO
model = YOLO('D:/Down/best.pt')

# Charger le modèle de traduction (anglais -> français)
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translator = MarianMTModel.from_pretrained(model_name)

# Fonction de traduction
def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = translator.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).upper()  # Convertir en MAJUSCULES

# Fonction pour ajuster le texte au rectangle
def wrap_text(text, box_width, font, font_size):
    lines = []
    words = text.split()
    line = ""
    for word in words:
        temp_line = f"{line} {word}".strip()
        if font.getsize(temp_line)[0] > box_width:
            lines.append(line.strip())
            line = word
        else:
            line = temp_line
    if line:
        lines.append(line.strip())
    return lines

# Application Streamlit
st.title("Traduction de pages de manga")

uploaded_file = st.file_uploader("Téléchargez une page de manga", type=["jpg", "png"])

if uploaded_file is not None:
    # Charger l'image
    original_image = np.array(Image.open(uploaded_file))
    image_with_boxes = original_image.copy()  # Image pour les bulles entourées
    pil_image_with_translation = Image.fromarray(original_image)  # Utiliser PIL pour l'édition
    draw = ImageDraw.Draw(pil_image_with_translation)

    # Charger une police compatible Unicode
    font_path = "arial.ttf"  # Remplacez par le chemin de votre police si nécessaire
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)

    # Détecter les bulles avec YOLO
    results = model.predict(source=original_image, conf=0.25)

    for result in results:
        for box, conf in zip(result.boxes.xyxy, result.boxes.conf):  # Récupérer les coordonnées et scores
            if conf < 0.6:  # Filtrer les bulles avec un score < 0.6
                continue

            x1, y1, x2, y2 = map(int, box[:4])  # Convertir les coordonnées en entiers

            # Dessiner les rectangles autour des bulles sur image_with_boxes
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

            # Garder la taille originale pour l'extraction du texte
            cropped_bubble = original_image[y1:y2, x1:x2]
            cropped_bubble_gray = cv2.cvtColor(cropped_bubble, cv2.COLOR_BGR2GRAY)

            # Extraire le texte avec pytesseract
            text = pytesseract.image_to_string(cropped_bubble_gray).strip()

            if len(text) < 1:  # Ignore les textes trop courts
                continue

            # Traduire le texte extrait
            translated_text = translate_text(text)

            # Ajuster la taille des rectangles à 75% pour affichage
            box_width = x2 - x1
            box_height = y2 - y1
            reduced_x1 = x1 + box_width // 6
            reduced_y1 = y1 + box_height // 6
            reduced_x2 = x2 - box_width // 6
            reduced_y2 = y2 - box_height // 6

            # Diviser le texte traduit en lignes pour qu'il tienne dans le rectangle réduit
            wrapped_text = wrap_text(translated_text, reduced_x2 - reduced_x1, font, font_size)

            # Calculer la hauteur totale nécessaire pour le texte
            line_height = font.getsize("Test")[1] + 5
            total_text_height = len(wrapped_text) * line_height

            # Centrer verticalement
            current_y = reduced_y1 + (reduced_y2 - reduced_y1 - total_text_height) // 2

            # Dessiner un rectangle blanc derrière le texte (agrandi)
            padding_x = 6  # Ajouter un espace horizontal
            padding_y = 6  # Ajouter un espace vertical
            text_background_y1 = max(reduced_y1 - padding_y, y1)
            text_background_y2 = min(reduced_y2 + padding_y, y2)
            draw.rectangle(
                [reduced_x1 - padding_x, text_background_y1, reduced_x2 + padding_x, text_background_y2],
                fill="white"
            )

            # Ajouter le texte traduit ligne par ligne
            for line in wrapped_text:
                text_width, text_height = font.getsize(line)
                text_x = reduced_x1 + (reduced_x2 - reduced_x1 - text_width) // 2  # Centrer horizontalement
                draw.text((text_x, current_y), line, font=font, fill="black")
                current_y += line_height

    # Afficher les deux images
    st.image(image_with_boxes, caption="Image avec bulles détectées", use_column_width=True)
    st.image(pil_image_with_translation, caption="Image avec texte traduit", use_column_width=True)
