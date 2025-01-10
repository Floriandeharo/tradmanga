import streamlit as st
from ultralytics import YOLO
import cv2
import pytesseract
from transformers import MarianMTModel, MarianTokenizer
import numpy as np
from PIL import Image
import pytesseract

# Spécifiez le chemin vers l'exécutable Tesseract (pour Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Pour Linux/MacOS, tesseract est souvent déjà dans PATH, donc pas besoin de configuration
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Exemple pour Linux

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
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



# Fonction pour ajuster le texte au rectangle
def wrap_text(text, box_width, font, font_scale, thickness):
    lines = []
    words = text.split()
    line = ""
    for word in words:
        temp_line = f"{line} {word}".strip()
        text_size = cv2.getTextSize(temp_line, font, font_scale, thickness)[0]
        if text_size[0] > box_width:
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
    image = np.array(Image.open(uploaded_file))

    # Détecter les bulles avec YOLO
    results = model.predict(source=image, conf=0.25)

    for result in results:
        for box, conf in zip(result.boxes.xyxy, result.boxes.conf):  # Récupérer les coordonnées et scores
            if conf < 0.6:  # Filtrer les bulles avec un score < 0.6
                continue

            x1, y1, x2, y2 = map(int, box[:4])  # Convertir les coordonnées en entiers

            # Garder la taille originale pour l'extraction du texte
            cropped_bubble = image[y1:y2, x1:x2]
            cropped_bubble_gray = cv2.cvtColor(cropped_bubble, cv2.COLOR_BGR2GRAY)

            # Extraire le texte avec pytesseract
            text = pytesseract.image_to_string(cropped_bubble_gray)

            # Traduire le texte extrait
            translated_text = translate_text(text)

            # Ajuster la taille des rectangles à 75% pour affichage
            box_width = x2 - x1
            box_height = y2 - y1
            reduced_x1 = x1 + box_width // 8
            reduced_y1 = y1 + box_height // 8
            reduced_x2 = x2 - box_width // 8
            reduced_y2 = y2 - box_height // 8

            # Diviser le texte traduit en lignes pour qu'il tienne dans le rectangle réduit
            font_scale = 0.5
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            reduced_box_width = reduced_x2 - reduced_x1
            wrapped_text = wrap_text(translated_text, reduced_box_width, font, font_scale, thickness)

            # Calculer la hauteur totale nécessaire pour le texte
            line_height = cv2.getTextSize("Test", font, font_scale, thickness)[0][1] + 5
            total_text_height = len(wrapped_text) * line_height

            # Ajuster la taille de la police si le texte dépasse verticalement
            if total_text_height > (reduced_y2 - reduced_y1):
                font_scale *= (reduced_y2 - reduced_y1) / total_text_height
                wrapped_text = wrap_text(translated_text, reduced_box_width, font, font_scale, thickness)

            # Supprimer l'ancien texte (remplir le rectangle réduit avec du blanc)
            cv2.rectangle(image, (reduced_x1, reduced_y1), (reduced_x2, reduced_y2), (255, 255, 255), thickness=-1)

            # Ajouter le texte traduit ligne par ligne dans le rectangle réduit
            current_y = reduced_y1 + (reduced_y2 - reduced_y1 - total_text_height) // 2  # Centrer verticalement
            for line in wrapped_text:
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                text_x = reduced_x1 + (reduced_box_width - text_size[0]) // 2  # Centrer horizontalement
                cv2.putText(image, line, (text_x, current_y), font, font_scale, (0, 0, 0), thickness)
                current_y += line_height

    # Sauvegarder et afficher l'image annotée
    annotated_image_path = 'annotated_image_with_translation.jpg'
    cv2.imwrite(annotated_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    st.image(image, caption="Image avec texte traduit", use_column_width=True)
