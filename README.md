Le projet consiste en une application web interactive qui permet aux utilisateurs d'uploader une page de manga en anglais (ou dans une autre langue prise en charge) et de recevoir une version annotée où les bulles sont traduites et modifiées avec le texte dans une autre langue, comme le français.

## L’application repose sur trois étapes principales :

Détection des bulles : Utilisation de modèles de détection d’objets pour identifier les zones contenant des bulles de texte.
Extraction du texte : Reconnaissance optique de caractères (OCR) pour extraire le texte des zones détectées.
Traduction du texte : Utilisation d’un modèle de traduction pour convertir le texte dans la langue souhaitée.
Étapes Clés du Projet :
# Préparation des données :

Un dataset d'images de manga a été constitué pour entraîner le modèle YOLO.
Les bulles de texte dans les pages de manga ont été annotées manuellement à l’aide d’outils comme LabelImg ou Roboflow.
Des fichiers d’annotation au format YOLO ont été générés, contenant les coordonnées des boîtes englobantes des bulles.
# Entraînement du modèle YOLO :

Un modèle YOLOv8 a été entraîné pour détecter les bulles dans les pages de manga.
Les ensembles d'entraînement, de validation, et de test ont été soigneusement séparés pour évaluer la performance du modèle.
Des augmentations de données ont été utilisées pour améliorer la robustesse du modèle face à des variations (taille, rotation, styles de texte).
Implémentation de l’OCR :

Le texte des bulles détectées a été extrait à l’aide de Tesseract OCR ou d'outils avancés comme EasyOCR.
Des prétraitements (binarisation, nettoyage des images) ont été appliqués pour améliorer la reconnaissance du texte.
Traduction des textes :

Les textes extraits ont été traduits à l’aide de modèles de traduction automatique, tels que Helsinki-NLP MarianMT disponibles via Hugging Face.
Le pipeline gère plusieurs langues, avec un focus sur l’anglais vers le français.
Annotation des images :

Les bulles ont été modifiées en remplaçant le texte d'origine par le texte traduit.
Des ajustements ont été apportés pour garantir que le texte traduit tienne dans les limites des bulles.
Déploiement de l’application :

L’application a été déployée avec Streamlit, un framework de création d’applications web interactives.
Les utilisateurs peuvent uploader une image et visualiser directement les résultats.
Fonctionnalités de l'Application :
Téléchargement d'une image :

Les utilisateurs peuvent uploader une image au format .jpg ou .png.
# Détection et traduction :

Les bulles de texte sont détectées et analysées automatiquement.
Les textes traduits remplacent les originaux.
Résultats visuels :

Une image annotée est affichée, avec les textes traduits insérés dans les bulles.
# Technologies Utilisées :
Détection d’objets :

YOLOv8, entraîné sur des images de manga annotées.
Reconnaissance de texte (OCR) :

Tesseract OCR pour extraire les textes des bulles.
Traduction :

Hugging Face Transformers (Helsinki-NLP MarianMT) pour la traduction automatique.
Développement de l’interface :

Streamlit pour la création d’une interface web interactive.
Bibliothèques Python :

OpenCV pour le traitement des images.
Pillow pour le chargement des images.
NumPy pour les calculs matriciels.
Transformers pour le NLP.
Résultats Attendus :
Performance du modèle YOLO :

Capacité de détecter avec précision les bulles dans des styles variés.
Un score de confiance minimum de 0.6 est requis pour exclure les fausses détections.
Qualité de la traduction :

Une traduction fluide et précise des textes.
Gestion des ajustements de mise en page pour les bulles contenant plusieurs lignes.
Expérience utilisateur :

Une interface intuitive permettant un accès facile à la traduction des pages.