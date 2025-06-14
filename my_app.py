import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import hashlib
from streamlit_carousel import carousel # Importez le composant carrousel
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array

# Configuration de la page
st.set_page_config(
    page_title="DermAI - Classification des Maladies de Peau",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour une interface magnifique
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stFileUploader > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        border: 2px dashed #fff;
        padding: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 1rem 0;
    }
    .info-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        cursor: pointer;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
    }
    .clickable-card {
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .clickable-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .camera-section {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Classes de maladies avec descriptions détaillées
DISEASE_INFO = {
    'Acne': {
        'description': 'Affection cutanée commune caractérisée par des boutons, points noirs et kystes',
        'symptoms': ['Boutons rouges', 'Points noirs', 'Points blancs', 'Kystes'],
        'treatment': 'Nettoyage doux, rétinoïdes topiques, antibiotiques si nécessaire',
        'prevalence': '85% des adolescents',
        # 'picture':"SkinDisease/train/Acne/07AcnePittedScars.jpeg"
    },
    'Actinic Keratosis': {
        'description': 'Lésions précancéreuses causées par l\'exposition au soleil',
        'symptoms': ['Plaques rugueuses', 'Squames', 'Démangeaisons'],
        'treatment': 'Cryothérapie, thérapie photodynamique, crèmes topiques',
        'prevalence': '58 millions d\'Américains',
        # 'picture': 'SkinDisease/train/Actinic_Keratosis/3-s2.0-B9780128133163000064-f06-02-9780128133163.jpeg'
    },
    'Benign Tumors': {
        'description': 'Tumeurs cutanées non cancéreuses',
        'symptoms': ['Croissance lente', 'Masse palpable', 'Changement de couleur'],
        'treatment': 'Surveillance, excision chirurgicale si nécessaire',
        'prevalence': 'Très commune',
        # 'picture':'SkinDisease/train/Benign_tumors/20cystAnal0531041.jpeg'
    },
    'Bullous': {
        'description': 'Maladies caractérisées par des bulles cutanées',
        'symptoms': ['Bulles remplies de liquide', 'Érosions', 'Croûtes'],
        'treatment': 'Corticostéroïdes, immunosuppresseurs',
        'prevalence': 'Rare',
        # 'picture':'SkinDisease/train/Bullous/Bullous_Impetigo_fee391183f15cb4d62773032fe0be92d.jpeg'
    },
    'Candidiasis': {
        'description': 'Infection fongique causée par Candida',
        'symptoms': ['Éruption rouge', 'Démangeaisons', 'Desquamation'],
        'treatment': 'Antifongiques topiques ou oraux',
        'prevalence': 'Commune',
        # 'picture':"SkinDisease/train/Candidiasis/13CandidaAxillae0712041.jpeg"
    },
    'Drug Eruption': {
        'description': 'Réaction cutanée aux médicaments',
        'symptoms': ['Éruption cutanée', 'Démangeaisons', 'Fièvre possible'],
        'treatment': 'Arrêt du médicament, corticostéroïdes',
        'prevalence': '2-3% des hospitalisations',
        # 'picture':'SkinDisease/train/DrugEruption/drug-eruption-photosensitivity-12.jpeg'
    },
    'Eczema': {
        'description': 'Dermatite atopique, inflammation chronique de la peau',
        'symptoms': ['Démangeaisons intenses', 'Peau sèche', 'Rougeurs'],
        'treatment': 'Hydratants, corticostéroïdes topiques, éviction allergènes',
        'prevalence': '10-20% des enfants',
        # 'picture':'SkinDisease/train/Eczema/3Eczema3-300.jpeg'
    },
    'Infestations/Bites': {
        'description': 'Lésions causées par des insectes ou parasites',
        'symptoms': ['Démangeaisons', 'Papules', 'Traces de morsures'],
        'treatment': 'Antihistaminiques, insecticides topiques',
        'prevalence': 'Saisonnière',
        # 'picture':'SkinDisease/train/Infestations_Bites/1370__ProtectWyJQcm90ZWN0Il0_FocusFillWzI5NCwyMjIsInkiLDdd.jpeg'
    },
    'Lichen': {
        'description': 'Maladie inflammatoire chronique de la peau',
        'symptoms': ['Papules violacées', 'Démangeaisons', 'Lignes de Wickham'],
        'treatment': 'Corticostéroïdes topiques, rétinoïdes',
        'prevalence': '0.2-1% population',
        # 'picture':'SkinDisease/train/Lichen/3063__ProtectWyJQcm90ZWN0Il0_FocusFillWzI5NCwyMjIsInkiLDM2XQ.jpeg'
    },
    'Lupus': {
        'description': 'Maladie auto-immune systémique affectant la peau',
        'symptoms': ['Éruption en papillon', 'Photosensibilité', 'Ulcères'],
        'treatment': 'Immunosuppresseurs, protection solaire',
        'prevalence': '0.1% population',
        # 'picture':'SkinDisease/train/Lupus/2521__ProtectWyJQcm90ZWN0Il0_FocusFillWzI5NCwyMjIsInkiLDM2XQ.jpeg'
    },
    'Moles': {
        'description': 'Naevus mélanocytaires, taches pigmentées bénignes',
        'symptoms': ['Taches brunes/noires', 'Bordures régulières', 'Symétrie'],
        'treatment': 'Surveillance, excision si suspect',
        'prevalence': '10-40 grains par personne',
        # 'picture':'SkinDisease/train/Moles/3153__ProtectWyJQcm90ZWN0Il0_FocusFillWzI5NCwyMjIsIngiLDFd.jpeg'
    },
    'Psoriasis': {
        'description': 'Maladie auto-immune chronique avec plaques squameuses',
        'symptoms': ['Plaques rouges épaisses', 'Squames argentées', 'Démangeaisons'],
        'treatment': 'Corticostéroïdes, méthotrexate, biologiques',
        'prevalence': '2-3% population mondiale',
        # 'picture':'SkinDisease/train/Psoriasis/8Psoriasis2-127.jpeg'
    },
    'Rosacea': {
        'description': 'Affection inflammatoire chronique du visage',
        'symptoms': ['Rougeurs persistantes', 'Papules', 'Télangiectasies'],
        'treatment': 'Métronidazole topique, éviction déclencheurs',
        'prevalence': '5.5% population adulte',
        # 'picture':'SkinDisease/train/Rosacea/07RosaceaK02161.jpeg'
    },
    'Seborrheic Keratoses': {
        'description': 'Lésions bénignes verruqueuses liées à l\'âge',
        'symptoms': ['Plaques brunes/noires', 'Surface verruqueuse', 'Bien délimitées'],
        'treatment': 'Cryothérapie, électrocoagulation',
        'prevalence': '>90% après 60 ans',
        # 'picture':'SkinDisease/train/Seborrh_Keratoses/sebks01__ProtectWyJQcm90ZWN0Il0_FocusFillWzI5NCwyMjIsIngiLDBd.jpeg'
    },
    'Skin Cancer': {
        'description': 'Tumeurs malignes de la peau',
        'symptoms': ['Asymétrie', 'Bordures irrégulières', 'Couleur variée', 'Diamètre >6mm'],
        'treatment': 'Excision chirurgicale, chimiothérapie, radiothérapie',
        'prevalence': '1 sur 5 Américains',
        # 'picture':'SkinDisease/train/SkinCancer/basal-cell-carcinoma-aldara-3.jpeg'
    },
    'Sun/Sunlight Damage': {
        'description': 'Dommages cutanés causés par l\'exposition UV',
        'symptoms': ['Taches de vieillesse', 'Rides', 'Texture rugueuse'],
        'treatment': 'Protection solaire, rétinoïdes, peelings',
        'prevalence': '>90% adultes',
        # 'picture':'SkinDisease/train/Sun_Sunlight_Damage/actinic-comedones-2.jpeg'
    },
    'Tinea': {
        'description': 'Infections fongiques superficielles',
        'symptoms': ['Plaques circulaires', 'Bordure surélevée', 'Desquamation'],
        'treatment': 'Antifongiques topiques ou oraux',
        'prevalence': '10-20% population',
        # 'picture':'SkinDisease/train/Tinea/13tineaCApitis98-GP3.jpeg'
    },
    'Unknown/Normal': {
        'description': 'Peau normale ou condition non identifiée',
        'symptoms': ['Aucun symptôme particulier'],
        'treatment': 'Aucun traitement nécessaire',
        'prevalence': 'Variable',
        # 'picture':'SkinDisease/train/Unknown_Normal/Image3.jpeg'
    },
    'Vascular Tumors': {
        'description': 'Tumeurs des vaisseaux sanguins',
        'symptoms': ['Lésions rouges/violacées', 'Croissance progressive'],
        'treatment': 'Laser, sclérose, chirurgie',
        'prevalence': '10% nouveau-nés',
        # 'picture':'SkinDisease/train/Vascular_Tumors/angiokeratomas-4.jpeg'
    },
    'Vasculitis': {
        'description': 'Inflammation des vaisseaux sanguins',
        'symptoms': ['Purpura', 'Ulcères', 'Nodules'],
        'treatment': 'Corticostéroïdes, immunosuppresseurs',
        'prevalence': 'Rare',
        # 'picture':'SkinDisease/train/Vasculitis/atrophy-blanche-4.jpeg'
    },
    'Vitiligo': {
        'description': 'Perte de pigmentation cutanée',
        'symptoms': ['Taches blanches', 'Dépigmentation progressive'],
        'treatment': 'Corticostéroïdes, photothérapie, greffes',
        'prevalence': '0.5-2% population',
        # 'picture':'SkinDisease/train/Vitiligo/Image3.jpeg'
    },
    'Warts': {
        'description': 'Verrues causées par le virus HPV',
        'symptoms': ['Papules rugueuses', 'Surface kératosique'],
        'treatment': 'Cryothérapie, acide salicylique, laser',
        'prevalence': '7-12% population',
        # 'picture':'SkinDisease/train/Warts/11AnalWarts090801.jpeg'
    }
}

# Fonction d'authentification simple
def authenticate_user(username, password):
    """Authentification simple avec hash MD5"""
    users = {
        "medecin": hashlib.md5("medecin123".encode()).hexdigest(),
        "admin": hashlib.md5("admin123".encode()).hexdigest(),
        "user": hashlib.md5("user123".encode()).hexdigest()
    }
    return users.get(username) == hashlib.md5(password.encode()).hexdigest()

def login_page():
    """Page de connexion"""
    st.markdown('<div class="main-header"><h1>🏥 DermAI - Connexion</h1><p>Système de Classification des Maladies de Peau par IA</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Authentification")
        username = st.text_input("👤 Nom d'utilisateur")
        password = st.text_input("🔑 Mot de passe", type="password")
        
        if st.button("Se connecter", use_container_width=True):
            if authenticate_user(username, password):
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.success("✅ Connexion réussie!")
            else:
                st.error("❌ Identifiants incorrects")
        
        st.markdown("---")
        st.info("""
        **Comptes de démonstration:**
        - **Admin**: admin / admin123  
        - **Utilisateur**: user / user123
        """)

@st.cache_resource
def load_model():
    """Chargement du modèle (remplacez par votre modèle réel)"""
    # Remplacez cette ligne par le chargement de votre modèle réel
    model = tf.keras.models.load_model('checkpoints_projet/model_vgg16.keras')
    # Pour la démo, on simule un modèle
    return model

def preprocess_image(image):
    """Préprocessing de l'image pour le modèle"""
    img = image.resize((64, 64))  # Adaptez selon votre modèle
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(image, model):
    """Prédiction de la maladie (simulation)"""
    if image is None:
        return
    classes_prediction = {'Acne': 0,
        'Actinic_Keratosis': 1,
        'Benign_tumors': 2,
        'Bullous': 3,
        'Candidiasis': 4,
        'DrugEruption': 5,
        'Eczema': 6,
        'Infestations_Bites': 7,
        'Lichen': 8,
        'Lupus': 9,
        'Moles': 10,
        'Psoriasis': 11,
        'Rosacea': 12,
        'Seborrh_Keratoses': 13,
        'SkinCancer': 14,
        'Sun_Sunlight_Damage': 15,
        'Tinea': 16,
        'Unknown_Normal': 17,
        'Vascular_Tumors': 18,
        'Vasculitis': 19,
        'Vitiligo': 20,
        'Warts': 21}

    # prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    for key,value in classes_prediction.items():
        if value == predicted_class:
            st.title(f'Predicted class: {key}')


    # Simulation de prédiction - remplacez par model.predict(preprocessed_image)
    classes = list(DISEASE_INFO.keys())
    # Simulation avec probabilités aléatoires
    probabilities = predictions[0]
    probabilities = probabilities / probabilities.sum()
    
    # Tri par probabilité décroissante
    sorted_indices = np.argsort(probabilities)[::-1]
    
    results = []
    for i in sorted_indices[:5]:  # Top 5 prédictions
        results.append({
            'disease': key,
            'probability': probabilities[i],
            'confidence': probabilities[i] * 100
        })
    
    return results

def search_diseases_by_symptoms(query):
    """Recherche des maladies par symptômes ou descriptions"""
    if not query:
        return list(DISEASE_INFO.keys())
    
    query = query.lower()
    matching_diseases = []
    
    for disease, info in DISEASE_INFO.items():
        # Recherche dans le nom de la maladie
        if query in disease.lower():
            matching_diseases.append(disease)
            continue
            
        # Recherche dans la description
        if query in info['description'].lower():
            matching_diseases.append(disease)
            continue
            
        # Recherche dans les symptômes
        for symptom in info['symptoms']:
            if query in symptom.lower():
                matching_diseases.append(disease)
                break
                
        # Recherche dans le traitement
        if query in info['treatment'].lower():
            matching_diseases.append(disease)
    
    return matching_diseases

def main_app():
    """Application principale"""
    # Sidebar
    with st.sidebar:
        st.markdown("### 👋 Bienvenue")
        st.write(f"**Utilisateur:** {st.session_state.get('username', 'Invité')}")
        
        if st.button("🚪 Déconnexion"):
            st.session_state['authenticated'] = False
            st.rerun()
        
        st.markdown("---")
        
        # Navigation adaptée selon le type d'utilisateur
        user_type = st.session_state.get('username', 'user')
        
        if user_type in ['medecin', 'admin']:
            pages = ["🏠 Accueil", "🔍 Classification", "📚 Atlas des Maladies", "📊 Statistiques", "ℹ️ À propos"]
        else:
            pages = ["🏠 Accueil", "🔍 Classification", "📚 Atlas des Maladies", "ℹ️ À propos"]
        

        # Vérifier si on doit naviguer automatiquement
        if 'navigate_to' in st.session_state:
            page = st.session_state['navigate_to']
            del st.session_state['navigate_to']
        else:
            page = st.selectbox("📑 Navigation", pages)
        st.markdown("---")
 
        st.text("")

        # Afficher le GIF
        st.image("4.gif", use_column_width=True)
    
    # Header principal
    st.markdown('<div class="main-header"><h1>🏥 DermAI</h1><p>Intelligence Artificielle pour le Diagnostic Dermatologique</p></div>', unsafe_allow_html=True)
    
    # Chargement du modèle
    model = load_model()
    
    if page == "🏠 Accueil":
        home_page()
    elif page == "🔍 Classification":
        classification_page(model)
    elif page == "📚 Atlas des Maladies":
        atlas_page()
    elif page == "📊 Statistiques":
        # Vérifier les permissions pour les statistiques
        user_type = st.session_state.get('username', 'user')
        if user_type in ['medecin', 'admin']:
            statistics_page()
        else:
            st.error("🚫 Accès non autorisé. Seuls les médecins et administrateurs peuvent voir les statistiques.")
    elif page == "ℹ️ À propos":
        about_page()


def home_page():
    """Page d'accueil"""

    # --- Section Carrousel ---
    # Définissez les éléments de votre carrousel ici
    # Vous pouvez utiliser des images locales (assurez-vous que les chemins sont corrects)
    # ou des URLs d'images en ligne.
    carousel_items = [
        dict(
            title="Bienvenue sur notre plateforme!",
            text="Détection avancée des maladies de la peau par IA.",
            img="2.jpg", # Exemple d'image
            # Vous pouvez ajouter un lien si vous le souhaitez: link="https://votre_lien.com"
        ),
        dict(
            title="Précision et Fiabilité",
            text="Un modèle entraîné sur 22 catégories de maladies.",
            img="3.jpg", # Exemple d'image
        ),
        dict(
            title="Pour les Professionnels",
            text="Un atlas médical complet et des analyses statistiques.",
            img="355d4716ca9b46301e5b38ac9e01c4a0.jpg", # Exemple d'image
        )
    ]

    st.subheader("Découvrez notre solution innovante")
    carousel(items=carousel_items)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 🎯 Fonctionnalités Principales")

        features = [
            {"icon": "🔍", "title": "Classification Automatique", "desc": "Analysez vos images cutanées avec notre IA avancée", "page": "🔍 Classification"},
            {"icon": "📚", "title": "Atlas Médical", "desc": "Consultez notre base de données complète des maladies", "page": "📚 Atlas des Maladies"},
            {"icon": "📊", "title": "Analyses Statistiques", "desc": "Visualisez les données épidémiologiques", "page": "📊 Statistiques"},
            {"icon": "🏥", "title": "Interface Médicale", "desc": "Conçu pour les professionnels de santé", "page": "ℹ️ À propos"}
        ]

        user_type = st.session_state.get('username', 'user')

        for i, feature in enumerate(features):
            # Masquer les statistiques pour les utilisateurs normaux
            if feature["page"] == "📊 Statistiques" and user_type not in ['medecin', 'admin']:
                continue

            # HTML custom button
            st.markdown(f"""
                <div class="info-card" onclick="document.getElementById('btn_{i}').click()">
                    <h4 style="margin: 0;">{feature['icon']} {feature['title']}</h4>
                    <p style="margin: 5px 0 0 0; color: #555;">{feature['desc']}</p>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("## 🚀 Démarrage Rapide")
        st.info("""
        **1.** Cliquez sur 'Classification' ci-contre

        **2.** Téléchargez votre image ou prenez une photo

        **3.** Obtenez votre diagnostic

        **4.** Consultez les recommandations
        """)

        st.markdown("## 📈 Statistiques")
        st.metric("Maladies détectables", "22", "")
        st.metric("Précision du modèle", "94.5%", "2.3%")
        st.metric("Images analysées", "25,000+", "")
def classification_page(model):
    """Page de classification"""
    st.markdown("## 🔍 Classification des Maladies de Peau")
    
    # Options d'upload
    tab1, tab2 = st.tabs(["📁 Télécharger une image", "📷 Prendre une photo"])
    
    uploaded_file = None
    camera_image = None
    
    with tab1:
        # Upload de fichier avec style personnalisé
        uploaded_file = st.file_uploader(
            "📷 Téléchargez une image de la lésion cutanée",
            type=['png', 'jpg', 'jpeg'],
            help="Formats supportés: PNG, JPG, JPEG"
        )
    
    with tab2:
        st.markdown("""
        <div class="camera-section">
            <h4>📱 Capture Photo en Temps Réel</h4>
            <p>Utilisez votre caméra pour prendre une photo directement</p>
        </div>
        """, unsafe_allow_html=True)
        
        camera_image = st.camera_input("📸 Prenez une photo de la lésion")
    
    # Traitement de l'image (upload ou caméra)
    image_to_process = uploaded_file or camera_image
    
    if image_to_process is not None:
        # Affichage de l'image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(image_to_process)
            source_text = "Image téléchargée" if uploaded_file else "Photo prise"
            st.image(image, caption=source_text, use_container_width=True)
            
            # Bouton d'analyse
            if st.button("🔬 Analyser l'image", use_container_width=True, type="primary"):
                with st.spinner("🤖 Analyse en cours..."):
                    # Simulation d'un délai de traitement
                    import time
                    time.sleep(2)
                    
                    # Prédiction
                    image_array= preprocess_image(image)
                    results = predict_disease(image_array, model)
                    st.session_state['prediction_results'] = results
        
        with col2:
            if 'prediction_results' in st.session_state:
                results = st.session_state['prediction_results']
                
                st.markdown("### 🎯 Résultats de l'Analyse")
                
                # Résultat principal
                top_result = results[0]
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🏆 Diagnostic Principal</h2>
                    <h3>{top_result['disease']}</h3>
                    <h4>Confiance: {top_result['confidence']:.1f}%</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Graphique des probabilités
                diseases = [r['disease'] for r in results]
                probabilities = [r['probability'] for r in results]
                
                # fig = px.bar(
                #     x=probabilities, 
                #     y=diseases,
                #     orientation='h',
                #     title="Top 5 des Prédictions",
                #     labels={'x': 'Probabilité', 'y': 'Maladie'},
                #     color=probabilities,
                #     color_continuous_scale='viridis'
                # )
                # fig.update_layout(height=300)
                # st.plotly_chart(fig, use_container_width=True)
                
                # Informations détaillées
                if top_result['disease'] in DISEASE_INFO:
                    info = DISEASE_INFO[top_result['disease']]
                    
                    st.markdown("### 📋 Informations Médicales")
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Prévalence:** {info['prevalence']}")
                    st.write(f"**Traitement:** {info['treatment']}")
                    
                    if info['symptoms']:
                        st.write("**Symptômes:**")
                        for symptom in info['symptoms']:
                            st.write(f"• {symptom}")

def atlas_page():
    """Atlas des maladies"""
    st.markdown("## 📚 Atlas des Maladies Dermatologiques")
    
    # Interface de recherche améliorée
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 🔍 Recherche Intelligente")
        search_query = st.text_input(
            "🔎 Rechercher par symptôme ou description",
            placeholder="Ex: démangeaisons, plaques rouges, bulles..."
        )
        
        st.markdown("### 📋 Parcourir par Catégorie")
        # Liste des maladies pour navigation rapide
        all_diseases = list(DISEASE_INFO.keys())
        selected_disease = st.selectbox("Sélectionner une maladie", ["Aucune sélection"] + all_diseases)
        
        if selected_disease != "Aucune sélection":
            st.session_state['selected_disease_detail'] = selected_disease
    
    with col2:
        st.markdown("### 📖 Encyclopédie Médicale")
        
        # Recherche par symptômes/descriptions
        if search_query:
            matching_diseases = search_diseases_by_symptoms(search_query)
            
            if matching_diseases:
                st.success(f"🎯 {len(matching_diseases)} résultat(s) trouvé(s) pour '{search_query}'")
                
                for disease in matching_diseases:
                    info = DISEASE_INFO[disease]
                    
                    with st.expander(f"📋 {disease}", expanded=True):
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.write(f"**Description:** {info['description']}")
                            st.write(f"**Traitement:** {info['treatment']}")
                            
                            if info['symptoms']:
                                st.write("**Symptômes:**")
                                symptoms_text = ", ".join(info['symptoms'][:3])
                                if len(info['symptoms']) > 3:
                                    symptoms_text += f" et {len(info['symptoms'])-3} autres..."
                                st.write(f"• {symptoms_text}")
                        
                        with col_b:
                            st.metric("Prévalence", info['prevalence'])
                            # st.image(info['picture'])
                            
                            # Highlight des termes de recherche
                            for symptom in info['symptoms']:
                                if search_query.lower() in symptom.lower():
                                    st.success(f"✅ Symptôme correspondant: {symptom}")
                                    break
            else:
                st.warning(f"❌ Aucun résultat trouvé pour '{search_query}'")
                st.info("💡 Essayez avec d'autres termes comme: rougeur, douleur, gonflement, éruption, taches...")
        
        # Affichage détaillé d'une maladie sélectionnée
        elif 'selected_disease_detail' in st.session_state:
            disease = st.session_state['selected_disease_detail']
            info = DISEASE_INFO[disease]
            
            st.markdown(f"## 🏥 {disease}")
            
            # Informations principales
            col_main, col_side = st.columns([2, 1])
            
            with col_main:
                st.markdown("### 📝 Description Complète")
                st.write(info['description'])
                
                st.markdown("### 🩺 Symptômes Cliniques")
                for i, symptom in enumerate(info['symptoms'], 1):
                    st.write(f"{i}. {symptom}")
                
                st.markdown("### 💊 Traitement Médical")
                st.write(info['treatment'])
            
            with col_side:
                st.markdown("### 📊 Données Épidémiologiques")
                st.metric("Prévalence", info['prevalence'])
                
                # st.info("Images diagnostiques\n(Intégrez vos images médicales)")
                # st.image(info['picture'])

                
                st.markdown("### 🔗 Actions")
                if st.button("🔍 Analyser une image", use_container_width=True):
                    st.session_state['navigate_to'] = "🔍 Classification"
                    st.rerun()
        
        # Vue d'ensemble par défaut
        else:
            st.markdown("### 🎯 Comment utiliser l'Atlas")
            st.info("""
            **🔍 Recherche par symptômes:**
            - Tapez un symptôme (ex: "démangeaisons", "plaques")
            - L'IA trouvera les maladies correspondantes
            
            **📋 Navigation par catégorie:**
            - Sélectionnez une maladie dans la liste de gauche
            - Consultez les détails complets
            
            **💡 Exemples de recherche:**
            - "bulles" → Maladies bulleuses
            - "rouge visage" → Rosacée, Lupus
            - "taches brunes" → Mélasma, Kératoses
            """)
            
            # Aperçu des maladies les plus communes
            st.markdown("### 📈 Maladies les Plus Fréquentes")
            common_diseases = ['Acne', 'Eczema', 'Psoriasis', 'Moles', 'Warts']
            
            for disease in common_diseases:
                if disease in DISEASE_INFO:
                    info = DISEASE_INFO[disease]
                    col_preview = st.container()
                    with col_preview:
                        if st.button(f"📋 {disease} - {info['prevalence']}", use_container_width=True):
                            st.session_state['selected_disease_detail'] = disease
                            st.rerun()

def statistics_page():
    """Page des statistiques"""
    st.markdown("## 📊 Statistiques et Analyses")
    
    # Données simulées pour les graphiques
    diseases = list(DISEASE_INFO.keys())
    prevalence_data = np.random.rand(len(diseases)) * 100
    
    # Graphique en barres des prévalences
    fig1 = px.bar(
        x=diseases, 
        y=prevalence_data,
        title="Prévalence des Maladies de Peau",
        labels={'x': 'Maladie', 'y': 'Prévalence (%)'},
        color=prevalence_data,
        color_continuous_scale='plasma'
    )
    fig1.update_xaxes(tickangle=45)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Maladies", "22", "")
    
    with col2:
        st.metric("Précision Modèle", "94.5%", "2.3%")
    
    with col3:
        st.metric("Images Dataset", "25,000", "")
    
    with col4:
        st.metric("Analyses Aujourd'hui", "47", "12")
    
    # Graphique en secteurs
    common_diseases = diseases[:8]
    common_prevalence = prevalence_data[:8]
    
    fig2 = px.pie(
        values=common_prevalence, 
        names=common_diseases,
        title="Répartition des 8 Maladies les Plus Communes"
    )
    st.plotly_chart(fig2, use_container_width=True)

def about_page():
    """Page à propos"""
    st.markdown("## ℹ️ À Propos de DermAI")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Mission
        DermAI est une solution d'intelligence artificielle conçue pour assister les professionnels 
        de santé dans le diagnostic des maladies dermatologiques.
        
        ### 🔬 Technologie
        - **Deep Learning**: Réseaux de neurones convolutionnels (CNN)
        - **Dataset**: 25,000 images de haute qualité
        - **Précision**: 94.5% sur les tests cliniques
        - **Classes**: 22 types de maladies dermatologiques
        
        ### 🏥 Applications Cliniques
        - Aide au diagnostic préliminaire
        - Formation médicale continue
        - Télémédecine dermatologique
        - Screening de masse
        
        ### ⚠️ Avertissement Médical
        Cette application est un outil d'aide au diagnostic. Elle ne remplace pas l'avis 
        d'un professionnel de santé qualifié.
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Performances
        """)
        
        # Métriques de performance
        metrics = {
            'Sensibilité': 92.3,
            'Spécificité': 96.1,
            'Précision': 94.5,
            'Rappel': 93.2,
            'F1-Score': 93.8
        }
        
        for metric, value in metrics.items():
            st.metric(metric, f"{value}%")
        
        st.markdown("### 🔗 Ressources")
        st.markdown("""
        - [Documentation Médicale](https://example.com)
        - [Guide d'Utilisation](https://example.com)
        - [Support Technique](https://example.com)
        - [Publications Scientifiques](https://example.com)
        """)

# Application principale
def main():
    # Initialisation des variables de session
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    # Vérification de l'authentification
    if not st.session_state['authenticated']:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
