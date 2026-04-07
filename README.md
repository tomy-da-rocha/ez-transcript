# 🎙️ EZ Transcript

**Transcrivez vos fichiers audio et vidéo en texte, directement sur votre ordinateur.**
Aucun compte, aucun cloud, aucune donnée envoyée sur Internet. 100% privé.

---

## ✨ Ce que fait cette application

- Glissez un fichier audio ou vidéo dans l'interface
- L'IA le transcrit automatiquement en texte
- Téléchargez le résultat en `.txt`, `.srt` (sous-titres) ou `.json`
- Tout se passe **sur votre machine**, rien ne part sur Internet

**Formats supportés :** MP3, MP4, WAV, OGG, FLAC, MKV, AVI, WEBM, M4A, et plus.

---

## 🚀 Installation

### Ce dont vous avez besoin

| Logiciel | Où le télécharger | Obligatoire ? |
|----------|-------------------|---------------|
| **Python 3.10+** | [python.org/downloads](https://www.python.org/downloads/) | ✅ Oui |
| **FFmpeg** | [ffmpeg.org/download](https://ffmpeg.org/download.html) | ✅ Oui |
| **Drivers NVIDIA** (pour GPU) | [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) | Recommandé |

> 💡 **Pas de GPU NVIDIA ?** Pas de problème ! L'application fonctionne aussi en mode CPU,
> c'est simplement un peu plus lent.

### Installer FFmpeg

**Windows :**
1. Téléchargez FFmpeg depuis [gyan.dev/ffmpeg](https://www.gyan.dev/ffmpeg/builds/) (version "essentials")
2. Extrayez l'archive
3. Ajoutez le dossier `bin` au PATH Windows
   (ou placez `ffmpeg.exe` dans le même dossier que ce projet)

**Linux (Ubuntu/Debian) :**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS :**
```bash
brew install ffmpeg
```

### Installer le projet

1. Téléchargez ou clonez ce dossier sur votre ordinateur
2. C'est tout ! Le script de démarrage s'occupe du reste

---

## ▶️ Utilisation

### Windows

Double-cliquez sur **`start.bat`**

### Linux / macOS

Ouvrez un terminal dans le dossier et tapez :
```bash
chmod +x start.sh   # (première fois seulement)
./start.sh
```

### Ce qui se passe ensuite

1. Le script vérifie que Python et FFmpeg sont installés
2. Il crée un environnement virtuel et installe les dépendances (première fois uniquement)
3. Il détecte votre GPU (si vous en avez un)
4. Le navigateur s'ouvre automatiquement sur l'interface
5. **C'est prêt !**

---

## 📖 Comment l'utiliser

### Étape 1 — Choisir un fichier
Glissez-déposez votre fichier audio ou vidéo dans la zone prévue,
ou cliquez pour parcourir vos fichiers.

### Étape 2 — Aperçu et options
- Écoutez/visionnez un aperçu du fichier
- Choisissez de tout transcrire ou sélectionnez des segments précis
- Optionnel : choisissez la langue (sinon, détection automatique)

### Étape 3 — Transcription
Cliquez sur **"Transcrire"** et suivez la progression en temps réel.

### Étape 4 — Résultat
- Lisez le texte directement dans l'interface
- Copiez-le dans le presse-papier
- Exportez en `.txt`, `.srt` ou `.json`

---

## 🧠 Fonctionnement technique

L'application choisit automatiquement la meilleure configuration selon votre matériel :

| VRAM GPU | Modèle Whisper | Qualité |
|----------|---------------|---------|
| ≥ 10 Go | `large-v3` | Excellente |
| ≥ 6 Go | `medium` | Très bonne |
| ≥ 4 Go | `small` | Bonne |
| CPU uniquement | `base` ou `small` | Correcte |

**Moteur :** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2)
— le meilleur équilibre vitesse/qualité pour la transcription locale.

Si le GPU manque de mémoire en cours de traitement,
l'application bascule automatiquement en mode CPU sans planter.

---

## 📁 Structure du projet

```
transcribe-local/
├── backend/
│   ├── main.py            # Serveur web (FastAPI)
│   ├── transcriber.py     # Moteur de transcription IA
│   ├── gpu_utils.py       # Détection GPU et sélection du modèle
│   ├── file_handler.py    # Gestion des fichiers et FFmpeg
│   └── models/            # Modèles IA téléchargés (cache local)
├── frontend/
│   ├── index.html         # Interface web
│   ├── app.js             # Logique de l'interface
│   └── style.css          # Apparence
├── output/                # Transcriptions sauvegardées
├── uploads/               # Fichiers uploadés (temporaire)
├── requirements.txt       # Dépendances Python
├── start.bat              # Lancement Windows
├── start.sh               # Lancement Linux/macOS
└── README.md              # Ce fichier
```

---

## ❓ Problèmes fréquents

### "FFmpeg non trouvé"
→ Installez FFmpeg et assurez-vous qu'il est dans votre PATH.
Testez en tapant `ffmpeg -version` dans un terminal.

### "Pas assez de mémoire GPU"
→ L'application bascule automatiquement sur le CPU.
Fermez d'autres applications gourmandes en VRAM si possible.

### "La transcription est lente"
→ Normal en mode CPU. Un GPU NVIDIA accélère le traitement de 5 à 20×.

### Le navigateur ne s'ouvre pas
→ Ouvrez manuellement : **http://localhost:7860**

---

## 📄 Licence

MIT — Libre d'utilisation, modification et redistribution.