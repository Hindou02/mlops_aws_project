# Apropos de Cette Repository : 
Ce repository contient les laboratoires du cours MLOps II – Automatisation du déploiement de modèles ML, destiné aux étudiants du SupNum (Institut Supérieur du Numérique), Master DEML M2, sous la supervision du Professeur Yehdhih ANNA.

Il illustre la mise en place d’une application simple (formulaire web) basée sur un modèle d’IA accessible via une API Flask. L’objectif est de construire un pipeline MLOps complet permettant l’automatisation du système : chaque validation de données déclenche l’entraînement d’un nouveau modèle à jour et son redéploiement. De plus, chaque push sur Git met à jour le code du serveur et relance également l’entraînement du modèle.

Le modèle suit un pipeline MLOps complet et est déployé sur une infrastructure AWS. Les technologies utilisées incluent :
Python, Flask, MLflow, DVC, AWS EC2, AWS IAM, AWS S3, HTML, CSS, JavaScript, Docker, GitHub Actions.

---

### Technologies & Tools

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-000000?style=for-the-badge&logo=mlflow&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-000000?style=for-the-badge&logo=dvc&logoColor=white)
![AWS EC2](https://img.shields.io/badge/AWS_EC2-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)
![AWS IAM](https://img.shields.io/badge/AWS_IAM-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)
![AWS S3](https://img.shields.io/badge/AWS_S3-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

---

# LAB 1 : Serveur MLFlow sur EC2 + Code Source + DVC Pipeline :
## 1.1. Mise de l'environement MLFLOW sur AWS : 

Le but de cet etape et des crée un serveur MLFlow, qui gère le registry de nos experiences MLs:

1. Allez dans votre compte AWS,
2. Crée un bucket s3, (rendre son accès public - pour minimiser les configuration nécessaire entre s3 et MLFLOW)
4. Crée un clé d'accès (Access Key) et  telecharger la, dans le fichier csv (vous trouvez votre Access_key_Id et Votre Secret_key).
5. Crée une machine Ec2 (type medium, OS:ubuntu), crée un key paire pour cette machine. donnez lui un accès Custom Port 5000 avec source 0.0.0.0/0 (tous le monde), 
6. sur cette machine on doit configurer notre MLFLOW avec les commandes suivantes (y inclut l'installation de quelques dependances) : 
```bash
sudo apt update
sudo apt install python3-pip
sudo apt install pipenv
sudo apt install virtualenv
mkdir mlflow
cd mlflow
pipenv install mlflow
pipenv install awscli
pipenv install boto3
pipenv shell
```
6. Maintenant configurer les accès à s3 pour MLflow depuis la machine (grace au clé que vous avez crée pour l'utilisateur dans l'etape 4)

```bash
# Set aws credentials
aws configure
```

7. Maintenant vous pouvez lancer mlflow dans cette machine en y précisant le nom de votre bucket (NB: le bucket sera utilisé pour stoker le cashe de MLFLOW : les runs, les experiments, les artefacts loggué, les models ...)

```bash
    # run mlflow server to be accessible globaly
    mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root s3://YOUR_BUCKET_NAME --allowed-hosts "*" --cors-allowed-origins "*" 
```

6. [Conditionel] Si vous avez skipper le custom port 5000 accès vous pouvez allez dans la liste des instances EC2, clicker sur l'ID de votre instance, et dans l'onglet Security clicker sur le nom de votre security group,
et ajouter une nouvelle règle dans les inboud rules. 
_Custom - Port 5000 - 0.0.0.0/0_

8. Mlflow est running donc vous pouvez l'accéder depuis l'adresse le lien `http://ADDRESS_IP_EC2:5000`

9. Si vous avez stopper la machine et vous avez connecter à nouveau il faut reactiver l'environement pipenv et relancer votre mlflow.
> pipenv est un gestionaire d'environement python basé sur le dossier (dans ce cas "mlflow") ce n'est pas commes les autres packageurs d'environnement qui peuvent etre activé globalement avec un chemin absolue.
```bash
    cd mlflow
    pipenv shell
    mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root s3://YOUR_BUCKET_NAME --allowed-hosts "*" --cors-allowed-origins "*" 
    cd mlflow
    pipenv shell
    mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root s3://lab1-s3-yana --allowed-hosts "*" --cors-allowed-origins "*" 



    # to keeps from stopping after closing the terminal you can use this command
    cd mlflow
    pipenv shell
    nohup mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root s3://lab1-s3-yana --allowed-hosts "*" --cors-allowed-origins "*" > mlflow.log 2>&1 &
```

10. Maintenant vous pouvez exploiter MLFlow pour faire le suivi de vos modèles (en précisant le tracking uri dans votre code ) :

```python
import mlflow
mlflow.set_tracking_uri("http://ADDRESS_IP_EC2:5000")
```

> TRES BIEN VOUS AVEZ COMPLETE CETTE ETAPE.

## 1.2.Repository git et Code de départ :

Le but de cet etape est d'avoir un code de départ qui simule quelques etapes classique dans le processus ML (preprocess, train, evaluate) (des étapes qui sont utilisé dans le 'Building' de modèles.)

> CI Traditionel : Code Change --> Build --> Unit Tests --> Artifact
> ML CI : Code Change OU **Data Change** --> Envirnoment Setup --> Dvc Repro (lance toutes les phases) --> MLFlow Logging (Registry)

> Dans le DevOps traditionel uniquement le code change (git) peut changer le système (l'artéfact) finale, alors que dans le MLOps on 'a pas uniquement le code change mais aussi les data change (dvc).

> Ceci implique que nous avons un outil pour le versionning code --> Git

> Et un outil de versionning de données --> dvc

11. Crée un dossier de travail et cloner cette repository dedans
  `git clone : git@github.com:yehdihouldanna/mlops_aws.git`

12. Allez dans votre compte github et crée une nouvelle repository (laisser la vierge) (`github/username/YOUR_REPOSITORY_NAME`)

13. Dans le dossier du repository cloné (etape 11) : Initer git et mettez le remote à votre repository, et pusher le code cloné sur votre repository

```bash 
git init
git remote add origin git@github.com:USERNAME/YOUR_REPOSITORY_NAME.git
git add .
git commit -m "first commit : Starting Code"
git push
```

## 1.3. Data Originale (Source de données) :
Pour similer une situation du monde réel avec une source de données externe dynamique nous allons crée sur notre bucket s3 un dossier dédié pour les données raw (ceci pourra venir de plusieurs sources).

Dans votre bucket S3 crée les dossier ``data/raw`` et uploader votre fichier ``diabetes.csv``
en meme temps crée aussi le dossier ``data/processed``

Donc vous devrez avoir ce chemin dans votre bucket s3
`s3://YOUR_BUCKET_NAME/data/raw/diabetes.csv`

> Dans le monde réel ceci pourra simuler le resultat d'un ETL lourd

## 1.4. Repository dvc sur s3 :

> Il existe des outils pour faire le repository dvc, cependant il faut comprendre que le dvc crée un fichier d'historique des metadata des données (elle ne stocke pas les données, mais chaque push de dvc correspond à un etat 'metadata' de données, elle peut contenir le path, sans contenir les données eux memes.)
> Cependant nous allons heberger notre repository dvc sur s3 c'est simple comme ça.
> Du coup on a deux chose à garder sur notre s3, nos données de base (n'a rien avoir avec dvc nécessairement, on peut avoir des données qui sont misees à jour journalierement)

14. initier dvc avec `dvc init`
15. crée une repository dvc  (attention : il faut crée cette dossier dans le meme dossier que votre git.)
(La ceci est le store c'est à dire il contient notre tree dvc)

```bash
    dvc init
    conda install dvc[s3]

    # 1. Define where DVC should store its actual "packages" (The Cache)
    dvc remote add -d dvcstore s3://YOUR_BUCKET_NAME

    # 2. Tell DVC to use your AWS credentials
    dvc remote modify dvcstore access_key_id   VOTRE_ACCESS_KEY_ID
    dvc remote modify dvcstore secret_access_key VOTRE_SECRET_ACCESS_KEY
```
16. Ajouter la source de données à tracker : 

```bash
    dvc import-url s3://YOUR_BUCKET_NAME/data/ .
    dvc import-url s3://lab1-s3-yana/data/ .
    #dvc import-url --to-remote s3://YOUR_BUCKET_NAME/data/ .
    dvc push # now you can check on your s3 to see the result
```
> --to-remote permet de n'a pas avoir une copie local du fichier dans le dossier du code (la copie sera telechargé ailleur et son hash md5 sera claculé et sera supprimé après), si vous l'enlever le fichier sera dans votre dossier data

## 1.5. Adjuster votre fichier params :

ex:
```yaml
preprocess:
  input: "s3://lab1-s3-yana/data/raw/diabetes.csv". # Original data source
  output: "data/processed/data.csv" 

train : 
  data : data/processed/data.csv
  model_path : models/model.pkl
  random_state : 42
  n_estimators: 100
  max_depth: 5

mlflow: 
  MLFLOW_TRACKING_URI : http://EC2-PUBLIC-IP-ADRESS:5000
aws :
  aws_access_key_id: YOUR_ACCESS_KEY
  aws_secret_access_key: YOUR_SECRET_ACESS_KEY
  region_name: AWS_REGION
```
> Ce fichier assume que votre code preprocessing (utilise des données diabetes csv local [ou traked avec dvc et pulled]), applique le preprocessing pour sauvergarder une version preprocessed sur s3, et utlise la version sauvegardé sur s3 pour faire le training.
> EN suite le fichier evaluate utilise le mlflow running sur ec2 pour recuperer la version nommé dans le code de votre modèle pour l'appeler.

### La commande qui automatise le lancement du pipeline dvc :
Maintenant avec la façon dont notre code est structuré en phases (preprocess et train)
nous pouvons lancer tous le flow quand on veut avec 

```bash
dvc repro
```


---

# LAB 2 : Déploiement d’un modèle MLflow avec Docker :

### 2.1. Objectif du laboratoire

Dans ce laboratoire, nous allons compléter notre pipeline MLOps en ajoutant une étape d’exploitation du modèle (inference / serving).

L’objectif est de :

1. Charger automatiquement le **dernier modèle enregistré dans le MLflow Registry**
2. Créer une **API de prédiction**
3. Conteneuriser cette API avec **Docker**
4. Déployer l’API sur une instance **EC2**
5. Communiquer avec le serveur **MLflow** via son **adresse IP**

Le modèle utilisé est :

```
Best_RandomForestClassifier
```

qui a été enregistré dans le **MLflow Model Registry** lors de l'étape d'entraînement.

---

## 2.2. Structure du projet

Nous allons ajouter un dossier pour l’exploitation du modèle.

Structure finale :

```
mlops_aws
│
├── data
├── models
├── params.yaml ## this is put in git ignore because it contains sensitive ID and Key, a reference of it is given instead
├── README.md
├── src
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│
├── serving
│   └── predict_api.py # this is the inference and server file
│
├── requirements.txt
└── Dockerfile
```

## 2.3. Création du script d’inférence
vous allez remarquer que vous avez un fichier :
serving/predict_api.py

qui contient le code du serveur d'exploitation :

- se connecter au **MLflow Tracking Server**
- charger le **dernier modèle enregistré**
- exposer une **API de prédiction**

```python
from fastapi import FastAPI
import pandas as pd
import mlflow
import os
import yaml

## Chargement des paramètres
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(BASE_DIR, "params.yaml")

params = yaml.safe_load(open(params_path))
mlflow_params = params['mlflow']

## Définition de l'adresse du serveur MLflow
mlflow.set_tracking_uri(mlflow_params["MLFLOW_TRACKING_URI"])

MODEL_NAME = "Best_RandomForestClassifier"
MODEL_VERSION = "latest"

print("Chargement du modèle depuis MLflow Registry...")

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)

print("Modèle chargé avec succès")

app = FastAPI(title="API de prédiction ML")

@app.get("/")
def home():
    return {"message": "API MLflow opérationnelle"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0])
    }
```
## 2.4. Configuration de la connexion au serveur MLflow

Nous allons communiquer avec MLflow via son **adresse IP**.
> Même si nous utilisons **la même instance EC2**, pour l'optimisation de ressources nous allons garder la meme approche pour montrer l'extensibilité sur plusieurs ressources.

Cela permet de garder une architecture compatible si les services qui sont déployés sur **plusieurs serveurs EC2**.

## 2.5. Les fichiers nécessaires : 
### requirements.txt

Vous allez remarquer un fichier requirements.txt contenant les librairies nécessaire pour le fonctionnnement du serveur basé sur FastAPI et exploitant le 'latest' model provenant de MLFlow registry.

requirements.txt
```
fastapi
uvicorn
pandas
scikit-learn
mlflow
boto3
s3fs
pyyaml
```

### Création du Dockerfile

Nous allons maintenant crée un fichier Dockerfile dans notre code pour pouvoir containeriser le code du serveur sur le cloud EC2.

Dockerfile

```
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "serving.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 2.6. Clonning du code du serveur : 

Actuellement nous clonons la totalité du repository pour la simplicité : 
mais un approche plus propre, on clonne uniquement les fichier nécessaire pour le serveur de prediciton (ces fichiers sont reduit au fichier suivant:)

```bash
.
├── params.yaml ## Ceci devrait etre recreé manuellement pour stocker les secret aws id et key, et aussi le tracking_uri
├── Dockerfile
├── requirements.txt
├── serving
│   └── predict_api.py
```

## 2.7. Construction de l’image Docker

### Installation Docker : 

Pour installer docker sur ubuntu : 

```bash 
sudo apt update && sudo apt upgrade -y
sudo apt install ca-certificates curl gnupg lsb-release -y
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
```

Pour verifier l'installation : 
```bash
sudo docker run hello-world
```

### Création de l'image docker : 

Sur notre Serveur (Machine EC2 dans ce cas)
on clone le code nécessaire dans un dossier : ex: home/username/
`git clone repo.git`

Recrée votre fichier params.yaml (ceci est dans git ignore à cause des secret id et key)
du coup après le clonage du dossier de code, on doit recrée le fichier params.yaml

et maintenant après le clone placer vous dans le dossier cloné :
et builder votre image : 

```bash
docker build -t mlflow-model-api .
```

## 2.8. Lancement du conteneur
IMPORTANT : Autoriser le port 8000 dans le sg (security group) de votre Instance EC2 :


Pour que notre container puisse communiquer avec s3 on doit le lancer avec les crédentiels :

```bash
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID="your_key" \
  -e AWS_SECRET_ACCESS_KEY="your_secret" \
  -e AWS_DEFAULT_REGION="your_region" \
  mlflow-model-api

```

### Testing the Server Backend API :

Tester avec curl :

```
curl -X POST http://EC2_PUBLIC_IP:8000/predict \
-H "Content-Type: application/json" \
-d '{
"Pregnancies": 6,
"Glucose": 148,
"BloodPressure": 72,
"SkinThickness": 35,
"Insulin": 0,
"BMI": 33.6,
"DiabetesPedigreeFunction": 0.627,
"Age": 50
}'
```

Réponse attendue :

```
{
 "prediction": 1
}
```

---

## 2.8. Architecture finale

Architecture du système :

```
                +----------------------+
                |   MLflow Tracking    |
                |       Server         |
                |    EC2 :5000         |
                +----------+-----------+
                           |
                           | accès au modèle
                           |
                   +-------v--------+
                   |   Docker API   |
                   |    FastAPI     |
                   |    port 8000   |
                   +-------+--------+
                           |
                           |
                     Requêtes utilisateur
```

Même si les deux services sont sur **le même EC2**, ils communiquent via **l’adresse IP**.

Cela permet facilement :

- de séparer les services sur plusieurs machines
- de scaler l’API indépendamment du serveur MLflow.

---

## 2.9. Configuration du Security Group

Vérifier que l’instance EC2 autorise les ports :

```
5000   MLflow Server
8000   API de prédiction
```

---

# LAB 3 : Automatisation des workflow CI/CD avec Github Actions :

### 3.1. Objectif du laboratoire

Dans ce laboratoire, nous allons ajouter des workflows (pipeline) github qui nous permettent d'automatiser la mise à jour du serveur après chaque push sur la branche principale du code, ou à la suite de chaque changement de données. (un push sur dvc)

Autrement dit on ferme la boucle — chaque changement de code ou de données déclenche automatiquement le pipeline.

L'objectif de ce lab est d'automatiser entièrement ces deux déclencheurs grâce à GitHub Actions :


| Composant |	Rôle |
|--- | ---|
| **`git push`**	| Reconstruit et redéploie automatiquement le conteneur Docker sur EC2 |
| **`dvc push`**	| Relance dvc repro, enregistre un nouveau modèle dans MLflow, redéploie l'API |


---

## 3.2. Structure du projet

Nous allons ajouter un dossier pour la configuration des pipeline avec github actions

Structure finale :

```bash
mlops_aws
│
├── .github/
│   └── workflows/
│       ├── deploy_code.yml      # Workflow 1 : déclencheur Git (code)
│       └── retrain_data.yml     # Workflow 2 : déclencheur DVC (données)
├── data
├── models
├── params.yaml ## this is put in git ignore because it contains sensitive ID and Key, a reference of it is given instead
├── README.md
├── src
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│
├── serving
│   └── predict_api.py # this is the inference and server file
│
├── requirements.txt
└── Dockerfile
```

## 3.3 Notions Clès :

CI/CD Traditionnel vs ML CI/CD
Dans le DevOps traditionnel, un seul déclencheur provoque le build :

**CI Traditionnel :**  `Code Change`  →  `Build`  →  `Tests`  →  `Artifact`

Dans le MLOps, il y a deux déclencheurs indépendants :

**ML CI (Code)  :**  `git push`    →  `Setup`  →  `docker build`  →  `redeploy`

**ML CI (Data)  :**  `dvc push`    →  `Setup`  →  `dvc repro`     →  `MLflow log`  →  `redeploy`

💡 Pourquoi deux pipelines ?
Un changement de code (ex: nouveau preprocessing) ne modifie pas les données.
Un changement de données (nouvelles observations) ne modifie pas le code.
Les deux doivent quand même produire un nouveau modèle et redéployer l'API.
GitHub Actions permet de définir un workflow séparé pour chaque déclencheur.

GitHub Actions — Notions Fondamentales
GitHub Actions est le moteur CI/CD intégré à GitHub. Il repose sur trois notions :

-	**Workflow :** un fichier YAML dans .github/workflows/ décrivant une automation.
- **Trigger (on:) :** l'événement qui déclenche le workflow (push, pull_request, workflow_dispatch…).
- **Job / Step :** un job est un ensemble d'étapes (steps) qui s'exécutent sur un runner.

Structure minimale d'un workflow :

```
name: Mon Workflow

on:
  push:
    branches: [main]

jobs:
  mon-job:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Ma commande
        run: echo 'Hello MLOps !'
```

#### GitHub Secrets — Gestion des Credentials
Les clés AWS et la clé SSH de votre instance EC2 ne doivent jamais apparaître dans le code. GitHub Secrets les stocke de façon chiffrée et les injecte dans le workflow comme variables d'environnement.

> 🔐 Règle de sécurité fondamentale : 
Ne jamais committer une clé AWS, un mot de passe ou une clé SSH dans git.
Même dans un repository privé : les secrets appartiennent à GitHub Secrets, pas au code.

## 3.4. Configuration des GitHub Secrets

GitHub Secrets permet de stocker des informations sensibles qui seront injectées dans les workflows sous forme de variables d'environnement, sans jamais les exposer dans les logs ni dans le code.

### 3.2.1. Accéder aux Secrets du Repository
5.	Allez sur votre repository GitHub.
6.	Cliquez sur Settings (onglet en haut à droite).
7.	Dans le menu gauche, choisissez Secrets and variables > Actions.
8.	Cliquez sur New repository secret pour chaque secret listé ci-dessous.

### 3.2.2. Liste des Secrets à créer
|Composant	| Rôle |
| --- | --- |
EC2_HOST |	Adresse IP publique de votre instance EC2 (ex: 54.210.x.x)
EC2_USER	| Utilisateur SSH de l'instance par défaut pour une novuelle machine EC2 c'est `ubuntu` (pour un OS Linux Ubuntu)
EC2_SSH_KEY	| Contenu complet du fichier .pem (clé privée SSH) , verifier dans vos téléchargements (chercher avec l'extension `*.pem`), c'est le fichier crée lors de la création de la machine EC2
AWS_ACCESS_KEY_ID	| Votre Access Key ID AWS
AWS_SECRET_ACCESS_KEY	|Votre Secret Access Key AWS
AWS_DEFAULT_REGION	| Région AWS de vos ressources (ex: eu-west-3)
MLFLOW_TRACKING_URI	| URL du serveur MLflow (ex: http://54.210.x.x:5000)


## 3.5. Workflow 1 — CI/CD Code (git push) :
après avoir crée un fichier vide deploy_code.yml (dans .github/workflows)

voici le contenu de ce fichier : 

```yml
name: Deploy Code - Build & Redeploy API

on:
  push:
    branches:
      - main
    paths-ignore:
      - 'README.md'
      - 'data/**'
      - '*.dvc'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Configure SSH key
        run: |
          mkdir -p ~/.ssh
          echo "${{ secrets.EC2_SSH_KEY }}" > ~/.ssh/ec2_key.pem
          chmod 600 ~/.ssh/ec2_key.pem
          ssh-keyscan -H ${{ secrets.EC2_HOST }} >> ~/.ssh/known_hosts

      - name: Deploy to EC2
        run: |
          ssh -i ~/.ssh/ec2_key.pem ${{ secrets.EC2_USER }}@${{ secrets.EC2_HOST }} << 'ENDSSH'

          # 1. Export variables so they exist in this specific SSH session
          export S3_BUCKET="${{ secrets.S3_BUCKET }}"
          export AWS_ACCESS_KEY_ID="${{ secrets.AWS_ACCESS_KEY_ID }}"
          export AWS_SECRET_ACCESS_KEY="${{ secrets.AWS_SECRET_ACCESS_KEY }}"
          export AWS_DEFAULT_REGION="${{ secrets.AWS_REGION }}"
          export MLFLOW_TRACKING_URI="${{ secrets.MLFLOW_TRACKING_URI }}"

          cd mlflow
          pipenv run nohup mlflow server --host 0.0.0.0 --port 5000 \
            --default-artifact-root ${{ secrets.S3_BUCKET }} \
            --allowed-hosts "*" --cors-allowed-origins "*" > mlflow.log 2>&1 &
          cd ..
          cd ~/mlops_aws
          git pull origin main

  
          # 2. Build the image 
          # Note: Only use --build-arg if your Dockerfile needs them during 'docker build'
          sudo docker build \
            --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
            --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
            -t mlflow-model-api .

          # 3. Stop & remove old container
          sudo docker stop api_container 2>/dev/null || true
          sudo docker rm api_container 2>/dev/null || true

          # 4. Launch new container - Passing the variables explicitly
          sudo docker run -d \
            --name api_container \
            -p 8000:8000 \
            -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
            -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
            -e AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
            -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
            mlflow-model-api
          ENDSSH
        env:
          AWS_KEY:    ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: ${{ secrets.AWS_REGION }}

      - name: Health check
        run: |
          sleep 30
          curl --fail http://${{ secrets.EC2_HOST }}:8000/ || exit 1
```

### Explication des blocs clés

**Le déclencheur (on:)**

Le workflow se déclenche uniquement sur les push vers main. L'option paths-ignore évite un redéploiement inutile quand seuls des fichiers .dvc ou le README changent.

**La configuration SSH**

Le secret `EC2_SSH_KEY` est écrit dans un fichier temporaire `~/.ssh/ec2_key.pem` sur le runner. Le `chmod 600` est obligatoire : SSH refuse d'utiliser une clé avec des permissions trop ouvertes.

**Le heredoc SSH**

Le bloc `<< 'ENDSSH'` permet d'envoyer plusieurs commandes en une seule connexion SSH. Le `|| true` sur les commandes `docker stop/rm` évite que le workflow échoue si le conteneur n'existe pas encore.

**Le Health Check**

Après déploiement, le workflow attend `15 secondes` puis appelle GET / sur l'API. Si l'API ne répond pas, le step échoue et GitHub marque le workflow en erreur — vous êtes notifié.

## 3.6. Workflow 2 — CI/CD Data (dvc push)

Ce workflow se déclenche manuellement (workflow_dispatch) ou sur un push contenant des fichiers .dvc modifiés. Il lance dvc repro sur l'EC2, enregistre le nouveau modèle dans MLflow, et redémarre l'API.

> **Pourquoi workflow_dispatch et non un cron ?** : 
Un cron lancerait le pipeline même si les données n'ont pas changé, ce qui gaspille des ressources et pollue le MLflow Registry.
workflow_dispatch permet de déclencher le pipeline manuellement depuis l'interface GitHub ou via l'API.
On ajoute également le déclencheur sur les fichiers .dvc modifiés pour une automatisation complète.

3.5.1. Schéma du Workflow

```bash
dvc push  (ou push d'un fichier .dvc modifié)
    │
    ▼
GitHub Actions Runner
    │
    ├── 1. Checkout du code
    ├── 2. SSH → EC2
    │        ├── git pull origin main
    │        ├── pip install dvc[s3]
    │        ├── dvc pull          (récupère les nouvelles données)
    │        ├── dvc repro         (preprocess → train → evaluate)
    │        │        ↳ MLflow log du nouveau modèle
    │        ├── docker stop / rm api_container
    │        └── docker run ... mlflow-model-api
    └── 3. Health check API
```

```yaml
name: Retrain - DVC Data Pipeline

on:
  workflow_dispatch:  # Déclenchement manuel depuis GitHub
    inputs:
      reason:
        description: 'Raison du retraining'
        required: false
        default: 'Mise à jour des données'

  push:
    branches:
      - main
    paths:
      - '**.dvc'       # Tout fichier .dvc modifié
      - 'dvc.lock'     # Ou le fichier lock DVC

jobs:
  retrain:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Configure SSH key
        run: |
          mkdir -p ~/.ssh
          echo "${{ secrets.EC2_SSH_KEY }}" > ~/.ssh/ec2_key.pem
          chmod 600 ~/.ssh/ec2_key.pem
          ssh-keyscan -H ${{ secrets.EC2_HOST }} >> ~/.ssh/known_hosts

      - name: Run DVC pipeline & retrain on EC2
        run: |
          ssh -i ~/.ssh/ec2_key.pem ${{ secrets.EC2_USER }}@${{ secrets.EC2_HOST }} << 'ENDSSH'
            
            # 1. Export variables so they exist in this specific SSH session
            export S3_BUCKET="${{ secrets.S3_BUCKET }}"
            export AWS_ACCESS_KEY_ID="${{ secrets.AWS_ACCESS_KEY_ID }}"
            export AWS_SECRET_ACCESS_KEY="${{ secrets.AWS_SECRET_ACCESS_KEY }}"
            export AWS_DEFAULT_REGION="${{ secrets.AWS_REGION }}"
            export MLFLOW_TRACKING_URI="${{ secrets.MLFLOW_TRACKING_URI }}"
            
            # Launch MLFLOW server And Navigate to project directory
            cd mlflow
            pipenv run nohup mlflow server --host 0.0.0.0 --port 5000 \
                --default-artifact-root ${{ secrets.S3_BUCKET }} \
                --allowed-hosts "*" --cors-allowed-origins "*" > mlflow.log 2>&1 &
            cd ..
            cd ~/mlops_aws
            git pull origin main

            # Activate conda/venv if needed
            source ~/.bashrc

            # Configure AWS for DVC
            export AWS_ACCESS_KEY_ID=$AWS_KEY
            export AWS_SECRET_ACCESS_KEY=$AWS_SECRET
            export AWS_DEFAULT_REGION=$AWS_REGION

            # Pull latest data from S3 via DVC
            dvc pull

            # Run full ML pipeline
            dvc repro

            echo 'Pipeline DVC terminé — nouveau modèle enregistré dans MLflow'

            
            # 2. Build the image 
            # Note: Only use --build-arg if your Dockerfile needs them during 'docker build'
            sudo docker build \
                --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
                --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
                -t mlflow-model-api .

            # 3. Stop & remove old container
            sudo docker stop api_container 2>/dev/null || true
            sudo docker rm api_container 2>/dev/null || true

            # 4. Launch new container - Passing the variables explicitly
            sudo docker run -d \
                --name api_container \
                -p 8000:8000 \
                -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
                -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
                -e AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
                -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
                mlflow-model-api
            ENDSSH
        env:
          AWS_KEY:    ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: ${{ secrets.AWS_DEFAULT_REGION }}
          MLFLOW_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

      - name: Health check
        run: |
          sleep 20
          curl --fail http://${{ secrets.EC2_HOST }}:8000/ || exit 1

```

## 3.7. Préparer l'EC2 pour recevoir les Workflows

Le runner GitHub Actions va se connecter en SSH à l'EC2 et exécuter des commandes. L'EC2 doit être prête à recevoir ces commandes sans intervention manuelle.

#### 3.6.1. Cloner le repository sur l'EC2
Si ce n'est pas déjà fait (Lab 2), clonez votre repository sur l'EC2 dans le dossier `~/mlops_aws` :

```bash 
ssh -i your-key.pem ubuntu@EC2_PUBLIC_IP

# Générer une clé SSH pour GitHub (si pas déjà fait)
ssh-keygen -t ed25519 -C 'ec2-mlops' -f ~/.ssh/id_ed25519 -N ''
cat ~/.ssh/id_ed25519.pub
# → Copiez cette clé publique dans GitHub > Settings > SSH Keys

# Cloner le repository
cd ~
git clone git@github.com:USERNAME/YOUR_REPOSITORY_NAME.git mlops_aws
```

#### 3.6.2. Installer les dépendances sur l'EC2
Les workflows vont exécuter dvc repro et docker build directement sur l'EC2. Assurez-vous que les outils sont installés :

```bash
# Python & pip
sudo apt update
sudo apt install python3-pip python3-venv -y

# DVC avec support S3
pip3 install 'dvc[s3]' --break-system-packages

# Docker (si pas déjà installé – voir Lab 2)
sudo docker --version

# Vérifier que git est configuré
git config --global user.email 'ec2@mlops.lab'
git config --global user.name 'EC2 MLOps Runner'
```

#### 3.6.3. Recreer le fichier params.yaml
Rappel : `params.yaml` est dans le `.gitignore` car il contient les secrets AWS. Il faut le recréer manuellement sur l'EC2 après le clonage :

```bash
cd ~/mlops_aws
nano params.yaml
```

Remplir son contenu à partir de cela :
```yml
preprocess:
  input:  's3://YOUR_BUCKET_NAME/data/raw/diabetes.csv'
  output: 'data/processed/data.csv'

train:
  data:          data/processed/data.csv
  model_path:    models/model.pkl
  random_state:  42
  n_estimators:  100
  max_depth:     5

mlflow:
  MLFLOW_TRACKING_URI: 'http://EC2_PUBLIC_IP:5000'

aws:
  aws_access_key_id:     YOUR_ACCESS_KEY_ID
  aws_secret_access_key: YOUR_SECRET_ACCESS_KEY
  region_name:           YOUR_REGION # should be eu-west-3 pour paris
```

## 3.8. Pousser les Workflows et Premier Test

#### 3.8.1. Commit et Push des fichiers de workflow
De retour sur votre machine locale, ajoutez les fichiers workflow à git et poussez :

```bash
git add .github/
git commit -m 'lab3: add GitHub Actions CI/CD workflows'
git push origin main
```

> Ce push va déclencher le Workflow 1 (deploy_code.yml) immédiatement !
Allez dans `GitHub` > `votre repository` > onglet `Actions`.
Vous verrez le workflow `'Deploy Code - Build & Redeploy API'` s'exécuter.
Cliquez dessus pour voir les logs en temps réel à chaque step.

#### 3.8.2. Suivre l'exécution dans GitHub Actions
-	Allez sur `GitHub` > `votre repository` > `onglet Actions.`
-	Cliquez sur le workflow en cours d'exécution.
-	Cliquez sur le job deploy pour voir les logs step by step.
-	Vérifiez que chaque step est vert (✓). En cas d'erreur, lisez le log du step rouge.

#### 3.8.3. Tester l'API après déploiement
Une fois le workflow terminé (toutes les étapes vertes), testez l'API déployée :

```bash
curl -X POST http://EC2_PUBLIC_IP:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

Réponse attendue :

```bash
{
  "prediction": 1
}
```

## 3.9 Tester le Workflow Data (Retraining)

On va simuler un changement de données en uploadant un fichier mis à jour sur S3, puis on déclenche le workflow de retraining.

#### 3.9.1 Simuler un changement de données

-	Ouvrez votre fichier diabetes.csv local.
-	Ajoutez quelques lignes de nouvelles observations (ou dupliquez quelques lignes existantes pour simuler de nouveaux enregistrements).
-	Uploadez le fichier mis à jour sur S3 :

`aws s3 cp diabetes.csv s3://YOUR_BUCKET_NAME/data/raw/diabetes.csv`

#### 3.9.2. Mettre à jour le tracking DVC
DVC doit être informé que la source de données a changé :

```bash
# Sur votre machine locale (ou directement sur l'EC2)
dvc update
dvc push

# Committer le nouveau .dvc file (qui contient le nouveau hash md5)
git add .
git commit -m 'data: update diabetes dataset with new observations'
git push origin main
```
>Ce push contient un fichier .dvc modifié
Le workflow retrain_data.yml se déclenche automatiquement (chemin paths: '**.dvc').
Il lancera dvc pull, puis dvc repro, puis enregistrera un nouveau modèle dans MLflow.
L'API sera redémarrée avec le nouveau modèle automatiquement.

#### 3.9.3. Déclencher manuellement le Workflow Data
Vous pouvez aussi déclencher le workflow de retraining manuellement depuis GitHub, sans modifier les données :
-	Allez sur GitHub > votre repository > onglet Actions.
-	Dans la liste des workflows à gauche, cliquez sur Retrain - DVC Data Pipeline.
-	Cliquez sur Run workflow (bouton bleu en haut à droite).
-	Entrez une raison (optionnel) et cliquez sur Run workflow.

Le workflow s'exécute exactement comme s'il avait été déclenché par un push de données.



aws s3 cp diabetes.csv s3://lab1-s3-yana/data/raw/diabetes.csv