dvc repro

> **Why this is powerful:**
> mart Execution: If you change n_estimators in params.yaml and run dvc repro, DVC sees that the preprocess stage hasn't changed, so it skips it and only runs the train stage.

Consistency: It ensures that your MLflow tracking URI and AWS keys are always pulled from the central config.

# Etape finale versioning du resultat
Once dvc repro finishes, you will have a new models.pkl and a new state. You should lock this in:

```Bash
git add dvc.yaml dvc.lock params.yaml
git commit -m "Experiment: changed max_depth to 5"
dvc push
```

````
C'est quoi le dvc.lock ?
Si params.yaml est ton intention (ce que tu veux faire), le dvc.lock est la preuve (ce qui a été réellement exécuté). C'est un petit fichier texte que DVC génère automatiquement après un dvc repro.

À quoi ça sert concrètement ?
Le "Check-point" mathématique : Il enregistre le hash (l'empreinte numérique) exact de tes données, de ton code et de tes paramètres au moment T. Si tu changes ne serait-ce qu'une virgule dans ton CSV, le hash change et DVC le sait.

Éviter de recalculer (Gain de temps) : Quand tu tapes dvc repro, DVC compare l'état actuel de tes fichiers avec les hashs dans le .lock.

Identique ? Il saute l'étape (Skip).

Différent ? Il relance uniquement ce qui a changé.

Le Voyage dans le temps (Reproductibilité) : C'est le lien entre Git et S3. En faisant un git checkout d'une ancienne version, tu récupères l'ancien dvc.lock. DVC lit alors les hashs et va chercher sur S3 exactement le modèle et les données qui correspondent à ce commit précis.

Le flux de travail résumé :
Modifier tes paramètres dans params.yaml.

Exécuter dvc repro (ceci met à jour le dvc.lock).

Versionner : git add dvc.lock pour garder la trace du changement.

Stocker : dvc push pour envoyer les fichiers lourds (modèles/data) vers S3.

```


## Complemenatires  :
1. Comment connecter à votre machine EC2 depuis votre propre terminale:
```bash
ssh -i path/to/your-key.pem ec2-user@<your-ec2-public-ip>
# ex:
ssh -i "C:\Users\yana\Downloads\mlflow-server-ya.pem" ubuntu@13.53.36.3
```


2. for newly cloned code : (given that you the same role, or access)
we need first to pull the raw data : 
dvc pull data/raw.dvc



image 1 :
![alt text](image.png)





stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/raw/diabetes.csv
    params:
      - preprocess.input
      - preprocess.output
    outs:
      - data/processed/data.csv

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/data.csv
    params:
      - train.n_estimators
      - train.max_depth
      - train.random_state
      - mlflow.MLFLOW_TRACKING_URI
    outs:
      - models/model.pkl


  

  dvc remote add -d dvcstore s3://lab1-s3-yana/dvctest







dvc init

# 1. Define where DVC should store its actual "packages" (The Cache)
dvc remote add -d dvcstore s3://lab1-s3-yana/dvc_repo

#dvc import-url --to-remote s3://lab1-s3-yana/data/ .
dvc import-url s3://lab1-s3-yana/data/ .
dvc push 