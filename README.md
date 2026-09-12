# Insegnare alle Macchine

Codice sorgente e dataset del libro **Insegnare alle Macchine**, Machine Learning in Python con progetti reali, per chi parte da zero.

ISBN: 9798193716099

Ogni cartella corrisponde a un progetto o a un algoritmo trattato nel libro, non a un capitolo: alcuni capitoli ne raggruppano diversi. I notebook contengono gli output già eseguiti, così puoi confrontare i tuoi risultati con quelli stampati nel libro riga per riga.

## Mappa cartelle → capitoli

Capitolo 2, Machine Learning Supervisionato
- `1) Linear Regression From Scratch`
- `2) Linear Regression Real-World`
- `3) Logistic Regression From Scratch`
- `4) Logistic Regression Real-World`
- `5) K-NN Sklearn`
- `5.1) KNN Regressor`
- `6) Naive Bayes`
- `7) Regularization`
- `9) SVM`
- `9.1) SVR`
- `10) Decision Trees`
- `11) Random Forest`
- `12) Gradient Boosting`

Capitolo 3, Gestione Dati Sporchi
- `13) Dati sporchi` (con la sottocartella `13.1) Credit Card Fraud`)

Capitolo 4, Machine Learning non Supervisionato
- `14) K-means From Scratch`
- `15) K-means Sklearn`
- `16) DBSCAN`
- `17) Clustering gerarchico`
- `18) GMM`
- `19) PCA`

Capitolo 5, Dalla preparazione dei dati a servire il modello
- `20) End-to-end` (contiene anche `app/`, con `api.py` e i modelli salvati)

## Ambiente

Python 3.11.

*(Versioni esatte delle librerie in arrivo, vedi `requirements.txt` in questa stessa cartella.)*

## Dataset, fonte e licenza

Iris (Capitolo 5, K-NN Sklearn)
Fonte: kaggle.com/datasets/uciml/iris, originariamente pubblicato su UCI Machine Learning Repository (archive.ics.uci.edu/dataset/53/iris)
Licenza: CC BY 4.0

Breast Cancer Wisconsin Diagnostic (Capitolo 9, SVM)
Fonte: kaggle.com/datasets/uciml/breast-cancer-wisconsin-data, originariamente pubblicato su UCI Machine Learning Repository (archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
Licenza: CC BY 4.0

Concrete Compressive Strength (Capitolo 9.1, SVR)
Fonte: kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set, originariamente pubblicato su UCI Machine Learning Repository (archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
Licenza: CC BY 4.0

Pima Indians Diabetes Dataset (Capitolo 6, Naive Bayes)
Fonte: kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset
Licenza: CC0, dominio pubblico

California Housing Prices (Capitolo 5.1, K-NN Regressor)
Fonte: kaggle.com/datasets/camnugent/california-housing-prices
Licenza: CC0, dominio pubblico

Customer Segmentation Tutorial - Mall Customers (Capitolo 15, K-means Sklearn)
Fonte: kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
Licenza: indicata dall'autore nella descrizione del dataset

Instagram Visit Clustering (Capitolo 14, K-means da zero)
Fonte: kaggle.com/datasets/chaandsheikh/instagram-visit-clustering
Licenza: dati protetti dagli autori originali

Car Purchase Price Beginner Dataset (Capitolo 2, Regressione Lineare)
Fonte: kaggle.com/datasets/yashk07/car-purchase-price-beginner-dataset
Licenza: non specificata dall'autore

Telco Customer Churn (Capitolo 20, progetto end-to-end)
Fonte: kaggle.com/datasets/blastchar/telco-customer-churn
Licenza: dati protetti dagli autori originali

Credit Card Fraud Detection (Capitolo 13.1, dati sporchi)
Fonte: kaggle.com/datasets/mlg-ulb/creditcardfraud
Licenza: Open Database License (ODbL) 1.0, opendatacommons.org/licenses/dbcl/1-0
Citazione richiesta dal dataset: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi, "Calibrating Probability with Undersampling for Unbalanced Classification", IEEE Symposium Series on Computational Intelligence (SSCI), 2015

Teen Mental Health Dataset (Capitolo 4, Regressione Logistica)
Fonte: kaggle.com/datasets/sebz0206/teen-mental-health-dataset
Licenza: Apache 2.0

Student Performance and Study Habits Dataset (Capitolo 10, Alberi decisionali)
Fonte: kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset
Licenza: CC0, dominio pubblico

Market features (Capitolo 18, Gaussian Mixture Model)
Le variabili settimanali standardizzate di rendimento S&P 500, VIX e spread dei tassi usate in questo capitolo sono un'elaborazione dell'autore a partire da serie di mercato pubbliche, non un dataset scaricato da una singola pagina Kaggle.

## Come lanciare l'API del Capitolo 5

```
cd "20) End-to-end/app"
pip install fastapi uvicorn
uvicorn app.api:app --reload
```

Poi apri `http://127.0.0.1:8000/docs` per la documentazione interattiva e testare l'endpoint di previsione.

## Errata

Se trovi un errore nel testo o nel codice, apri una issue in questo repository. La correzione finisce nella prossima edizione con il tuo nome accanto.
