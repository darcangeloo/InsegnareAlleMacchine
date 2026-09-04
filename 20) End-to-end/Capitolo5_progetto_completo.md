Svilupperemo un modello che stima la probabilità di abbandono di un cliente e lo esporremo come servizio web, così che altri programmi possano interrogarlo. Il dataset è Telco Customer Churn, 7043 righe e 21 colonne, una riga per cliente. Iniziamo importando le librerie e guardando cosa contiene il file.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import joblib

df = pd.read_csv("telco_churn.csv")
df.info()
OUTPUT:
<class 'pandas.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 21 columns):
 #   Column            Non-Null Count  Dtype
---  ------            --------------  -----
 0   customerID        7043 non-null   str
 1   gender            7043 non-null   str
 2   SeniorCitizen     7043 non-null   int64
 3   Partner           7043 non-null   str
 4   Dependents        7043 non-null   str
 5   tenure            7043 non-null   int64
 6   PhoneService      7043 non-null   str
 7   MultipleLines     7043 non-null   str
 8   InternetService   7043 non-null   str
 9   OnlineSecurity    7043 non-null   str
 10  OnlineBackup      7043 non-null   str
 11  DeviceProtection  7043 non-null   str
 12  TechSupport       7043 non-null   str
 13  StreamingTV       7043 non-null   str
 14  StreamingMovies   7043 non-null   str
 15  Contract          7043 non-null   str
 16  PaperlessBilling  7043 non-null   str
 17  PaymentMethod     7043 non-null   str
 18  MonthlyCharges    7043 non-null   float64
 19  TotalCharges      7043 non-null   str
 20  Churn             7043 non-null   str
dtypes: float64(1), int64(2), str(18)
```

Non ci sono valori mancanti, quindi possiamo saltare la parte sull’imputazione. La colonna che interessa è *Dtype*. Diciotto colonne su ventuno contengono testo e andranno codificate, ma una di queste non dovrebbe starci, perché *TotalCharges* è il totale pagato dal cliente e dovrebbe essere un numero come *MonthlyCharges*. Notiamo anche che *SeniorCitizen* è già int64, quindi contiene 0 e 1 e la tratteremo come numerica.

Proviamo a convertire *TotalCharges* e vediamo quanti valori resistono.

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print(f"Valori non convertibili in numero: {df['TotalCharges'].isna().sum()}")
print(f"Anzianità di quei clienti: {df.loc[df['TotalCharges'].isna(), 'tenure'].unique()}")

df['TotalCharges'] = df['TotalCharges'].fillna(0)
OUTPUT:
Valori non convertibili in numero: 11
Anzianità di quei clienti: [0]
```

Con *errors=’coerce’* i valori che non si convertono diventano NaN invece di sollevare un’eccezione, quindi possiamo contarli e guardarli invece di sapere solo che qualcosa non va. Sono 11 e hanno tutti *tenure* uguale a zero, cioè clienti al primo mese che non hanno ancora ricevuto una fattura. Una cella vuota in mezzo a una colonna di numeri basta a far leggere a pandas l’intera colonna come testo.

Riempiamo con *fillna(0)* e non con la media, perché il totale pagato da chi non ha ancora pagato è zero e non è un dato mancante. La media inventerebbe una storia di pagamenti inesistente proprio sui clienti nuovi, che come vedremo tra poco sono i più a rischio.

Separiamo le colonne per tipo, visto che le tratteremo in modo diverso.

```python
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
```

Sono quindici categoriche e quattro numeriche. Restano fuori *Churn*, che è la risposta, e *customerID*, che ha 7043 valori tutti diversi e una volta codificato produrrebbe 7043 colonne inutili. Adesso separiamo le feature dal target.

```python
X = df[num_cols + cat_cols]
y = df['Churn'].map({'Yes': 1, 'No': 0})
```

L’ordine delle colonne in X non è casuale, prima le numeriche e poi le categoriche, perché è lo stesso in cui le ricomporremo dopo la trasformazione ed è lo stesso che dovrà rispettare il servizio alla fine del capitolo. Un modello non conosce i nomi delle colonne, conosce le posizioni, quindi due colonne scambiate danno previsioni sbagliate senza nessun errore.

Mappiamo Yes a 1 e non il contrario perché la classe 1 è quella positiva in scikit-learn, cioè quella di cui parlano precision e recall. Invertendo i numeri le metriche funzionerebbero lo stesso ma ci direbbero quanto siamo bravi a riconoscere i clienti che restano.

Vediamo com’è divisa la colonna che vogliamo prevedere.

```python
df['Churn'].value_counts(normalize=True)
OUTPUT:
Churn
No     0.73463
Yes    0.26537
Name: proportion, dtype: float64
```

Il 26.5 per cento dei clienti abbandona. Teniamo a mente questo numero, perché un modello che rispondesse sempre No, senza guardare niente, avrebbe il 73.5 per cento di accuratezza, e tornerà utile quando confronteremo i classificatori.

Prima di modellare guardiamo i dati con tre grafici. Non uno per colonna, perché con diciannove colonne avremmo diciannove grafici e nessuna risposta. Il primo misura il rischio di abbandono al crescere dell’anzianità.

```python
churn_by_tenure = df.groupby(pd.cut(df['tenure'], bins=range(0, 81, 6)))['Churn'].apply(lambda s: (s == 'Yes').mean())
churn_by_tenure.plot(marker='o', color='#d1495b', figsize=(8, 4), title='Rischio di abbandono per anzianità')
plt.show()
```

[fig1.png]

*pd.cut* taglia *tenure* in fasce da sei mesi e il risultato lo passiamo direttamente a *groupby*, che quindi raggruppa per anzianità invece che per un valore già presente nel dataset. Arriviamo a 81 e non a 72 perché *range* esclude l’ultimo valore.

La lambda usa un trucco che riprenderemo nel grafico dopo. L’espressione *(s == ’Yes’)* trasforma la colonna in booleani, e la media di una serie booleana è la proporzione di veri, quindi restituisce già il tasso di abbandono della fascia senza contare i Yes e i No separatamente.

Nei primi sei mesi abbandona più di un cliente su due, dopo cinque anni siamo intorno al cinque per cento. Chi resta abbastanza a lungo tende a restare, quindi le azioni di retention hanno senso concentrate all’inizio del rapporto.

Il secondo grafico cerca quali categorie spostano il rischio.

```python
rates = pd.concat([df.groupby(c)['Churn'].apply(lambda s: (s == 'Yes').mean()) for c in cat_cols], keys=cat_cols)
top = (rates / df['Churn'].eq('Yes').mean()).sort_values().pipe(lambda s: pd.concat([s.head(6), s.tail(6)]))
top.plot.barh(color=['#2986cc' if v < 1 else '#d1495b' for v in top], figsize=(7, 6), title='Driver del churn (lift vs media)')
plt.show()
```

[fig2.png]

La comprensione di lista ripete su ogni colonna categorica lo stesso calcolo del grafico precedente, e *keys=cat_cols* costruisce un indice a due livelli così sappiamo da quale colonna viene ogni categoria. Senza, ci ritroveremmo categorie che si chiamano allo stesso modo in colonne diverse.

Dividendo il tasso di ogni categoria per il tasso medio otteniamo il lift, sopra 1 abbandona più della media e sotto 1 meno. Con *pipe* restiamo su una sola catena e teniamo le sei categorie più basse e le sei più alte, perché le decine che stanno intorno a 1 non spostano niente e renderebbero illeggibile il grafico.

In alto troviamo l’assegno elettronico, il contratto mensile e la fibra ottica. In basso il contratto biennale a 0.11, di gran lunga il segnale più forte del dataset. Le varie *No internet service* valgono tutte 0.28 perché sono le stesse persone viste da colonne diverse, cioè i clienti senza internet che non possono avere né backup né streaming.

Il terzo grafico mette insieme due colonne numeriche per cercare una combinazione, cosa che i primi due non potevano fare.

```python
colors = df['Churn'].map({'Yes': '#d1495b', 'No': '#2986cc'})
df.plot.scatter(x='tenure', y='MonthlyCharges', c=colors, alpha=0.4, figsize=(7, 5), title='Zona di rischio churn')
plt.show()
```

[fig3.png]

Qui *map* converte in colori invece che in numeri. Il parametro *alpha=0.4* non è estetica, perché con 7043 punti sovrapposti senza trasparenza vedremmo solo l’ultimo disegnato e perderemmo proprio l’informazione sulla densità.

Il rosso si addensa a sinistra e in alto, cioè poca anzianità e bolletta cara, mentre in basso a sinistra domina il blu, i clienti con la sola linea telefonica che pagano poco e restano. Non è l’anzianità da sola. Fra i clienti con meno di un anno, chi paga sotto i trentacinque euro abbandona nel 24 per cento dei casi e chi paga sopra gli ottantacinque nel 74.

Adesso dividiamo i dati, e lo facciamo prima di trasformarli.

```python
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

Restano 5634 clienti per addestrare e 1409 per la verifica. Il parametro *stratify=y* mantiene il 26.5 per cento di churn identico nelle due parti, e con una classe di minoranza che pesa un quarto del totale la differenza non sarebbe trascurabile.

```python
weights = compute_sample_weight(class_weight='balanced', y=y_train)
```

Il parametro *class_weight* non esiste su *GradientBoostingClassifier*, quindi calcoliamo i pesi a parte e li passeremo direttamente al *fit*. La modalità *balanced* assegna a ogni cliente un peso inversamente proporzionale alla frequenza della sua classe, quindi un churner pesa circa 1.88 e chi resta circa 0.68. È il modo di dire all’algoritmo che i due errori non costano uguale.

Adesso trasformiamo le colonne.

```python
one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

X_train = np.hstack([scaler.fit_transform(X_train_raw[num_cols]), one_hot.fit_transform(X_train_raw[cat_cols])])
X_test = np.hstack([scaler.transform(X_test_raw[num_cols]), one_hot.transform(X_test_raw[cat_cols])])

X_train.shape, X_test.shape
OUTPUT:
((5634, 45), (1409, 45))
```

Quarantacinque colonne, cioè le quattro numeriche più le quarantuno che il one hot ha generato dalle quindici categoriche. Il parametro *handle_unknown=’ignore’* serve in produzione, perché se al servizio arrivasse una categoria mai vista in addestramento, per esempio un metodo di pagamento nuovo, l’encoder solleverebbe un errore e la richiesta fallirebbe. Con *ignore* le colonne di quella variabile restano a zero e la previsione viene comunque prodotta. Il parametro *sparse_output=False* ci serve per avere un array normale da concatenare.

La differenza che conta davvero è tra le due righe centrali. Sul train chiamiamo *fit_transform*, sul test soltanto *transform*. Medie, deviazioni standard e lista delle categorie si imparano una volta sola sui dati di addestramento, altrimenti il test smetterebbe di essere fatto di dati mai visti. Ed è esattamente quello che farà il servizio alla fine del capitolo su un cliente singolo, quindi il notebook e l’API raccontano la stessa storia.

Confrontiamo tre modelli sullo stesso problema.

```python
log_reg = LogisticRegression(class_weight='balanced', random_state=42)
forest = RandomForestClassifier(n_estimators = 100, random_state=42, class_weight='balanced')
boosting = GradientBoostingClassifier(n_estimators = 100, random_state=42)

log_reg.fit(X_train, y_train)
forest.fit(X_train, y_train)
boosting.fit(X_train, y_train)

log_pred = log_reg.predict(X_test)
forest_pred = forest.predict(X_test)
boosting_pred = boosting.predict(X_test)

print(f'Regressione Logistica: {classification_report(y_test, log_pred)}')
print(f'Random Forest: {classification_report(y_test, forest_pred)}')
print(f'Gradient Boosting: {classification_report(y_test, boosting_pred)}')
OUTPUT:
Regressione Logistica:               precision    recall  f1-score   support

           0       0.90      0.72      0.80      1035
           1       0.50      0.78      0.61       374

    accuracy                           0.74      1409
   macro avg       0.70      0.75      0.71      1409
weighted avg       0.80      0.74      0.75      1409

Random Forest:               precision    recall  f1-score   support

           0       0.86      0.82      0.84      1035
           1       0.56      0.63      0.59       374

    accuracy                           0.77      1409
   macro avg       0.71      0.73      0.72      1409
weighted avg       0.78      0.77      0.77      1409

Gradient Boosting:               precision    recall  f1-score   support

           0       0.84      0.91      0.87      1035
           1       0.67      0.52      0.58       374

    accuracy                           0.80      1409
   macro avg       0.75      0.71      0.73      1409
weighted avg       0.79      0.80      0.79      1409
```

Ai primi due passiamo *class_weight=’balanced’*, il terzo resta sbilanciato perché non ha quel parametro, e ci serve proprio così per vedere come si comporta un modello che ignora lo squilibrio.

Se guardiamo l’accuratezza vince il gradient boosting con 0.80 e la regressione logistica è ultima con 0.74. Se guardiamo la recall sulla classe 1 l’ordine si rovescia, perché la logistica intercetta il 78 per cento di chi abbandona e il boosting il 52.

Il boosting ottiene l’accuratezza migliore proprio perché segnala poche persone, quindi sbaglia raramente quando parla, ma tace su metà dei churner. Su una base di centomila clienti sono tredicimila abbandoni che nessuno vede arrivare. È il caso in cui la metrica predefinita risponde alla domanda sbagliata, e conviene ricordarlo ogni volta che un modello sembra andare bene.

Portiamo avanti il gradient boosting, ma bilanciato, perché il suo problema qui non è la capacità di imparare, è che nessuno gli ha detto quanto costa un churner mancato. Cerchiamo i suoi parametri migliori.

```python
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.7, 0.85, 1.0],
}

random_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_dist, n_iter=20, cv=5, scoring='f1', random_state=42
)
random_search.fit(X_train, y_train, sample_weight=weights)

print(f"Migliori parametri: {random_search.best_params_}")
print(f"Miglior F1 in CV: {random_search.best_score_:.4f}")
OUTPUT:
Migliori parametri: {'subsample': 0.85, 'n_estimators': 100, 'min_samples_split': 5, 'max_depth': 2, 'learning_rate': 0.2}
Miglior F1 in CV: 0.7765
```

Le combinazioni possibili sono 432 e con la validazione incrociata a cinque diventerebbero 2160 addestramenti. Con *RandomizedSearchCV* ne peschiamo venti a caso e ci fermiamo a cento. In problemi come questo solo due o tre parametri spostano davvero il risultato, quindi una ricerca esaustiva spenderebbe la maggior parte del tempo su quelli che non contano. Il parametro *subsample* è la frazione di dati che ogni albero vede, e sotto 1 introduce casualità che riduce il rischio di adattarsi troppo al train.

La riga che conta è *scoring=’f1’*. Senza, scikit-learn userebbe l’accuratezza, che è il suo criterio predefinito, e abbiamo appena visto quale modello sceglierebbe. La metrica che indichiamo qui è la definizione operativa di migliore.

Il *best_score_* va letto con attenzione. Passando *sample_weight* al *fit* della ricerca, i pesi finiscono anche nel calcolo del punteggio, quindi 0.7765 è un f1 pesato e non si confronta con quello che vedremo tra poco sul test. Serve a mettere in fila le venti combinazioni tra loro, tutte giudicate con lo stesso metro, non a stimare quanto andrà bene il modello.

Riaddestriamo con i parametri trovati.

```python
classifier = GradientBoostingClassifier(
    subsample=0.85,
    n_estimators=100,
    min_samples_split=5,
    max_depth=2,
    learning_rate=0.2,
    random_state=42,
)

classifier.fit(X_train, y_train, sample_weight=weights)
y_pred = classifier.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
OUTPUT:
              precision    recall  f1-score   support

           0       0.91      0.72      0.81      1035
           1       0.51      0.79      0.62       374

    accuracy                           0.74      1409
   macro avg       0.71      0.76      0.71      1409
weighted avg       0.80      0.74      0.76      1409
```

L’accuratezza scende da 0.80 a 0.74 e la recall sale dal 52 al 79 per cento. È il prezzo che avevamo deciso di pagare.

Traduciamo i numeri in persone. Nel test ci sono 374 clienti che hanno davvero abbandonato, il modello ne intercetta circa 295 segnalandone poco meno di seicento in tutto. Circa metà delle persone che l’azienda contatterebbe stava effettivamente per andarsene, contro il 26.5 per cento di una lista compilata a caso. Il modello quasi raddoppia la resa di ogni telefonata, e questo, non lo 0.62 di f1, è il numero da portare a chi decide.

Chiediamo al modello cosa ha guardato per decidere.

```python
feature_names = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'] + list(one_hot.get_feature_names_out(cat_cols))
importances = pd.Series(classifier.feature_importances_, index=feature_names).sort_values(ascending=False)

print("Le 12 colonne più usate dal modello (%)")
print((importances.head(12) * 100).round(2).to_string())

per_variabile = importances.groupby(importances.index.str.split('_').str[0]).sum().sort_values(ascending=False)
print("\nImportanza per variabile originale (%)")
print((per_variabile.head(8) * 100).round(2).to_string())
OUTPUT:
Le 12 colonne più usate dal modello (%)
Contract_Month-to-month           41.77
tenure                            12.02
InternetService_Fiber optic        8.50
OnlineSecurity_No                  8.10
TotalCharges                       7.35
MonthlyCharges                     5.85
Contract_Two year                  3.62
TechSupport_No                     3.02
PaymentMethod_Electronic check     2.89
PaperlessBilling_Yes               1.42
StreamingMovies_Yes                0.92
MultipleLines_No                   0.69

Importanza per variabile originale (%)
Contract           45.64
tenure             12.02
InternetService     8.70
OnlineSecurity      8.10
TotalCharges        7.35
MonthlyCharges      5.85
PaymentMethod       3.18
TechSupport         3.09
```

L’attributo *feature_importances_* è un array senza nomi, quindi li ricostruiamo con *get_feature_names_out* nello stesso ordine in cui abbiamo concatenato le colonne. Li mettiamo in una *pd.Series* perché tiene insieme nome e valore anche durante l’ordinamento, cosa che un array non fa.

La seconda tabella somma le colonne che vengono dalla stessa variabile di partenza, perché il one hot spezza una variabile in più colonne e ne frammenta l’importanza. *Contract* sembra valere 41.77 e in realtà ne vale 45.64. Il raggruppamento funziona perché i nomi generati hanno la forma colonna sottolineato categoria e nessuna delle nostre quindici colonne ha un underscore nel nome.

Il tipo di contratto vale quasi metà di tutto quello che il modello guarda, poi anzianità, fibra ottica e assenza di sicurezza online. È la stessa storia che ci avevano raccontato i tre grafici prima di addestrare qualsiasi cosa, e la coincidenza è un buon segno, perché un modello le cui variabili importanti non c’entrano niente con quello che i dati mostravano di solito ha trovato una scorciatoia.

Va detto che questa è importanza da impurità, cioè misura quanto il modello ha usato una colonna mentre imparava, e tende a favorire le variabili continue perché offrono più punti di taglio. Per sapere quanto ci dipende davvero su dati nuovi esiste *permutation_importance*, che mescola una colonna alla volta sul test e misura di quanto peggiora il punteggio.

```python
top = importances.head(10).sort_values()
top.plot.barh(color='#2986cc', figsize=(7, 5), title='Cosa guarda il modello (importanza %)')
plt.xlabel('Importanza')
plt.tight_layout()
plt.savefig('importanze.png', dpi=300)
plt.show()
```

[fig4.png]

Il *sort_values* prima di *barh* serve perché matplotlib disegna le barre orizzontali dal basso, quindi senza riordino la più lunga finirebbe in fondo. Adesso portiamo il modello fuori dal notebook, e il primo passo è salvarlo.

```python
joblib.dump(one_hot, 'app/models/encoder.pkl')
joblib.dump(scaler, 'app/models/scaler.pkl')
joblib.dump(classifier, 'app/models/classifier.pkl')
```

Salviamo tre oggetti e non uno. L’encoder e lo scaler fanno parte del modello quanto il classificatore, perché un cliente grezzo va trasformato con le stesse medie e lo stesso elenco di categorie nello stesso ordine. Salvare solo il classificatore vuol dire ritrovarsi con un modello che nessuno sa più alimentare. Usiamo *joblib* invece di *pickle* perché è ottimizzato per gli oggetti che contengono grandi array numerici.

Un modello dentro un notebook è utile solo a chi lo ha scritto. Perché serva all’azienda deve rispondere ad altri programmi, il gestionale che prepara le campagne o una pagina usata dal call center, e il modo standard per farlo è esporlo come servizio web. Servono poche parole nuove. Un *server* è un programma che resta acceso e aspetta, chi lo interroga manda una *richiesta* e riceve una *risposta*, e un *endpoint* è un indirizzo su quel server che svolge un compito preciso. Le richieste hanno un tipo, e a noi serve *POST*, che manda dei dati da elaborare, invece di *GET*, che chiede di leggere qualcosa. I dati viaggiano in *JSON*, un formato testuale che assomiglia a un dizionario Python.

Costruiamo il servizio con FastAPI, che controlla da sola che i dati in arrivo siano completi e del tipo giusto e genera una pagina di prova senza scrivere codice in più. Il file si chiama *api.py* e sta nella cartella *app*, accanto a *models*.

```python
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

models_folder = Path(__file__).parent / "models"
encoder = joblib.load(models_folder / "encoder.pkl")
scaler = joblib.load(models_folder / "scaler.pkl")
classifier = joblib.load(models_folder / "classifier.pkl")
```

Il percorso lo costruiamo da *__file__*, che contiene la posizione del file in esecuzione, invece di scrivere *’app/models’* come nel notebook. Quel percorso sarebbe relativo alla cartella da cui lanciamo il comando, che cambia, mentre *__file__* punta sempre allo stesso posto. I tre caricamenti stanno fuori da qualsiasi funzione, quindi avvengono una volta sola all’avvio e non a ogni richiesta.

```python
categorical_columns = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"]
numeric_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

app = FastAPI()


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
```

Le liste di colonne sono identiche a quelle del notebook e devono restarlo, perché l’encoder salvato si aspetta le sue quindici colonne in quell’ordine. La classe *Customer* è il contratto del servizio, cioè dichiara i diciannove campi accettati e il loro tipo. Se qualcuno manda una richiesta senza *tenure*, o con *tenure* scritto come testo, FastAPI risponde da sola con un errore che dice quale campo è sbagliato, prima che il nostro codice venga eseguito.

```python
@app.post("/predict")
def predict(customer: Customer):
    customer_data = pd.DataFrame([customer.model_dump()])
    encoded_categories = encoder.transform(customer_data[categorical_columns])
    scaled_numbers = scaler.transform(customer_data[numeric_columns])
    all_features = np.hstack([scaled_numbers, encoded_categories])
    churn_probability = classifier.predict_proba(all_features)[0][1]
    return {"churn_probability": round(float(churn_probability), 4)}
```

Il decoratore dice a FastAPI che questa funzione risponde alle richieste POST dirette a */predict*, e il parametro annotato con *Customer* è il modo in cui capisce che deve leggere il corpo della richiesta. *model_dump* converte l’oggetto in un dizionario, che mettiamo dentro una lista perché encoder e scaler si aspettano una tabella e non una riga sciolta.

Le due trasformazioni chiamano *transform* e mai *fit_transform*. Qui abbiamo un cliente solo, quindi un *fit* calcolerebbe la media di un singolo valore e restituirebbe zero per ogni colonna numerica. Il servizio risponderebbe senza errori e le previsioni sarebbero senza senso, ed è per questo che lo scaler va salvato e trasportato invece che ricostruito.

Chiamiamo *predict_proba* invece di *predict* perché all’azienda serve una classifica dei clienti per rischio, non un sì o un no. Il metodo restituisce una riga per cliente e una colonna per classe, quindi *[0][1]* prende il primo cliente e la probabilità della classe 1. La conversione con *float* serve perché il numero che arriva da scikit-learn è un tipo di NumPy e la traduzione in JSON vuole un numero normale.

Per far partire il servizio installiamo entrambi con *pip install fastapi uvicorn* e poi lanciamo il comando dalla cartella del progetto.

```python
uvicorn app.api:app --reload
OUTPUT:
INFO:     Will watch for changes in these directories: ['/progetti/end-to-end']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [3201] using StatReload
INFO:     Started server process [3204]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

La scritta *app.api:app* ha due parti, a sinistra il file *api.py* dentro la cartella *app* e a destra il nome della variabile dentro quel file. L’opzione *--reload* fa ripartire il server a ogni salvataggio, comoda mentre si sviluppa e da non usare su un servizio vero.

Aprendo il browser su *http://127.0.0.1:8000/docs* troviamo una pagina che FastAPI ha generato leggendo la classe *Customer*. Elenca gli endpoint, mostra quali campi servono e permette di inviare una richiesta vera con un bottone, quindi la usiamo come interfaccia di prova invece di costruirne una.

Proviamo con un cliente che ha tutte le caratteristiche pericolose viste nell’analisi, due mesi di anzianità, contratto mensile, fibra ottica, nessun servizio di sicurezza e pagamento con assegno elettronico.

```python
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
  "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
  "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 94.4, "TotalCharges": 188.8
}
OUTPUT:
{"churn_probability": 0.9192}
```

Novantadue per cento. Proviamo l’opposto, cinque anni di anzianità, contratto biennale, sicurezza e assistenza attive e addebito automatico in banca.

```python
{
  "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
  "tenure": 62, "PhoneService": "Yes", "MultipleLines": "Yes",
  "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
  "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
  "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 65.2, "TotalCharges": 4042.4
}
OUTPUT:
{"churn_probability": 0.0295}
```

Tre per cento. Il modello addestrato dentro un notebook adesso risponde a chiunque sappia mandargli una richiesta, e restituisce un numero con cui un ufficio marketing può ordinare una lista di telefonate.

Resta da dire cosa manca, come promesso all’inizio. Il servizio gira sul nostro computer e con un server di sviluppo, mentre in produzione servirebbe un container, cioè un pacchetto che porta con sé anche le versioni esatte delle librerie. Mancano l’autenticazione, perché ora chiunque raggiunga l’indirizzo può interrogare il modello, e la registrazione delle richieste, che è l’unico modo per capire in seguito perché il sistema ha suggerito una certa cosa. Manca soprattutto la sorveglianza nel tempo, perché i clienti cambiano e un modello addestrato oggi peggiora lentamente senza avvisare nessuno.

Sono passaggi reali e nessuno di essi è machine learning, appartengono all’ingegneria del software, e li ho lasciati fuori perché questo libro finisce dove finisce il modello. Sapere che esistono però è già metà del lavoro, perché il modo più comune di fallire un progetto di questo tipo non è sbagliare l’algoritmo, è credere che dopo il notebook non ci sia più niente da fare.

Ripercorriamo. Siamo partiti da un file grezzo con una colonna che fingeva di essere testo, l’abbiamo sistemata guardando cosa significava quel vuoto invece di riempirlo per abitudine, e abbiamo esplorato i dati con tre grafici che ci hanno detto in anticipo quali variabili avrebbero contato. Abbiamo diviso prima di trasformare, imparato le medie e le categorie una volta sola, confrontato tre modelli e scoperto che il migliore secondo l’accuratezza era il peggiore per il nostro problema. Poi abbiamo cercato i parametri, salvato il modello insieme ai suoi trasformatori e costruito un servizio che risponde con una probabilità.

Se dovessi tenere una cosa sola di questo capitolo, tieni la scelta della metrica. Gli strumenti cambieranno, mentre la domanda su cosa significhi esattamente sbagliare, nel problema che hai davanti e per chi ti ha chiesto di risolverlo, resterà identica anche quando scikit-learn non si chiamerà più così.
