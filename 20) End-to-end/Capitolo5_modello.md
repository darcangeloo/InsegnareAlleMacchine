Adesso mettiamo da parte una porzione di dati che il modello non vedrà mai durante l’addestramento, perché è l’unico modo per sapere se ha imparato o se ha soltanto memorizzato.

```python
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

Il parametro *test_size=0.2* tiene da parte il venti per cento delle righe, quindi 5634 clienti per addestrare e 1409 per la verifica finale. Il parametro *stratify=y* è quello che conta di più qui. Senza di esso la divisione sarebbe casuale e le due parti potrebbero avere proporzioni di abbandono diverse tra loro, e con una classe che pesa un quarto del totale la differenza non sarebbe trascurabile. Con *stratify* imponiamo che il 26.5 per cento di churn si ritrovi identico in entrambe le parti, quindi il test misura il modello su una popolazione fatta come quella su cui si è addestrato.

Il *random_state=42* fissa il generatore casuale. Senza di esso ogni esecuzione produrrebbe una divisione diversa e quindi punteggi diversi, e non sapremmo più se un miglioramento viene da una nostra modifica o dalla fortuna di aver pescato un test più facile. Il numero in sé non ha nessun significato, conta solo che resti sempre lo stesso.

Il passo successivo affronta lo sbilanciamento che avevamo visto nella distribuzione del target.

```python
weights = compute_sample_weight(class_weight='balanced', y=y_train)
```

Un modello cerca di sbagliare il meno possibile in totale, e quando una classe è tre volte più numerosa dell’altra la strada più comoda per sbagliare poco è dire spesso No. I pesi servono a rendere quella strada scomoda. La modalità *balanced* assegna a ogni cliente un’importanza inversamente proporzionale a quanto è frequente la sua classe, quindi un cliente che ha abbandonato pesa circa 1.88 e uno che è rimasto pesa circa 0.68. Sbagliare su un churner costa quasi tre volte di più che sbagliare su chi resta, ed è esattamente l’asimmetria di costi di cui parlavamo all’inizio, tradotta in un numero che l’algoritmo capisce.

Usiamo *compute_sample_weight* e non il parametro *class_weight* perché la regressione logistica e la random forest accettano *class_weight* mentre il *GradientBoostingClassifier* di scikit-learn no. L’idea è la stessa, cambia solo la porta d’ingresso, e siccome il gradient boosting è il modello che useremo alla fine conviene passare dai pesi campione fin da subito.

Adesso trasformiamo le colonne nella forma che un modello può leggere.

```python
one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

X_train = np.hstack([scaler.fit_transform(X_train_raw[num_cols]), one_hot.fit_transform(X_train_raw[cat_cols])])
X_test = np.hstack([scaler.transform(X_test_raw[num_cols]), one_hot.transform(X_test_raw[cat_cols])])

X_train.shape, X_test.shape
```

```
OUTPUT:
((5634, 45), (1409, 45))
```

L’*OneHotEncoder* crea una colonna nuova per ogni categoria e ci mette 1 se il cliente appartiene a quella categoria e 0 altrimenti. Lo facciamo invece di numerare le categorie perché scrivere 0, 1 e 2 al posto di mensile, annuale e biennale introdurrebbe un ordine e una distanza che non esistono, e il modello concluderebbe che il biennale è il doppio dell’annuale. Con il one hot le categorie restano solo diverse fra loro, senza gerarchia.

Il parametro *handle_unknown=’ignore’* è una scelta di produzione più che di modellazione. Quando l’applicazione finale riceverà un cliente vero potrebbe arrivare una categoria che nei dati di addestramento non c’era, per esempio un metodo di pagamento nuovo. Senza questo parametro l’encoder solleverebbe un errore e il servizio risponderebbe con un guasto. Con *ignore* tutte le colonne di quella variabile restano a zero e la previsione viene comunque prodotta, meno informata ma consegnata. Il parametro *sparse_output=False* chiede invece un array normale, perché il risultato di default è una matrice sparsa, cioè una struttura che memorizza solo le posizioni diverse da zero, e non possiamo concatenarla direttamente con i numeri scalati.

Lo *StandardScaler* sottrae da ogni colonna la sua media e la divide per la sua deviazione standard, quindi tutte finiscono centrate sullo zero e con dispersione confrontabile. Serve perché *TotalCharges* arriva quasi a novemila mentre *SeniorCitizen* vale 0 o 1, e una regressione logistica che guarda quei numeri grezzi si fa dominare dalla colonna con i valori più grandi. Al gradient boosting la scala non interessa, perché un albero cerca soglie e una soglia si sposta insieme ai dati, ma stiamo per confrontare tre modelli e vogliamo che partano tutti dalle stesse condizioni.

La differenza tra le due righe centrali è la cosa più importante di questa cella. Sul train chiamiamo *fit_transform*, sul test soltanto *transform*. Il metodo *fit* è quello che impara, cioè calcola le medie, le deviazioni standard e l’elenco delle categorie esistenti, e deve vedere solo i dati di addestramento. Se lo lasciassimo guardare anche il test, il test smetterebbe di essere fatto di dati mai visti e il punteggio finale risponderebbe a una domanda un po’ più facile di quella vera. Il *transform* invece applica soltanto quello che è già stato imparato, ed è la stessa identica operazione che farà l’applicazione alla fine del capitolo su un cliente singolo arrivato dall’esterno.

L’output conferma i conti. Quarantacinque colonne, cioè le quattro numeriche più le quarantuno che il one hot ha generato dalle quindici categoriche, e lo stesso numero di colonne da entrambe le parti.

Con i dati pronti possiamo confrontare tre modelli diversi sullo stesso problema.

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
```

```
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

I tre modelli sono quelli che conosciamo dal capitolo 2, una regressione logistica come riferimento semplice, una random forest che media molti alberi indipendenti e un gradient boosting che costruisce alberi in sequenza dove ognuno corregge gli errori del precedente. Ai primi due passiamo *class_weight=’balanced’*, che è il loro modo di tenere conto dello sbilanciamento, mentre il terzo resta senza bilanciamento perché non ha quel parametro. Lo lasciamo così apposta, perché ci fa vedere come si comporta un modello che ignora lo squilibrio.

Prima di leggere i numeri conviene fissare cosa significano nel nostro caso. La *precision* sulla classe 1 risponde alla domanda, tra i clienti che il modello ha segnalato, quanti abbandonano davvero. La *recall* risponde alla domanda opposta, tra i clienti che abbandonano davvero, quanti il modello riesce a segnalarne. La *f1-score* è la media armonica delle due, e usiamo quella e non la media normale perché la media armonica crolla appena uno dei due valori crolla, quindi non si lascia ingannare da un modello che eccelle in una metrica e fallisce nell’altra.

Adesso il confronto, ed è la parte più istruttiva del capitolo. Se guardiamo l’accuratezza vince il gradient boosting con 0.80, la random forest è seconda con 0.77 e la regressione logistica è ultima con 0.74. Se guardiamo la recall sulla classe 1 l’ordine si rovescia completamente, perché la regressione logistica intercetta il 78 per cento di chi abbandona mentre il gradient boosting si ferma al 52 per cento.

Il gradient boosting ottiene l’accuratezza migliore proprio perché è prudente nel dire che un cliente se ne andrà. Segnala poche persone e quando lo fa ci prende spesso, e infatti ha la precision più alta con 0.67, ma metà dei clienti che stavano per andarsene gli sfugge. Su una base di centomila clienti significa lasciar scappare senza accorgersene circa tredicimila persone. Un’azienda non ha bisogno di un modello che sbaglia raramente quando parla, ha bisogno di un modello che non stia zitto proprio quando dovrebbe parlare, e questa è la differenza tra la metrica che conta e la metrica che rassicura.

Scegliamo il gradient boosting come modello da portare avanti, ma bilanciato, perché è l’algoritmo che nel capitolo 2 si era comportato meglio e perché il suo problema qui non è la capacità di apprendere, è che nessuno gli ha detto quanto costa un churner mancato. Adesso cerchiamo i suoi parametri migliori.

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
```

```
OUTPUT:
Migliori parametri: {'subsample': 0.85, 'n_estimators': 100, 'min_samples_split': 5, 'max_depth': 2, 'learning_rate': 0.2}
Miglior F1 in CV: 0.7765
```

Il dizionario elenca i valori da provare per cinque parametri. *n_estimators* è quanti alberi costruire in sequenza. *learning_rate* è quanto ogni albero è autorizzato a correggere, e un valore basso rende l’apprendimento più lento ma più stabile. *max_depth* è quanto può essere complesso ogni singolo albero, e nel boosting si tiene volutamente piccolo perché il metodo funziona sommando tanti apprendisti deboli e non pochi esperti. *min_samples_split* impedisce di dividere un nodo con troppi pochi clienti dentro. *subsample* è la frazione di dati che ogni albero vede, e un valore sotto 1 introduce casualità che riduce il rischio di adattarsi troppo al train.

Le combinazioni possibili sono 3 per 4 per 4 per 3 per 3, cioè 432, e ognuna andrebbe addestrata cinque volte per la validazione incrociata, quindi 2160 addestramenti. Con *RandomizedSearchCV* ne peschiamo venti a caso e ci fermiamo a cento addestramenti. Non è una rinuncia alla qualità, perché in quasi tutti i problemi reali solo due o tre parametri spostano davvero il risultato, e una ricerca esaustiva spende la maggior parte del tempo a esplorare in modo pignolo quelli che non contano.

Il parametro *scoring=’f1’* è la riga più importante della cella e vale la pena fermarsi. Se non lo scrivessimo, scikit-learn userebbe l’accuratezza, che è il suo criterio predefinito per un classificatore. Abbiamo appena visto dove porta l’accuratezza su questo dataset, cioè a preferire il modello che si lascia sfuggire metà dei churner. La metrica che scegliamo qui è la definizione operativa di migliore, quindi sceglierla è una decisione di prodotto travestita da parametro.

Passiamo infine *sample_weight=weights* al *fit* della ricerca, così ogni combinazione viene provata su un modello bilanciato e il confronto è onesto. I parametri vincenti sono alberi profondi due, cento in sequenza, un learning rate alto e un sottocampionamento all’85 per cento, cioè un modello semplice ripetuto molte volte, che è il profilo tipico di un buon boosting.

Sul secondo numero stampato serve un avvertimento, perché è il tipo di dettaglio che fa perdere ore a chi non lo sa. Quel 0.7765 non è confrontabile con l’f1 che vedremo tra poco sul test. Quando passiamo i pesi al *fit* della ricerca, scikit-learn li usa anche nel calcolo del punteggio delle validazioni, quindi è un f1 pesato, misurato su una scala in cui i churner contano quasi il doppio degli altri. Serve a mettere in fila le venti combinazioni fra loro, tutte giudicate con lo stesso metro, e per quello va benissimo. Non serve a prevedere quanto andrà bene il modello, e usarlo così porta a credere in un sovradattamento che non esiste. La stima vera è quella che otteniamo sul test, ed è il prossimo passo.

Riaddestriamo il modello con quei parametri e guardiamo come si comporta sul test.

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
```

```
OUTPUT:
              precision    recall  f1-score   support

           0       0.91      0.72      0.81      1035
           1       0.51      0.79      0.62       374

    accuracy                           0.74      1409
   macro avg       0.71      0.76      0.71      1409
weighted avg       0.80      0.74      0.76      1409
```

L’accuratezza è scesa da 0.80 a 0.74 e va benissimo così. È il prezzo che paghiamo per portare la recall dal 52 al 79 per cento, ed è un prezzo che abbiamo deciso di pagare quando abbiamo detto che un churner mancato costa più di uno sconto regalato.

Traduciamo i numeri in persone, perché è l’unico modo per capire se un modello serve. Nel test ci sono 374 clienti che hanno davvero abbandonato, e il modello ne intercetta circa 295, lasciandone scappare una ottantina. Per farlo segnala poco meno di seicento clienti in tutto, quindi circa la metà delle persone che l’azienda contatterebbe stava effettivamente per andarsene. Senza modello, telefonando a caso, la percentuale sarebbe il 26.5 per cento della base clienti. Il modello quasi raddoppia la resa di ogni telefonata e questo, non lo 0.62 di f1, è il numero che va portato a chi decide.

Adesso chiediamo al modello cosa ha guardato per decidere.

```python
feature_names = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'] + list(one_hot.get_feature_names_out(cat_cols))
importances = pd.Series(classifier.feature_importances_, index=feature_names).sort_values(ascending=False)

print("Le 12 colonne più usate dal modello (%)")
print((importances.head(12) * 100).round(2).to_string())

per_variabile = importances.groupby(importances.index.str.split('_').str[0]).sum().sort_values(ascending=False)
print("\nImportanza per variabile originale (%)")
print((per_variabile.head(8) * 100).round(2).to_string())
```

```
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

L’attributo *feature_importances_* è un array di numeri senza nomi, quindi la prima riga ricostruisce le etichette nello stesso ordine in cui abbiamo concatenato le colonne, prima le quattro numeriche e poi le quarantuno generate dall’encoder. Usiamo *get_feature_names_out* invece di scrivere i nomi a mano perché l’encoder sa quali categorie ha visto e in che ordine le ha messe, e se sbagliassimo un nome attribuiremmo l’importanza alla colonna sbagliata senza nessun errore che ce lo segnali. Poi incapsuliamo tutto in una *pd.Series*, che tiene insieme nome e valore anche durante l’ordinamento, cosa che un array da solo non fa.

La seconda tabella somma le colonne che vengono dalla stessa variabile di partenza. Serve perché il one hot spezza una variabile in più colonne e ne frammenta l’importanza, quindi *Contract* sembra valere il 41.77 per cento quando in realtà ne vale 45.64 sommando anche il biennale e l’annuale. Il raggruppamento funziona perché *get_feature_names_out* costruisce i nomi nella forma colonna sottolineato categoria, e nessuna delle nostre quindici colonne ha un underscore nel proprio nome.

Il risultato è che una sola variabile, il tipo di contratto, vale quasi metà di tutto quello che il modello guarda, seguita dall’anzianità, dalla fibra ottica e dall’assenza di sicurezza online. È esattamente la storia che ci avevano raccontato i tre grafici prima di addestrare qualsiasi cosa. Questa coincidenza non è banale ed è un buon segno, perché un modello le cui variabili importanti non hanno niente a che vedere con quello che i dati mostravano di solito ha trovato una scorciatoia, non una regolarità.

Una precisazione onesta prima di andare avanti. Questa è importanza da impurità, cioè misura quanto ogni colonna ha aiutato a separare i gruppi durante la costruzione degli alberi, e tende a favorire le variabili continue perché offrono più punti di taglio possibili. Dice quanto il modello ha usato una colonna mentre imparava, non quanto ci dipende davvero su dati nuovi. Per quel numero esiste *permutation_importance*, che mescola una colonna alla volta sul test e misura di quanto peggiora il punteggio. È più lento e qui non ci serve, ma vale la pena sapere che le due cose non sono la stessa.

```python
top = importances.head(10).sort_values()
top.plot.barh(color='#2986cc', figsize=(7, 5), title='Cosa guarda il modello (importanza %)')
plt.xlabel('Importanza')
plt.tight_layout()
plt.savefig('importanze.png', dpi=300)
plt.show()
```

[fig4.png]
*Figura 5.4 Le dieci colonne più usate dal modello. I valori sull’asse sono frazioni, quindi 0.42 corrisponde al 42 per cento dell’importanza totale.*

Il *sort_values* senza argomenti prima di *barh* non è superfluo, perché matplotlib disegna le barre orizzontali dal basso verso l’alto e senza quel riordino la barra più lunga finirebbe in fondo al grafico invece che in cima. Il *dpi=300* serve perché la figura finisce stampata e la risoluzione predefinita, pensata per lo schermo, sulla carta si vede sgranata.

Il modello è finito. Adesso lo portiamo fuori dal notebook, e il primo passo è salvarlo su disco.

```python
joblib.dump(one_hot, 'app/models/encoder.pkl')
joblib.dump(scaler, 'app/models/scaler.pkl')
joblib.dump(classifier, 'app/models/classifier.pkl')
```

Salviamo tre oggetti e non uno solo, ed è il punto in cui sbaglia quasi chiunque lo faccia per la prima volta. L’encoder e lo scaler fanno parte del modello tanto quanto il classificatore, perché un cliente grezzo deve essere trasformato esattamente come lo erano i dati di addestramento, con le stesse medie, le stesse deviazioni standard e lo stesso elenco di categorie nello stesso ordine. Salvare solo il classificatore vuol dire ritrovarsi con un modello che nessuno sa più come alimentare, e riaddestrare l’encoder altrove produrrebbe colonne in ordine diverso e previsioni sbagliate senza nessun errore.

Usiamo *joblib* invece del modulo *pickle* della libreria standard perché è ottimizzato per gli oggetti che contengono grandi array numerici, che è esattamente il caso di un modello di scikit-learn. Va detto che questi file dipendono dalle versioni delle librerie con cui sono stati creati, quindi se in futuro reinstallerai una versione diversa di scikit-learn potrebbero non caricarsi. Nei progetti veri per questo motivo si fissano le versioni esatte in un file di dipendenze.

Un modello dentro un notebook è utile soltanto a chi lo ha scritto. Perché serva all’azienda deve poter rispondere a programmi diversi, il gestionale che prepara le campagne, il sistema di fatturazione, una pagina web usata dagli operatori del call center. Il modo standard per farlo è esporlo come servizio web, e qui servono alcune parole che nel libro non sono ancora comparse.

Un *server* è semplicemente un programma che resta acceso e aspetta. Chi lo interroga manda una *richiesta* e riceve una *risposta*, e tutto quello che succede in rete è fatto di questi due movimenti. Un *endpoint* è un indirizzo su quel server che svolge un compito preciso, e il nostro ne avrà uno solo, che si chiama */predict*. Le richieste hanno un tipo, e i due che incontrerai ovunque sono *GET*, che chiede di leggere qualcosa, e *POST*, che manda dei dati da elaborare. A noi serve POST perché dobbiamo spedire un cliente intero. I dati viaggiano in *JSON*, un formato testuale che assomiglia moltissimo a un dizionario Python e che tutti i linguaggi sanno leggere e scrivere.

Costruiamo il servizio con FastAPI, che ci regala due cose senza scrivere codice in più. Controlla che i dati in arrivo siano completi e del tipo giusto, e genera da sola una pagina di documentazione interattiva da cui possiamo provare l’endpoint. Il file si chiama *api.py* e sta nella cartella *app*, accanto alla cartella *models* dove abbiamo appena salvato i tre artefatti.

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

Il percorso della cartella lo costruiamo da *__file__*, che è una variabile che Python riempie con la posizione del file in esecuzione. Potremmo scrivere *’app/models’* come nel notebook, ma quel percorso è relativo alla cartella da cui lanciamo il programma, che cambia a seconda di dove ci troviamo quando digitiamo il comando, mentre *__file__* punta sempre allo stesso posto. È il tipo di dettaglio che nel notebook non conta niente e in un servizio che parte da solo conta moltissimo.

I tre caricamenti stanno fuori da qualsiasi funzione, quindi vengono eseguiti una volta sola quando il servizio si avvia. Se li mettessimo dentro la funzione di previsione il programma funzionerebbe lo stesso, ma rileggerebbe tre file da disco a ogni singola richiesta, e sarebbe decine di volte più lento senza nessun vantaggio.

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

Le due liste di colonne sono identiche a quelle del notebook e devono restarlo, perché l’encoder salvato si aspetta le sue quindici colonne in quell’ordine esatto. È una duplicazione fastidiosa e in un progetto più grande si risolverebbe salvando anche i nomi delle colonne insieme agli artefatti.

La classe *Customer* è il contratto del servizio. Elencando i diciannove campi con il loro tipo stiamo dichiarando cosa il servizio accetta, e Pydantic si occupa di far rispettare quella dichiarazione. Se qualcuno manda una richiesta senza *tenure*, oppure con *tenure* scritto come testo, FastAPI risponde da sola con un errore che spiega quale campo è sbagliato e perché, prima ancora che il nostro codice venga eseguito. È validazione che non abbiamo dovuto scrivere, ed è il motivo principale per cui scegliamo questa libreria in un capitolo che vuole restare corto.

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

Il decoratore sopra la funzione dice a FastAPI che questa funzione risponde alle richieste POST dirette a */predict*. Il parametro annotato con *Customer* è il modo in cui la libreria capisce che deve leggere il corpo della richiesta e trasformarlo in un oggetto di quella classe, e da lì in poi lavoriamo con dati già validati.

Il metodo *model_dump* converte l’oggetto in un dizionario, e lo mettiamo dentro una lista prima di passarlo a *pd.DataFrame* perché l’encoder e lo scaler si aspettano una tabella, non una riga sciolta. Otteniamo così un DataFrame di una riga sola, che è esattamente la forma che avevano i dati durante l’addestramento.

Le due righe successive chiamano *transform* e mai *fit_transform*, ed è lo stesso principio del notebook portato alle sue conseguenze. Qui abbiamo un cliente solo, quindi un *fit* calcolerebbe la media di un singolo valore, la sottrarrebbe da se stessa e restituirebbe zero per ogni colonna numerica. Il servizio risponderebbe senza errori e le previsioni sarebbero prive di senso, ed è la ragione per cui lo scaler va salvato e trasportato invece che ricostruito.

L’*np.hstack* rimette insieme le colonne con i numeri scalati prima e le categorie codificate dopo, nello stesso ordine in cui il modello le ha viste durante l’addestramento. Poi chiamiamo *predict_proba* invece di *predict*, perché vogliamo la probabilità e non l’etichetta, e questa è la scelta che avevamo annunciato all’inizio del capitolo quando dicevamo che all’azienda serve una classifica. Il metodo restituisce una tabella con una riga per cliente e una colonna per classe, quindi *[0][1]* prende il primo cliente e la seconda colonna, cioè la probabilità della classe 1, che è l’abbandono.

L’ultima riga arrotonda a quattro decimali e converte con *float*, perché il numero che arriva da scikit-learn è un tipo di NumPy e non uno dei tipi base di Python, e la conversione in JSON vuole un numero normale. Restituiamo un dizionario, e FastAPI lo traduce in JSON da sola.

Per far partire il servizio serve un server che esegua l’applicazione, e per FastAPI si usa Uvicorn. Si installano entrambi con *pip install fastapi uvicorn*, poi dalla cartella del progetto lanciamo il comando.

```python
uvicorn app.api:app --reload
```

```
OUTPUT:
INFO:     Will watch for changes in these directories: ['/progetti/end-to-end']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [3201] using StatReload
INFO:     Started server process [3204]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

La scritta *app.api:app* va letta come due parti separate dai due punti. A sinistra il percorso del modulo, cioè il file *api.py* dentro la cartella *app*, a destra il nome della variabile dentro quel file, che è l’*app = FastAPI()* che abbiamo creato. L’opzione *--reload* fa ripartire il server ogni volta che salviamo una modifica, comodissima mentre si sviluppa e da non usare mai su un servizio vero, perché resta in ascolto dei file e consuma risorse per niente.

L’indirizzo *127.0.0.1* è il computer stesso, quindi per ora il servizio risponde soltanto a chi lo interroga dalla stessa macchina. Aprendo il browser su *http://127.0.0.1:8000/docs* troviamo una pagina che FastAPI ha generato da sola leggendo la classe *Customer*. Elenca gli endpoint disponibili, mostra quali campi servono e permette di compilarli e inviare una richiesta vera con un bottone. La usiamo come interfaccia di prova invece di costruirne una, perché è gratis e perché resta sempre allineata al codice.

Proviamo con un cliente che ha tutte le caratteristiche pericolose viste nell’analisi, cioè due mesi di anzianità, contratto mensile, fibra ottica, nessun servizio di sicurezza e pagamento con assegno elettronico.

```python
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
  "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
  "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 94.4, "TotalCharges": 188.8
}
```

```
OUTPUT:
{"churn_probability": 0.9192}
```

Il servizio risponde che quel cliente ha il 92 per cento di probabilità di andarsene. Proviamo l’opposto, un cliente con cinque anni di anzianità, contratto biennale, sicurezza e assistenza tecnica attive e addebito automatico in banca.

```
OUTPUT:
{"churn_probability": 0.0295}
```

Tre per cento. Il modello che abbiamo addestrato dentro un notebook adesso risponde a chiunque sappia mandargli una richiesta, e restituisce un numero che un ufficio marketing può usare per ordinare una lista di telefonate. Questo è il punto in cui un progetto di machine learning smette di essere un esercizio.

Resta da dire cosa manca, come promesso all’inizio. Il servizio gira sul nostro computer e con un server di sviluppo, mentre in produzione servirebbe un server vero e un container, cioè un pacchetto che porta con sé anche le versioni esatte delle librerie, così che il modello si comporti allo stesso modo ovunque venga eseguito. Mancano l’autenticazione, perché adesso chiunque raggiunga l’indirizzo può interrogare il modello, e la registrazione delle richieste e delle risposte, che è l’unico modo per capire in seguito perché il sistema ha suggerito una certa cosa. Manca soprattutto la sorveglianza nel tempo, perché i clienti cambiano, l’azienda cambia i prezzi e i contratti, e un modello addestrato oggi peggiora lentamente senza avvisare nessuno. Accorgersene richiede misurare le previsioni contro quello che poi succede davvero, e riaddestrare quando lo scarto cresce.

Sono tutti passaggi reali e nessuno di essi è machine learning. Appartengono all’ingegneria del software e alle operazioni, e li ho lasciati fuori apposta perché questo libro finisce dove finisce il modello. Sapere che esistono, però, è già metà del lavoro, perché il modo più comune di fallire un progetto di questo tipo non è sbagliare l’algoritmo, è credere che dopo il notebook non ci sia più niente da fare.

Ripercorriamo cosa abbiamo costruito. Siamo partiti da un file grezzo con una colonna che fingeva di essere testo, abbiamo capito perché lo era e abbiamo scelto come sistemarla guardando cosa significava quel vuoto invece di riempirlo per abitudine. Abbiamo esplorato i dati con tre grafici scelti, e quei tre grafici ci hanno detto in anticipo quali variabili avrebbero contato. Abbiamo diviso in addestramento e test prima di preparare le feature, e abbiamo imparato una volta sola le medie e le categorie per poi applicarle ovunque servisse.

Abbiamo confrontato tre modelli e abbiamo scoperto che il migliore secondo l’accuratezza era il peggiore secondo il nostro problema, il che ci ha costretti a scegliere esplicitamente una metrica invece di accettarne una per default. Abbiamo cercato i parametri con una ricerca casuale, salvato il modello insieme ai suoi trasformatori, e costruito un servizio che risponde a chiunque con una probabilità invece che con un sì o un no.

Se dovessi tenere una sola cosa di questo capitolo, tieni la scelta della metrica. Tutto il resto sono strumenti che cambieranno nei prossimi anni, mentre la domanda su cosa significhi esattamente sbagliare, nel problema che hai davanti e per chi ti ha chiesto di risolverlo, resterà identica anche quando scikit-learn non si chiamerà più così.
