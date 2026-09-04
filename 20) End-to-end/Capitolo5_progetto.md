Iniziamo importando quello che serve e caricando il file. Gli import sono più numerosi del solito perché in questo capitolo usiamo insieme cose che finora abbiamo visto separate, cioè tre classificatori, due trasformatori, la ricerca dei parametri e il salvataggio su disco. Li mettiamo tutti in cima una volta sola invece di aggiungerli man mano, così quando riapriremo il notebook tra un mese la prima cella ci dirà subito di cosa è fatto il progetto.

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
```

```
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
memory usage: 1.9 MB
```

L’unico metodo che abbiamo chiamato è *info*, e vale la pena leggerne l’output con attenzione perché contiene già tre informazioni che decideranno il resto del capitolo. La prima è la dimensione, 7043 righe e 21 colonne. La seconda è la colonna *Non-Null Count*, che segna 7043 su ogni riga, quindi non ci sono valori mancanti dichiarati e possiamo saltare tutta la parte sull’imputazione vista nel capitolo 3.

La terza è la colonna *Dtype*, ed è quella interessante. Diciotto colonne su ventuno contengono testo, quindi nessun modello di scikit-learn le accetta così come sono e dovremo trasformarle. Fra queste però ce n’è una che non ci aspettavamo, perché *TotalCharges* è il totale pagato dal cliente e dovrebbe essere un numero come *MonthlyCharges*, che infatti risulta float64. Notiamo anche che *SeniorCitizen* è già int64, cioè contiene 0 e 1 invece di No e Yes, quindi si comporterà come una colonna numerica e non avrà bisogno di essere codificata.

Prima di andare avanti dobbiamo capire perché *TotalCharges* è arrivata come testo. Proviamo a convertirla e vediamo cosa succede.

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print(f"Valori non convertibili in numero: {df['TotalCharges'].isna().sum()}")
print(f"Anzianità di quei clienti: {df.loc[df['TotalCharges'].isna(), 'tenure'].unique()}")

df['TotalCharges'] = df['TotalCharges'].fillna(0)
```

```
OUTPUT:
Valori non convertibili in numero: 11
Anzianità di quei clienti: [0]
```

La funzione *pd.to_numeric* tenta di convertire ogni valore della colonna in numero. Il parametro che conta è *errors=’coerce’*, perché senza di esso la funzione lancia un’eccezione al primo valore che non riesce a convertire e il programma si ferma lì, dicendoci solo che qualcosa non va. Con *coerce* invece i valori impossibili diventano NaN, quindi restano nel dataset e possiamo contarli e guardarli. È la differenza tra un errore che interrompe e un errore che si lascia studiare.

Il primo print conta quanti sono e sono 11 su 7043. Il secondo va a vedere chi sono, e usa *df.loc* con una maschera booleana per prendere le righe dove la conversione è fallita e leggere la loro anzianità. Il risultato è un solo valore, zero. Tutti e undici quei clienti sono al loro primo mese e non hanno ancora ricevuto una fattura, quindi la cella nel file originale era vuota, e una cella vuota in mezzo a una colonna di numeri basta a far leggere a pandas l’intera colonna come testo.

Adesso la scelta. Riempiamo con *fillna(0)* e non con la media, perché il totale pagato da chi non ha ancora pagato nulla è zero e non è un dato mancante. Mettere la media inventerebbe una storia di pagamenti che non è mai esistita, e per di più la metterebbe proprio sui clienti nuovi, che come vedremo tra poco sono i più a rischio di abbandono. Sarebbe un errore piccolo nella scrittura e grande nelle conseguenze.

Le colonne vanno trattate in due modi diversi a seconda di cosa contengono, quindi conviene separarle subito in due liste e usare quelle da qui in avanti invece di riscrivere i nomi ogni volta.

```python
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
```

Sono quindici colonne categoriche e quattro numeriche, e la somma fa diciannove invece di ventuno. Le due che mancano sono escluse apposta. *Churn* resta fuori perché è la risposta che vogliamo prevedere e non può stare tra le domande, altrimenti il modello imparerebbe a copiare invece che a prevedere. *customerID* resta fuori perché è un codice identificativo, ha 7043 valori tutti diversi e non contiene nessuna informazione sul comportamento del cliente. Se la codificassimo insieme alle altre otterremmo 7043 colonne nuove, una per cliente, e il modello imparerebbe a memoria il dataset senza generalizzare nulla.

Notiamo che *SeniorCitizen* sta tra le numeriche pur essendo di fatto un sì o no. Ci sta perché è già scritta come 0 e 1, cioè è già nella forma in cui la vorrebbe un modello, e codificarla di nuovo produrrebbe due colonne che dicono la stessa cosa.

Adesso separiamo le colonne che useremo per prevedere da quella che vogliamo prevedere.

```python
X = df[num_cols + cat_cols]
y = df['Churn'].map({'Yes': 1, 'No': 0})
```

La prima riga costruisce X selezionando le diciannove colonne, e l’ordine con cui le elenchiamo non è casuale. Prima le numeriche e poi le categoriche, perché è lo stesso ordine in cui più avanti ricomporremo le feature trasformate, ed è lo stesso ordine che dovrà rispettare l’applicazione finale quando riceverà un cliente nuovo. Un modello non conosce i nomi delle colonne, conosce solo le posizioni, quindi due colonne scambiate producono previsioni sbagliate senza nessun messaggio di errore.

La seconda riga trasforma il target da testo a numero con *map*, che sostituisce ogni valore secondo il dizionario che gli passiamo. Assegniamo 1 a Yes e 0 a No, e non il contrario, perché in scikit-learn la classe 1 è quella positiva, cioè quella di cui parlano precision e recall quando le leggeremo. Positiva qui non vuol dire buona, vuol dire che è l’evento che ci interessa individuare. Se invertissimo i due numeri tutte le metriche continuerebbero a funzionare, ma ci racconterebbero quanto siamo bravi a riconoscere i clienti che restano, che è la domanda che non ci ha fatto nessuno.

Prima di guardare qualsiasi grafico controlliamo com’è divisa la colonna che vogliamo prevedere.

```python
df['Churn'].value_counts(normalize=True)
```

```
OUTPUT:
Churn
No     0.73463
Yes    0.26537
Name: proportion, dtype: float64
```

Il parametro *normalize=True* chiede le proporzioni invece dei conteggi, quindi leggiamo direttamente che il 73.5 per cento dei clienti resta e il 26.5 per cento se ne va. Le due classi non sono bilanciate, ma nemmeno estremamente sbilanciate come capiterebbe con le frodi o con una malattia rara, dove la classe positiva può stare sotto l’uno per cento.

Questo numero ha però una conseguenza immediata che conviene fissare adesso. Un modello che rispondesse sempre No, senza guardare nessuna colonna e senza imparare niente, indovinerebbe il 73.5 per cento delle volte. Qualsiasi modello vero dovrà fare meglio di così solo per essere considerato inutile allo stesso modo, e l’accuratezza da sola non riuscirà a distinguere i due casi. Torneremo su questo punto quando confronteremo i classificatori, e sarà il momento più importante del capitolo.

Adesso esploriamo i dati. Non disegniamo un istogramma per colonna come si fa di solito, perché con diciannove colonne otterremmo diciannove grafici e nessuna risposta. Ne facciamo tre, e ognuno risponde a una domanda che ci porremmo comunque prima o poi. Quando abbandonano i clienti, quali caratteristiche fanno la differenza, e se esiste una combinazione di caratteristiche più pericolosa delle altre.

Il primo grafico guarda l’anzianità, cioè da quanti mesi il cliente è con l’operatore.

```python
churn_by_tenure = df.groupby(pd.cut(df['tenure'], bins=range(0, 81, 6)))['Churn'].apply(lambda s: (s == 'Yes').mean())
churn_by_tenure.plot(marker='o', color='#d1495b', figsize=(8, 4), title='Rischio di abbandono per anzianità')
plt.show()
```

[fig1.png]
*Figura 5.1 Il tasso di abbandono per fascia di anzianità. La curva parte oltre il 50 per cento nel primo semestre e scende sotto il 6 per cento dopo cinque anni.*

La riga è densa, quindi la leggiamo da dentro verso fuori. *pd.cut* prende la colonna *tenure*, che contiene numeri da 0 a 72, e la taglia in fasce usando i confini che gli passiamo con *bins*. Scriviamo *range(0, 81, 6)* per ottenere fasce da sei mesi, cioè da 0 a 6, da 6 a 12 e così via, e arriviamo a 81 invece che a 72 perché *range* esclude l’ultimo valore e vogliamo essere sicuri di coprire anche i clienti più anziani.

Il risultato di *pd.cut* è una colonna di etichette, una per fascia, e la passiamo direttamente a *groupby*, che quindi raggruppa i clienti per anzianità invece che per un valore già presente nel dataset. Su ogni gruppo prendiamo la colonna *Churn* e applichiamo la lambda.

La lambda merita una riga a parte perché è il trucco che useremo anche nel grafico successivo. L’espressione *(s == ’Yes’)* trasforma la colonna di testo in una colonna di veri e falsi, e in Python un vero vale 1 e un falso vale 0. La media di quella colonna è quindi il numero di Yes diviso il totale, cioè esattamente il tasso di abbandono della fascia. Ci risparmia di contare i Yes, contare i No, sommarli e dividere, e sono quattro passaggi in meno dove sbagliare.

Il grafico mostra una curva che parte alta e scende. Nei primi sei mesi abbandona più di un cliente su due, tra i sei e i dodici mesi meno di uno su tre, e dopo i cinque anni il rischio è intorno al cinque per cento. Chi resta abbastanza a lungo tende a restare per sempre. Questo ci dice già una cosa pratica, cioè che le azioni di retention hanno senso concentrate nei primi mesi del rapporto e non distribuite su tutta la base clienti.

Il secondo grafico guarda le quindici colonne categoriche tutte insieme e cerca quali categorie fanno abbandonare più della media.

```python
rates = pd.concat([df.groupby(c)['Churn'].apply(lambda s: (s == 'Yes').mean()) for c in cat_cols], keys=cat_cols)
top = (rates / df['Churn'].eq('Yes').mean()).sort_values().pipe(lambda s: pd.concat([s.head(6), s.tail(6)]))
top.plot.barh(color=['#2986cc' if v < 1 else '#d1495b' for v in top], figsize=(7, 6), title='Driver del churn (lift vs media)')
plt.show()
```

[fig2.png]
*Figura 5.2 Le sei categorie che riducono di più il rischio e le sei che lo aumentano di più, misurate come rapporto rispetto al tasso medio di abbandono.*

La prima riga contiene una comprensione di lista che ripete per ogni colonna categorica lo stesso calcolo del grafico precedente, cioè raggruppa e prende la media della condizione booleana. Otteniamo una serie per colonna, e *pd.concat* le unisce in una sola. Il parametro *keys=cat_cols* è quello che rende leggibile il risultato, perché costruisce un indice a due livelli dove il primo è il nome della colonna e il secondo è la categoria, quindi sappiamo sempre da dove viene ogni valore. Senza *keys* ci ritroveremmo con una lunga lista di categorie senza sapere a quale colonna appartengono, e alcune si chiamano allo stesso modo in colonne diverse.

La seconda riga fa il lavoro concettuale. Dividiamo il tasso di ogni categoria per il tasso medio generale, che è quel 26.5 per cento visto prima, e otteniamo un numero che in statistica si chiama lift. Un lift di 1.5 vuol dire che quella categoria abbandona una volta e mezza più della media, un lift di 0.5 vuol dire che abbandona la metà. Usiamo il rapporto invece della differenza perché così le categorie sono confrontabili tra loro anche se partono da basi diverse.

Poi ordiniamo con *sort_values* e teniamo solo i due estremi, le sei categorie più basse e le sei più alte. Lo facciamo con *pipe*, che passa la serie alla funzione che gli scriviamo dentro e ci permette di restare su una sola catena invece di creare una variabile intermedia. Teniamo solo gli estremi perché il grafico deve mostrare cosa sposta davvero il rischio, e le decine di categorie che stanno intorno a 1 non spostano niente, in mezzo a loro le poche importanti diventerebbero invisibili.

La terza riga disegna, e il colore lo decidiamo con una comprensione di lista che attraversa i valori e sceglie il blu sotto 1 e il rosso sopra 1. È una scelta di leggibilità, perché la posizione della barra dice già la stessa cosa, ma il colore la dice prima che il lettore guardi l’asse.

In alto troviamo il pagamento con assegno elettronico, il contratto mensile, la fibra ottica, l’assenza di sicurezza online e l’assenza di assistenza tecnica. In basso il contratto biennale, che abbandona a circa un decimo della media ed è di gran lunga il segnale più forte del dataset. Notiamo anche che diverse categorie in basso valgono tutte 0.28 e si chiamano tutte *No internet service*, perché sono le stesse persone viste da colonne diverse, cioè i clienti senza internet che ovviamente non hanno né il backup né lo streaming. Questo grafico anticipa quali colonne peseranno di più nel modello, e più avanti potremo verificare se avevamo visto giusto.

Il terzo grafico mette insieme due colonne numeriche per cercare una combinazione pericolosa, cosa che i primi due non potevano fare perché guardavano una variabile alla volta.

```python
colors = df['Churn'].map({'Yes': '#d1495b', 'No': '#2986cc'})
df.plot.scatter(x='tenure', y='MonthlyCharges', c=colors, alpha=0.4, figsize=(7, 5), title='Zona di rischio churn')
plt.show()
```

[fig3.png]
*Figura 5.3 Ogni punto è un cliente. In rosso chi ha abbandonato, in blu chi è rimasto. Il rosso si addensa nella fascia alta a sinistra, cioè poca anzianità e bolletta cara.*

La prima riga riusa *map*, ma stavolta non per convertire in numeri, per convertire in colori. Otteniamo una colonna lunga quanto il dataset che contiene il codice del rosso o del blu per ogni cliente, e la passiamo a *plot.scatter* come parametro *c*, che accetta sia un colore singolo sia una lista di colori punto per punto.

Il parametro *alpha=0.4* non è un dettaglio estetico. Stiamo disegnando 7043 punti in uno spazio piccolo, quindi molti finiscono uno sopra l’altro, e senza trasparenza vedremmo solo l’ultimo disegnato e perderemmo completamente l’informazione sulla densità. Con la trasparenza le zone dove i punti si accumulano diventano più scure, e la densità è esattamente quello che ci interessa vedere.

Il rosso si addensa a sinistra, cioè tra i clienti nuovi, ma non su tutta la colonna sinistra. Si concentra nella parte alta, dove la bolletta mensile supera i settanta euro, mentre in basso a sinistra, sotto i trenta euro, il blu resta maggioranza netta. Sono i clienti con la sola linea telefonica, che entrano, pagano poco e non se ne vanno.

Non è quindi la poca anzianità da sola, ed è facile verificarlo. Fra i clienti che stanno con l’operatore da meno di un anno, chi paga meno di trentacinque euro al mese abbandona nel 24 per cento dei casi, e chi ne paga più di ottantacinque abbandona nel 74. La stessa anzianità, tre volte il rischio, e a fare la differenza è il prezzo. È la combinazione delle due cose che conta, non una delle due. Il primo grafico non poteva dirlo perché guardava l’anzianità isolata dal prezzo, il secondo nemmeno perché guardava solo colonne di testo. È il motivo per cui tre grafici scelti battono venti grafici fatti per abitudine.

Sappiamo cosa contengono i dati e sappiamo cosa aspettarci. Possiamo prepararli per il modello.
