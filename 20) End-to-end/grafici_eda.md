# I tre grafici dell'analisi esplorativa

## Grafico 1. Il rischio di abbandono per anzianità

Il codice raggruppa i clienti in fasce di sei mesi di tenure con `pd.cut`, poi calcola per ogni fascia la percentuale di clienti che ha abbandonato. La lambda `lambda s: (s == 'Yes').mean()` funziona perché una serie booleana ha come media la proporzione di valori True, quindi restituisce subito il tasso di churn della fascia, senza bisogno di contare a mano quanti Yes e quanti No ci sono.

Il grafico mostra una linea che parte alta nei primi mesi e scende con l'anzianità. I clienti nuovi sono quelli più a rischio, mentre chi resta abbastanza a lungo tende a restare per sempre. Questo suggerisce che le azioni di retention vanno concentrate nei primi mesi del rapporto, non distribuite in modo uniforme su tutta la base clienti.

## Grafico 2. I driver del churn

Qui il codice cicla su tutte le colonne categoriche in `cat_cols` e per ciascuna calcola il tasso di churn di ogni categoria, con lo stesso trucco della media su una serie booleana usato nel grafico 1. `pd.concat` unisce tutti questi risultati in un'unica serie, e `keys=cat_cols` tiene traccia di quale colonna ha generato ogni valore.

Il tasso di ogni categoria viene poi diviso per il tasso medio generale, così si ottiene un lift. Un valore sopra 1 vuol dire che quella categoria abbandona più della media, un valore sotto 1 vuol dire che abbandona meno. La serie viene ordinata e vengono tenute solo le sei categorie con il lift più basso e le sei con il lift più alto, perché il grafico deve mostrare solo gli estremi, cioè le caratteristiche che spostano davvero il rischio, invece di annegarle tra decine di categorie poco significative.

Il colore distingue le due direzioni, rosso per chi aumenta il rischio e blu per chi lo riduce. Alcune combinazioni di servizi o tipi di contratto sono quindi molto più predittive di altre, e questo grafico anticipa quali feature probabilmente conteranno di più nel modello di classificazione.

## Grafico 3. La zona di rischio

Il codice mappa ogni valore di Churn a un colore con `df['Churn'].map(...)`, poi passa quella mappatura come colore dei punti a `df.plot.scatter`. Ogni punto è un cliente, la posizione indica tenure e bolletta mensile, il colore indica se ha abbandonato oppure no.

Il grafico mostra una concentrazione di punti rossi nell'angolo in basso a sinistra, dove si trovano i clienti con poca anzianità e bollette relativamente alte. La combinazione di questi due fattori, non uno dei due preso da solo, sembra la zona più pericolosa. È un'informazione che il grafico 1 da solo non poteva dare, perché guardava la tenure isolata dalla bolletta.
