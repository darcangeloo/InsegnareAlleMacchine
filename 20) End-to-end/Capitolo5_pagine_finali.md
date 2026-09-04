## Dove andare adesso


Sei arrivato in fondo, quindi adesso sai fare una cosa precisa. Prendere un file che non hai mai visto, capire cosa contiene e cosa è rotto, esplorarlo abbastanza da farti un’idea prima di modellare, scegliere un algoritmo, giudicarlo con una metrica che corrisponde al problema vero e non a quella che viene di default, e infine metterlo dove altri programmi possono usarlo. È il percorso completo, ed è quello che in un’azienda chiamano fare machine learning.

Quello che non sai ancora fare è altrettanto preciso. Non hai toccato le reti neurali, quindi non hai strumenti per le immagini, per il testo, per l’audio o per qualsiasi dato in cui l’ordine conta. Sono esattamente le cose di cui si parla quando si dice intelligenza artificiale, e capisco che sia il motivo per cui molti aprono un libro come questo. Il punto è che ci si arriva da qui.

Tutto quello che hai imparato resta vero anche là. La divisione tra addestramento e test non cambia. Imparare le trasformazioni solo sul train e applicarle al resto non cambia. Lo sbilanciamento delle classi non cambia, l’overfitting non cambia, e soprattutto non cambia il fatto che scegliere la metrica è una decisione sul problema e non un dettaglio tecnico. Il deep learning cambia il modello, non il metodo. Chi ci arriva senza aver capito queste cose costruisce reti enormi e non sa dire se funzionano.

Se vuoi continuare, due libri fanno il lavoro. *Deep Learning with PyTorch* di Eli Stevens ti dà la meccanica, cioè i tensori, il calcolo automatico del gradiente e il ciclo di addestramento scritto a mano, che è il punto in cui capisci davvero cosa succedeva dietro il *fit* di scikit-learn. Poi *Build a Large Language Model from Scratch* di Sebastian Raschka, che costruisce un transformer pezzo per pezzo invece di spiegartelo. In quest’ordine, perché il secondo dà per scontato quello che insegna il primo.

Il consiglio su come leggerli è lo stesso che ti ho dato all’inizio del capitolo 5. Non leggerli. Scrivi il codice mentre li leggi, sbaglia, rilancia. Un libro tecnico letto sul divano si dimentica in due settimane.

E se vuoi un progetto per cominciare, ne ho uno che ti consiglio più di tanti esercizi inventati. Riprendi il dataset di questo capitolo, quello del churn, e prova a risolverlo con una piccola rete neurale in PyTorch invece che con il gradient boosting. Confronta le due f1 sulla stessa classe e sullo stesso test set.

Ti anticipo il risultato più probabile, perché è la parte interessante. Sui dati tabellari, cioè righe e colonne come quelli che abbiamo usato qui, gli alberi potenziati vincono quasi sempre contro una rete neurale di dimensioni ragionevoli. Non è un’opinione mia, è una cosa che si ripete in gran parte dei confronti pubblicati. Scoprirlo con le tue mani, dopo aver fatto la fatica di costruire la rete, vale più di qualunque frase che potrei scriverti qui, perché ti insegna la cosa che questo libro ha cercato di dirti dall’inizio. Lo strumento più nuovo non è lo strumento giusto, il problema decide.

---


## Fonti e risorse


Questo libro non nasce dal nulla. Ho imparato queste cose da documentazione, corsi universitari aperti al pubblico e canali divulgativi, e mi sembra giusto dire quali, per due motivi. Il primo è che una parte di quello che hai letto viene da lì. Il secondo è che se una spiegazione non ti è bastata, qui trovi dove andare a leggerla scritta da qualcun altro.

**Documentazione**

- scikit-learn, guida utente. scikit-learn.org/stable/user_guide.html

- pandas, documentazione. pandas.pydata.org/docs

- NumPy, documentazione. numpy.org/doc/stable

- Matplotlib, documentazione. matplotlib.org/stable

- FastAPI, documentazione. fastapi.tiangolo.com

La guida utente di scikit-learn è la fonte più importante di tutte, e non è un rimando di cortesia. Ogni parametro che ho spiegato in questo libro è documentato lì, spesso con la spiegazione matematica accanto, e imparare a leggerla è parte del mestiere quanto imparare gli algoritmi.

**Corsi**

- Stanford CS229, Machine Learning, Andrew Ng. cs229.stanford.edu «da confermare»

- MIT 6.036, Introduction to Machine Learning, oggi 6.390. ocw.mit.edu/courses/6-036-introduction-to-machine-learning-fall-2020 «da confermare»

- MIT 18.06, Linear Algebra, Gilbert Strang. ocw.mit.edu/courses/18-06-linear-algebra-spring-2010 «da confermare»

**Divulgazione e video**

- IBM Think, sezione Topics. ibm.com/think/topics

- GeeksforGeeks, sezione Machine Learning. geeksforgeeks.org/machine-learning

- StatQuest with Josh Starmer, YouTube. «da confermare»

- 3Blue1Brown, YouTube. «da confermare»

**Dataset**

- Telco Customer Churn, capitolo 5. Dati di esempio di IBM, distribuiti su Kaggle. kaggle.com/datasets/blastchar/telco-customer-churn «licenza da verificare»

- Mall Customers, capitolo 4. Kaggle. «link e licenza da verificare»

- Instagram visits clustering, capitolo 4. Kaggle. «link e licenza da verificare»

I dataset sono di chi li ha pubblicati, non miei. Nel repository del libro trovi per ognuno la fonte e la licenza, e dove la licenza non permette la ridistribuzione trovi lo script che lo scarica al posto del file.

---


## Il codice del libro


Tutto il codice di questo libro è pubblico e scaricabile, notebook per notebook, insieme ai dataset che abbiamo usato e alla cartella dell’applicazione del capitolo 5.

github.com/[UTENTE]/insegnare-alle-macchine

Dentro trovi una cartella per capitolo. I notebook sono quelli veri, con gli output già eseguiti, quindi puoi confrontare i tuoi numeri con i miei riga per riga. Nella cartella del capitolo 5 c’è anche *app* con *api.py* e i tre modelli salvati, pronta da lanciare con uvicorn.

Se trovi un errore nel testo o nel codice, aprire una segnalazione nel repository è il modo più veloce per farmelo sapere, e la correzione finisce nella prossima edizione con il tuo nome accanto.

---


## Ringraziamenti


A Rebecca, che ha letto ogni pagina di questo libro cercando gli errori che io non vedevo più, con una pazienza che non le ho mai restituito. Metà delle virgole al posto giusto sono sue.

A mio padre, che mi ha insegnato a trattare i progetti come lavoro e non come passatempo, e che mi ha detto di no abbastanza volte da farmi capire quando invece era sì.

E a chi ha comprato questo libro senza sapere niente di me. Scrivere qualcosa di lungo quando nessuno te lo ha chiesto è un atto di fede, e tu sei la ragione per cui ha funzionato.

---


## Una recensione


Se questo libro ti è servito, lasciare una recensione onesta su Amazon aiuta altre persone a capire se fa per loro. Bastano due righe, e contano più di quanto immagini per un libro pubblicato da una persona sola.

Se invece qualcosa non ha funzionato, scrivimelo prima di scriverlo lì. Se posso sistemarlo, lo sistemo.


## Dove trovarmi


Pubblico contenuti sul machine learning su TikTok, con lo stesso approccio di questo libro, cioè un concetto alla volta spiegato per intero.

TikTok, @macchinecheimparano

Lavoro anche su Sentia, un prodotto di intelligenza artificiale che sviluppo per conto mio, e da lì si passa per scrivermi.

asksentia.com
