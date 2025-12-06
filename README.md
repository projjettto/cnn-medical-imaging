# Progetto di Classificazione e Segmentazione Tumori

Questo progetto utilizza modelli di deep learning per eseguire due fasi principali su immagini di tumori: la **classificazione** tra immagini contenenti medulloblastomi e immagini senza tumore, e la **segmentazione** delle aree tumorali nelle immagini identificate come positive.

## Struttura del Progetto

Il progetto è organizzato in tre file principali:
- `main.py`: il file principale da cui avviare l'esecuzione del progetto.
- `Classification.py`: contiene la funzione `run_classification()` per eseguire il modello di classificazione.
- `Segmentation.py`: contiene la funzione `run_segmentation()` per eseguire il modello di segmentazione.

## Esecuzione del Progetto

Per eseguire il progetto, assicurati di avere installato le dipendenze necessarie. Puoi installarle tutte utilizzando `pip` con il comando seguente:

pip install numpy pandas matplotlib seaborn opencv-python scikit-learn keras tensorflow

Quindi avvia il file `main.py`.
