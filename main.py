# Importa le funzioni principali dagli script classification.py e segmentation.py
from Classification import run_classification
from Segmentation import run_segmentation

def main():
    # Esegue la classificazione
    print("Avvio della fase di classificazione...")
    print()
    run_classification()
    print()
    print("Fase di classificazione completata.")
    print()

    # Esegue la segmentazione
    print("Avvio della fase di segmentazione...")
    run_segmentation()
    print()
    print("Fase di segmentazione completata.")

if __name__ == "__main__":
    main()
