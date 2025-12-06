def run_classification():
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from tensorflow.keras.applications import VGG16
    import pandas as pd
    
    # Impostazioni del percorso
    train_dir = 'datasetClass/train'
    val_dir = 'datasetClass/val'
    test_dir = 'datasetClass/val'
    
    # Data Augmentation avanzata per il training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    
    # Preprocessing per il validation set
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    
    # Caricamento dei dati di training e validazione
    batch_size = 16
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Pesi di classe per bilanciare le classi
    class_weight = {0: 1.0, 1: 1.3}  # Aggiustare i pesi in base alle differenze nelle dimensioni delle classi
    
    # Carica il modello pre-addestrato VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Sblocca gli ultimi 4 layer del VGG16 per il fine-tuning
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    # Callback per salvare il miglior modello
    checkpoint = ModelCheckpoint('best_model_class.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    
    # Crea il modello con più layer densi e maggiore regolarizzazione
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),  # Regolarizzazione ridotta
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),  # Regolarizzazione ridotta
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),  # Regolarizzazione ridotta
        Dense(2, activation='softmax')  # Due classi: 'no_tumor' e 'medulloblastoma'
    ])
    
    # Compila il modello con una maggiore regolarizzazione
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Callback per ridurre il learning rate se le performance non migliorano
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
        
    # Addestramento del modello
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        class_weight=class_weight,  # Aggiunti pesi di classe
        callbacks=[checkpoint, reduce_lr]  
    )
    
    # Carica il miglior modello salvato
    model.load_weights('best_model_class.keras')
    
    # Visualizzazione delle metriche
    def plot_metrics(history):
        # Accuratezza
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.show()
    
        # Perdita (loss)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.show()
    
    # Visualizza metriche
    plot_metrics(history)
    
    # Previsioni sul validation set
    val_predictions = model.predict(val_generator)
    val_predictions = np.argmax(val_predictions, axis=1)
    
    # Ottenere le etichette reali dal validation set
    val_labels = val_generator.classes
    
    # Calcolo dell'accuratezza
    accuracy = accuracy_score(val_labels, val_predictions)
    print(f'Accuratezza del modello: {accuracy:.4f}')
    
    # Predizioni sul test set
    test_predictions = model.predict(test_generator)
    test_predictions = np.argmax(test_predictions, axis=1)

    # Etichette reali del test set
    test_labels = test_generator.classes

    # Calcolo della test accuracy
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f'\nAccuratezza sul test set: {test_accuracy:.4f}')
    
    # Creare la matrice di confusione
    conf_matrix = confusion_matrix(val_labels, val_predictions)
    
    # Visualizzare la matrice di confusione
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=val_generator.class_indices.keys(), 
                yticklabels=val_generator.class_indices.keys())
    plt.title('Matrice di Confusione')
    plt.ylabel('Etichette reali')
    plt.xlabel('Previsioni del modello')
    plt.show()
    
    # Stampare il report di classificazione
    classification_rep = classification_report(val_labels, val_predictions, target_names=val_generator.class_indices.keys(), output_dict=True)
    
    # Converti il report di classificazione in DataFrame e stampalo
    df_classification_report = pd.DataFrame(classification_rep).transpose()
    print("\nReport di classificazione in formato tabellare:")
    print(df_classification_report)
    
    
    # Funzione per visualizzare il report di classificazione in formato grafico
    def plot_classification_report(report_df):
        # Usare slicing per prendere solo le prime righe (le classi reali)
        class_report_df = report_df.iloc[:-3]  # Supponendo che le ultime 3 righe siano accuracy, macro avg, e weighted avg
        
        metrics = ['precision', 'recall', 'f1-score']
        
        plt.figure(figsize=(10, 6))
        class_report_df[metrics].plot(kind='bar')
        plt.title('Precision, Recall, and F1-Score per class')
        plt.ylabel('Score')
        plt.xlabel('Class')
        plt.xticks(rotation=0)
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.show()
        
    # Mostra il report di classificazione in forma di grafico
    plot_classification_report(df_classification_report)
    
    def print_final_stats(history, test_accuracy):
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        # Se viene fornito il test_accuracy, aggiungiamolo alla tabella
        data = {
            'Metric': ['Train Loss', 'Validation Loss', 'Train Accuracy', 'Validation Accuracy', 'Test Accuracy'],
            'Value': [final_train_loss, final_val_loss, final_train_acc, final_val_acc, test_accuracy] }
    
        df_stats = pd.DataFrame(data)
        
        # Stampa della tabella
        print("\nTabella con le statistiche finali:")
        print(df_stats)
        
        # Visualizzazione della tabella con matplotlib
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df_stats.values, colLabels=df_stats.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.show()
    
    # Stampa e visualizzazione delle statistiche finali
    print_final_stats(history, test_accuracy)
