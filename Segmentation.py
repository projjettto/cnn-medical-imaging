def run_segmentation():
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
    from keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    import cv2
    import tensorflow as tf
    from glob import glob
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import BatchNormalization
    
    
    def load_images_from_folder(folder, ext='.jpeg', target_size=(256, 256)):
        images = []
        tumor_filenames = []
        for subfolder in ['T1', 'T1C+', 'T2']:
            subfolder_path = os.path.join(folder, subfolder)
            img_paths = glob(os.path.join(subfolder_path, f'*{ext}'))
            
            print(f"Categoria {subfolder} - Immagini trovate: {len(img_paths)}")  # Diagnostica per conteggio immagini
            
            for img_path in img_paths:
                img = plt.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, target_size)
                    if img_resized.ndim == 3 and img_resized.shape[2] == 3:  # RGB
                        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
                    else:
                        img_gray = img_resized
                    images.append(img_gray)
                    tumor_filenames.append(os.path.basename(img_path))
        
        # Stampa diagnostica finale per controllo totale
        print(f"Totale immagini caricate: {len(images)}")
        return np.array(images), tumor_filenames
    
    
    # Funzione per caricare le maschere
    def load_masks_from_folder(folder, tumor_filenames, target_size=(256, 256)):
        masks = []
        count_T1 = 0
        count_T1C = 0
        count_T2 = 0
        
        for filename in tumor_filenames:
            if "T1C+" in filename:
                mask_subfolder = 'T1C+'
                count_T1C += 1
            elif "T1" in filename and "T1C+" not in filename:
                mask_subfolder = 'T1'
                count_T1 += 1
            else:
                mask_subfolder = 'T2'
                count_T2 += 1
                
            subfolder_path = os.path.join(folder, mask_subfolder)
            mask_path = os.path.join(subfolder_path, filename.replace('.jpeg', '.tif'))
            
            if os.path.exists(mask_path):
                img = plt.imread(mask_path)
                if img is not None:
                    img_resized = cv2.resize(img, target_size)
                    masks.append(img_resized)
            else:
                print(f"Mask not found for {filename}")
                
        # Stampa il numero di maschere caricate
        print("Categoria T1   - Immagini trovate: ", count_T1)
        print("Categoria T1C+ - Immagini trovate: ", count_T1C)
        print("Categoria T2   - Immagini trovate: ", count_T2)
        print(f"Numero di maschere caricate: {len(masks)}")
        
        
        return np.array(masks)
    
    # Caricamento delle immagini tumorali (.jpeg) e delle maschere (.tif)
    tumor_images, tumor_filenames = load_images_from_folder('datasetSeg2/train', ext='.jpeg', target_size=(256, 256))
    masks = load_masks_from_folder('datasetSeg2/mask', tumor_filenames, target_size=(256, 256))
    
    # Normalizzazione
    tumor_images = tumor_images / 255.0
    masks = masks / 255.0
    
    # Se le maschere sono in scala di grigi
    if masks.ndim == 4:
        masks = np.squeeze(masks, axis=-1)
    
    # Filtra le immagini e le maschere per ciascuna categoria
    t1_images = [img for img, name in zip(tumor_images, tumor_filenames) if "T1C+" not in name and "T1" in name]
    t1c_plus_images = [img for img, name in zip(tumor_images, tumor_filenames) if "T1C+" in name]
    t2_images = [img for img, name in zip(tumor_images, tumor_filenames) if "T2" in name]
    
    t1_masks = [mask for mask, name in zip(masks, tumor_filenames) if "T1C+" not in name and "T1" in name]
    t1c_plus_masks = [mask for mask, name in zip(masks, tumor_filenames) if "T1C+" in name]
    t2_masks = [mask for mask, name in zip(masks, tumor_filenames) if "T2" in name]
    
    # Suddivisione del dataset
    X_train_t1, X_val_t1, y_train_t1, y_val_t1 = train_test_split(t1_images, t1_masks, test_size=0.2, random_state=42)
    X_train_t1c, X_val_t1c, y_train_t1c, y_val_t1c = train_test_split(t1c_plus_images, t1c_plus_masks, test_size=0.2, random_state=42)
    X_train_t2, X_val_t2, y_train_t2, y_val_t2 = train_test_split(t2_images, t2_masks, test_size=0.2, random_state=42)
    
    # Unione del training e validation set
    X_train = np.array(X_train_t1 + X_train_t1c + X_train_t2, dtype=np.float32)
    y_train = np.array(y_train_t1 + y_train_t1c + y_train_t2, dtype=np.uint8)
    X_val = np.array(X_val_t1 + X_val_t1c + X_val_t2, dtype=np.float32)
    y_val = np.array(y_val_t1 + y_val_t1c + y_val_t2, dtype=np.uint8)
    
    # Reshape per adattare le immagini
    X_train = X_train[..., np.newaxis]
    y_train = y_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    y_val = y_val[..., np.newaxis]
    
    # # Visualizza la distribuzione finale delle categorie nel training e validation set con i nomi delle immagini e le maschere
    # print("\nDistribuzione delle categorie nel training e validation set:")
    
    # # Per T1
    # print(f"\nT1 in training: {len(X_train_t1)} immagini")
    # print("Nomi delle immagini in X_train_t1:", [tumor_filenames[i] for i, img in enumerate(t1_images) if any(np.array_equal(img, x) for x in X_train_t1)])
    # print("Nomi delle maschere in X_train_t1:", [tumor_filenames[i].replace('.jpeg', '_mask.tif') for i, img in enumerate(t1_images) if any(np.array_equal(img, x) for x in X_train_t1)])
    
    # print(f"\nT1 in validation: {len(X_val_t1)} immagini")
    # print("Nomi delle immagini in X_val_t1:", [tumor_filenames[i] for i, img in enumerate(t1_images) if any(np.array_equal(img, x) for x in X_val_t1)])
    # print("Nomi delle maschere in X_val_t1:", [tumor_filenames[i].replace('.jpeg', '_mask.tif') for i, img in enumerate(t1_images) if any(np.array_equal(img, x) for x in X_val_t1)])
    
    # # Per T1C+
    # print(f"\nT1C+ in training: {len(X_train_t1c)} immagini")
    # print("Nomi delle immagini in X_train_t1c:", [tumor_filenames[i] for i, img in enumerate(t1c_plus_images) if any(np.array_equal(img, x) for x in X_train_t1c)])
    # print("Nomi delle maschere in X_train_t1c:", [tumor_filenames[i].replace('.jpeg', '_mask.tif') for i, img in enumerate(t1c_plus_images) if any(np.array_equal(img, x) for x in X_train_t1c)])
    
    # print(f"\nT1C+ in validation: {len(X_val_t1c)} immagini")
    # print("Nomi delle immagini in X_val_t1c:", [tumor_filenames[i] for i, img in enumerate(t1c_plus_images) if any(np.array_equal(img, x) for x in X_val_t1c)])
    # print("Nomi delle maschere in X_val_t1c:", [tumor_filenames[i].replace('.jpeg', '_mask.tif') for i, img in enumerate(t1c_plus_images) if any(np.array_equal(img, x) for x in X_val_t1c)])
    
    # # Per T2
    # print(f"\nT2 in training: {len(X_train_t2)} immagini")
    # print("Nomi delle immagini in X_train_t2:", [tumor_filenames[i] for i, img in enumerate(t2_images) if any(np.array_equal(img, x) for x in X_train_t2)])
    # print("Nomi delle maschere in X_train_t2:", [tumor_filenames[i].replace('.jpeg', '_mask.tif') for i, img in enumerate(t2_images) if any(np.array_equal(img, x) for x in X_train_t2)])
    
    # print(f"\nT2 in validation: {len(X_val_t2)} immagini")
    # print("Nomi delle immagini in X_val_t2:", [tumor_filenames[i] for i, img in enumerate(t2_images) if any(np.array_equal(img, x) for x in X_val_t2)])
    # print("Nomi delle maschere in X_val_t2:", [tumor_filenames[i].replace('.jpeg', '_mask.tif') for i, img in enumerate(t2_images) if any(np.array_equal(img, x) for x in X_val_t2)])
        

  # Creazione del modello U-Net per la segmentazione
    def unet_model(input_size=(256, 256, 1)):
        inputs = Input(input_size)
     
        # Encoder
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        # conv1 = BatchNormalization()(conv1)  # Aggiungi BatchNormalization
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        # conv1 = BatchNormalization()(conv1)  # Aggiungi BatchNormalization
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # pool1 = Dropout(0.1)(pool1)  # Dropout
       
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        # conv2 = BatchNormalization()(conv2)  # Aggiungi BatchNormalization
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        # conv2 = BatchNormalization()(conv2)  # Aggiungi BatchNormalization
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # pool2 = Dropout(0.1)(pool2)  # Dropout
       
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        # conv3 = BatchNormalization()(conv3)  # Aggiungi BatchNormalization
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        # conv3 = BatchNormalization()(conv3)  # Aggiungi BatchNormalization
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # pool3 = Dropout(0.2)(pool3)  # Dropout
       
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        # conv4 = BatchNormalization()(conv4)  # Aggiungi BatchNormalization
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        # conv4 = BatchNormalization()(conv4)  # Aggiungi BatchNormalization
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        # pool4 = Dropout(0.2)(pool4)  # Dropout
       
        # Bottleneck
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        # conv5 = BatchNormalization()(conv5)  # Aggiungi BatchNormalization
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        # conv5 = BatchNormalization()(conv5)  # Aggiungi BatchNormalization
        # conv5 = Dropout(0.3)(conv5)  # Dropout
       
        # Decoder
        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
        # conv6 = BatchNormalization()(conv6)  # Aggiungi BatchNormalization
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
        # conv6 = BatchNormalization()(conv6)  # Aggiungi BatchNormalization
        # conv6 = Dropout(0.2)(conv6)  # Dropout
       
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
        # conv7 = BatchNormalization()(conv7)  # Aggiungi BatchNormalization
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
        # conv7 = BatchNormalization()(conv7)  # Aggiungi BatchNormalization
        # conv7 = Dropout(0.2)(conv7)  # Dropout
        
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
        # conv8 = BatchNormalization()(conv8)  # Aggiungi BatchNormalization
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
        # conv8 = BatchNormalization()(conv8)  # Aggiungi BatchNormalization
        # conv8 = Dropout(0.1)(conv8)  # Dropout
        
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
        # conv9 = BatchNormalization()(conv9)  # Aggiungi BatchNormalization
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
        # conv9 = BatchNormalization()(conv9)  # Aggiungi BatchNormalization
        # conv9 = Dropout(0.1)(conv9)  # Dropout
       
        # Output layer
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    
        model = Model(inputs=[inputs], outputs=[conv10])
    
        # model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
        return model
    
    # Callback
    checkpoint = ModelCheckpoint('best_model_seg.keras', monitor='val_loss', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    # Data augmentation
    data_gen_args = dict(rotation_range=10,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     fill_mode='nearest')
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    batch_size = 8
    
    # Generatore di batch
    def combined_generator(image_gen, mask_gen, imgs, masks, batch_size):
        image_gen = image_gen.flow(imgs, batch_size=batch_size, seed=42)
        mask_gen = mask_gen.flow(masks, batch_size=batch_size, seed=42)
        for img_batch, mask_batch in zip(image_gen, mask_gen):
            yield img_batch, mask_batch

            
    # Funzione per calcolare il Dice Coefficient
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        y_true_f = K.flatten(tf.cast(y_true, dtype=tf.float32))  # Converti y_true in float32
        y_pred_f = K.flatten(y_pred)  # y_pred è già in float32
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    # Generatori
    train_generator = combined_generator(image_datagen, mask_datagen, X_train, y_train, batch_size)
    val_generator = combined_generator(ImageDataGenerator(), ImageDataGenerator(), X_val, y_val, batch_size)

    
    # Compilazione del modello
    model = unet_model(input_size=(256, 256, 1))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', dice_coefficient])
    
    # Addestramento del modello
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size

    history = model.fit(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_generator,
                        validation_steps=validation_steps,
                        epochs=50,
                        callbacks=[checkpoint, reduce_lr])

    
    # Valutazione finale
    loss, accuracy, dice = model.evaluate(val_generator, steps=validation_steps)
    print(f'\nValutazione Finale - Loss: {loss}, Accuracy: {accuracy}, Dice Coefficient: {dice}')
    
    
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
    
        # Coefficiente Dice
        plt.plot(history.history['dice_coefficient'], label='Train Dice Coefficient')
        plt.plot(history.history['val_dice_coefficient'], label='Validation Dice Coefficient')
        plt.title('Dice Coefficient over Epochs')
        plt.ylabel('Dice Coefficient')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.show()
        
    plot_metrics(history)   

    # Funzione per visualizzare maschere predette
    def plot_predicted_masks(model, X_val_t1, X_val_t1c, X_val_t2, y_val_t1, y_val_t1c, y_val_t2, num_masks=3):
        categories = [('T1', X_val_t1, y_val_t1), ('T1C+', X_val_t1c, y_val_t1c), ('T2', X_val_t2, y_val_t2)]
        
        for category, X_val_cat, y_val_cat in categories:
            print(f"\nDisplaying predicted masks for category: {category}")
            plt.figure(figsize=(12, 12))
            
            X_val_cat = np.array(X_val_cat)
            y_val_cat = np.array(y_val_cat)
            
            if X_val_cat.ndim == 3:
                X_val_cat = np.expand_dims(X_val_cat, -1)
            if y_val_cat.ndim == 3:
                y_val_cat = np.expand_dims(y_val_cat, -1)
                
            for i in range(min(num_masks, len(X_val_cat))):
                img = X_val_cat[i:i+1]
                pred_mask = model.predict(img)[0, :, :, 0]
                
                plt.subplot(num_masks, 3, i * 3 + 1)
                plt.imshow(img[0, :, :, 0], cmap='gray')
                plt.title(f"{category} Image {i+1}")
                plt.axis('off')
    
                plt.subplot(num_masks, 3, i * 3 + 2)
                plt.imshow(y_val_cat[i, :, :, 0], cmap='gray')
                plt.title("True Mask")
                plt.axis('off')
    
                plt.subplot(num_masks, 3, i * 3 + 3)
                plt.imshow(pred_mask, cmap='gray')
                plt.title("Predicted Mask")
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    # Chiamata della funzione per visualizzare le maschere
    plot_predicted_masks(model, X_val_t1, X_val_t1c, X_val_t2, y_val_t1, y_val_t1c, y_val_t2, num_masks=3)
