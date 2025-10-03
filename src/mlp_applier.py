import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def mlp_apply(X_train, Y_train, X_val, Y_val):

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),   
        Dropout(0.4),                                                     
        Dense(64, activation='relu'),                                     
        Dropout(0.3),
        Dense(32, activation='relu'),                                     
        Dropout(0.2),
        Dense(8, activation='relu'),                                     
        Dropout(0.1),
        Dense(1, activation='sigmoid')                                  
    ])


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True
    )


    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=200,             
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred_prob = model.predict(X_val).ravel()
    y_pred_class = (y_pred_prob >= 0.5).astype(int)

    
    print("Classification Report:\n", classification_report(Y_val, y_pred_class))
    print("Confusion Matrix:\n", confusion_matrix(Y_val, y_pred_class))
    print("ROC-AUC Score:", roc_auc_score(Y_val, y_pred_prob))