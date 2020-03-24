import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()
    train_X = train_X.reshape(-1, 28, 28, 1).astype('float32') / 255
    test_X = test_X.reshape(-1, 28, 28, 1).astype('float32') / 255

    train_Y_one_hot = to_categorical(train_Y)
    test_Y_one_hot = to_categorical(test_Y)

    train_X, valid_X, train_label, valid_label = train_test_split(
        train_X, train_Y_one_hot, test_size=0.2,
        random_state=13
    )

    print('Data loaded')
    classes = np.unique(train_Y)
    nClasses = len(classes)

    fashion_model = Sequential()
    fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2), padding='same'))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(Dense(nClasses, activation='softmax'))

    print('Model created')

    fashion_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    print('Model compiled')

    fashion_train = fashion_model.fit(
        train_X,
        train_label,
        batch_size=1024,
        epochs=1,
        verbose=1,
        validation_data=(valid_X, valid_label)
    )

    print('Model fitted')

    test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)

    accuracy = fashion_train.history['accuracy']
    val_accuracy = fashion_train.history['val_accuracy']
    loss = fashion_train.history['loss']
    val_loss = fashion_train.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, label='Training accuracy')
    plt.plot(epochs, val_accuracy, label='Validation accuracy')
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation accuracy and loss')
    plt.legend()
    plt.savefig('epoch.png')

    pred_Y = fashion_model.predict(test_X)
    pred_Cl = np.argmax(pred_Y, axis=1)
    cm = confusion_matrix(test_Y, pred_Cl)
    print(cm)

    plt.clf()

    cmi = np.array([[0] * nClasses] * nClasses)
    cmq = np.array([[0.0] * nClasses] * nClasses)
    for i in range(len(pred_Y)):
        if cmq[test_Y[i]][pred_Cl[i]] < np.max(pred_Y[i]):
            cmi[test_Y[i]][pred_Cl[i]] = i
            cmq[test_Y[i]][pred_Cl[i]] = np.max(pred_Y[i])

    plt.figure(None, figsize=(100, 100))
    for i in range(nClasses):
        for j in range(nClasses):
            plt.subplot(nClasses, nClasses, i * nClasses + j + 1)
            plt.imshow(test_X[cmi[i][j]].reshape(28, 28), cmap='gray')
            plt.title(f'R: {test_Y[cmi[i][j]]}, P: {pred_Cl[cmi[i][j]]}')
            plt.tight_layout()

    plt.savefig('cmi.png')
