# Build:

### Library - - Version ###

pythonInterpreter - - 3.10.0<br/>
pip - - 23.2.1<br/>
matplotlib - - 3.10.0<br/>
numpy - - 2.0.2<br/>
scikit-learn - - 1.7.1<br/>
tensorflow - - 2.19.0<br/>
keras - - 3.10.0<br/>


I created a CNN inspired on the VGG16 model for training using the Cats&Dogs dataset, which contains 25,000 images. Training was performed in the "CNNtraining" project
and the pre-trainned model is this:

````
    keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(64, (3, 3), padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(64, (3, 3), padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(128),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(num_classes),
    keras.layers.Activation('softmax')
````

When using the dataset, I noticed two corrupted images: one in the Cats folder and the other in the Dogs folder.
Using the given code, it was impossible to train because the code was outdated. This outdated code causes
the code to consume excessive RAM.
A change was necessary: ​​instead of placing all the images in a single storage vector, we now process them as batches.
The result of the first training is shown in the image below:
![Final result of CNN trainning](/history_plot.png)

### Transfer learning ###
After training the CNN, we obtain a file of the trained model, "best_model.h5." We then use this model with some modifications to retrain it on a dataset with 3,000 images, but with one more class than in the first training. We now have cats,
dogs, and snakes.
The result is shown in the image below:
![Final result of Transfer Learning](/training_history.png)

>[!Important]
>In both training sessions, we have a "time" function that takes the start time of the training session and, at the end, returns the total time used for each training session.
>In the initial CNN training, the total time exceeds 60 minutes. In Transfer Learning, where we have a smaller number of images, the total time is less than 10 minutes.
