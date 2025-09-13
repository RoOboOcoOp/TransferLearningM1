Build:

### Library        Version ###

pythonInterpreter    3.10.0;
pip                  23.2.1;
matplotlib           3.10.0;
numpy                2.0.2;
scikit-learn         1.7.1;
tensorflow           2.19.0;
keras                3.10.0;


I created a CNN for training using the Cats&Dogs dataset, which contains 25,000 images. Training was performed in the "CNNtraining" project.
When using the dataset, I noticed two corrupted images: one in the Cats folder and the other in the Dogs folder.
Using the code, it was impossible to train because the code was outdated. This outdated code causes
the code to consume excessive RAM.
A change was necessary: ​​instead of placing all the images in a single storage vector, we now process them as batches.
The result of the first training is shown in the image "history_plot.png"


### Transfer learning ###
After training the CNN, we obtain a file of the trained model, "best_model.h5." We then use this model with some modifications to retrain it on a dataset with 3,000 images, but with one more class than in the first training. We now have cats,
dogs, and snakes.
The result is shown in the image "training_history.png."

