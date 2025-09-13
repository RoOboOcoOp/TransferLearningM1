Build:

Library\t         Version\n
pip\t              23.2.1\n
matplotlib\t       3.10.0\n
numpy\t            2.0.2\n
scikit-learn\t     1.7.1\n
tensorflow\t       2.19.0\n
keras\t            3.10.0\n


Criei uma CNN para realizar o treinamento utilizando o dataset Cats&Dogs que contém 25000 imagens. Treinamento realizado no projeto "CNNtraining".
Ao utilizar o dataset percebi que haviam duas imagens corrompidas, juma na pasta de Cats e outra na pasta de Dogs.
Utilizando o código exemplo não foi possível realizar o treinamento devido ao código estar desatualizado, essa desatualização fazia com
que o código consumisse memória RAM em demasia.
Foi necessário uma alteração para ao invés de colocar todas as imagens em um vetor de armazenamento agora processamos as imagens em batch.
O resultado do primeiro treinamento é mostrado na imagem "history_plot.png"


### Transfer learning ###
Após treinada a CNN obtemos um arquivo do modelo treinado "best_model.h5", depois utilizamos esse modelo com algumas alterações para
ser retreinado agora em um dataset com 3000 imagens, mas com uma classe a mais do que no primeiro treinamento. Agora temos cats,
dogs e snakes.
O resultado é mostrado na imagem "training_history.png"

