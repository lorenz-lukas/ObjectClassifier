# ***Método 1:***

(Pasta Deep_Learning)

1. Ambiente de desenvolimento

 Compreende a Triplet Network, desenvolvida em python 2.7, Tensorflow 1.8 e usando a KNN Cuda (https://github.com/chrischoy/knn_cuda) para classificação dos embeddings. 
 
2. Como executar
 
A execução do programa deve contar com as pastas Splits povoadas com os .pkl e TFRecords_files com os .tfrecords, sendo que esses últimos não foram enviados devido ao tamanho. Para gerá-los é necessário rodar generate_splits.py, generate_tfrecords.py e generate_tfrecords_test.py. Em seguida, o script principal é rodado com

./run_models_triplet.sh

E os resultados são gravados em Recently_evaluated.

3. Organização do diretório:

 - Auxiliary: pasta com código auxiliar
- Generate_embeddings: código que recebe as imagens e a rede treinada e gera os embeddings
- KNN: código que roda o KNN
- Ongoing_Models: pasta que contém o código dos modelos a serem treinados pelo script
- Recently_evaluated: onde os resultados são gravados
- Splits: pasta com os splits de validação
- Test embeddings: código que testa se os embeddings estão sendo criados de forma coerente (necessita debugger)

# ***Método 2:***
 
 Compreende o método Bag of Visual Words (BoW) desenvolvido em python 2.7. O arquivo "bow1.py" e "helpers.py" correspondem a etapa com o conjunto de treino igual a 5. Já os arquivos "bow2.py" e "helpers2.py" para os conjuntos de treino iguais a 10, 15, 20, 25 e 30. 

- Execução do programa: python bow1.py 

- Organização do diretório: Organização padrão das duas bases de dados fornecidas, CalTech 101 (http://www.vision.caltech.edu/Image_Datasets/Caltech101/) e a disponibilizada pelo professor. 

OBS: para se obter os resultados adquiridos, executar o programa como listado, mudando no código fonte os arquivos .txt de base, onde o primeiro número corresponde ao K utilizado e o segundo a porcentagem, em inteiros, correspondente do dataset. 
  EX: dictionary_500_1.txt corresponte a K = 500 para 100% das classes, ou seja 102 classes.

