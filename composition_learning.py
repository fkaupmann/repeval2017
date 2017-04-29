from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input
from keras.models import Model

from gensim.models import Word2Vec

import numpy as np
import theano.tensor as T
from theano import function

import sys
import file_util

def init_3d_identity(shape, dtype=None):
    arr  = []
    #iterate over the z-axis
    for i in range(0, shape[2]):
        arr.append(np.identity(shape[0]))
    initial_weights = np.array(arr)
    return K.variable(initial_weights)


#Used for the neural architecture, learning algorithm and preprocessing of data.

class MagicOperation(Layer):
    """Defines a layer that is used to train weights for attribute selection"""

    def __init__(self, output_dim, composition_mode = 'tensor_mult', **kwargs):
        """
        Initialization of the layer.
        :param output_dim: the number of output neurons (for BA: same as input dimension).
        :param composition_mode: the specific composition mode that is to be used for training.
        :param kwargs:
        :return: null
        """
        self.output_dim = output_dim
        self.composition_mode = composition_mode
        super(MagicOperation, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Initializes the weights and adds them to the trainable weights.
        The initialization depends on the specific composition_mode used.
        :param input_shape:
        :return:
        """
        
        input_dim = input_shape[2]
        initial_weight_value = 0
        
        if self.composition_mode == 'tensor_mult_identity':
            self.kernel = self.add_weight(shape=(input_dim,input_dim,input_dim),
                    initializer=init_3d_identity,
                    trainable=True)
        elif self.composition_mode == 'weighted_adj_and_noun_add_identity':
            self.kernel = self.add_weight(shape=(input_dim,input_dim),
                    initializer='identity',
                    trainable=True)

####These have been here before, do not uncomment without fitting to the new keras version
#        if self.composition_mode == 'tensor_mult_identity':
#            arr = []
#            for i in range(0,input_dim):
#                arr.append(np.identity(input_dim))
#            initial_weight_value = np.array(arr)
#        elif self.composition_mode == 'weighted_adj_add_identity':
#            initial_weight_value = np.identity(input_dim)
#        elif self.composition_mode == 'weighted_noun_add_identity':
#            initial_weight_value = np.identity(input_dim)
#        elif self.composition_mode == 'weighted_adj_and_noun_add_identity':
#            initial_weight_value = np.asarray([np.identity(input_dim).tolist(),
#                                               np.identity(input_dim).tolist()])
#        elif self.composition_mode == 'weighted_transitive_add_identity':
#             initial_weight_value = np.asarray([np.identity(input_dim).tolist(),
#                                               np.identity(input_dim).tolist(),
#                                               np.identity(input_dim).tolist()])
        

            
#        self.W = K.variable(initial_weight_value)
#        self.trainable_weights = [self.W]
        self.built = True

    def call(self, x, mask=None):
        """
        Defines the specific compositional function used for attribute selection.
        The data input x is divided into an adjective and a noun input.
        :param x: Input consisting of all adjective and noun inputs for all possible training examples
        :param mask:
        :return: attribute vector.
        """

        if not 'transitive' in self.composition_mode:
            adj = x[:,0,:] 
            noun = x[:,1,:]
        elif 'transitive' in self.composition_mode:
            pred = x[:,0,:]
            subj = x[:,1,:]
            obj = x[:,2,:]
        
        target = []

        if 'tensor_mult' in self.composition_mode:
            # print("Call mit tensor_mult!")    #todo normalisieren?
            adj_matrix = T.tensordot(adj,self.kernel,[[1],[2]])
            attribute = T.tensordot(noun,adj_matrix, [[1],[1]])     #anmerkung für später: hier einfach auch noch über 0-te achse summieren, macht das arbeiten mit dem modell nachher leichter
        elif 'weighted_adj_add' in self.composition_mode:
            weighted_adj = T.dot(adj, self.kernel)
            target = weighted_adj + noun
        elif 'weighted_noun_add' in self.composition_mode:
            weighted_noun = T.dot(noun, self.kernel)
            target = adj + weighted_noun
        elif 'weighted_adj_and_noun_add' in self.composition_mode:
            weighted_adj = T.dot(adj, self.kernel[0])
            weighted_noun = T.dot(noun, self.kernel[1])
            target = weighted_adj + weighted_noun
        elif 'weighted_transitive_add' in self.composition_mode:
            weighted_pred = T.dot(pred, self.kernel[0])
            weighted_subj = T.dot(subj, self.kernel[1])
            weighted_obj = T.dot(obj, self.kernel[2])
            target = weighted_pred + weighted_subj + weighted_obj
        return target

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def construct_data_and_labels(input_data, vector_space, targets_for_training, verbosity = 2):
    """
    Constructs data matrix for training.
    :param input_data: a list of inputs as strings.  
    :param vector_space: pre-trained word embeddings.
    :param verbosity:
    :return:
    """
    #falls attr_train_set nicht leer, werden nur vektoren aus attribute subset genutzt
    if verbosity >= 2:
        print("Trainset in construct data: {}".format(attr_train_set))
        print("aan_list in construct data hat Länge {}".format(len(aan_list)))
    tmp_data = []
    tmp_labels = []
    not_in_embedding_space = []

    #####BAUSTELLE#####

    for sample in input_data:
        target = sample[0]
        inputs = sample[1:]

        if target in targets_for_training or target.upper() in targets_for_training:
            #nur falls das attribut in der konkreten Trainingsmenge enthalten ist
            if verbosity >= 2:
                print("{} ist in Trainings-set".format(attr.upper()))
            input_embedding = []
            target_embedding = []

            x = False
            y = False

            try:
                for word in inputs:
                    input_embedding.append(vector_space[word])
                # adjective_noun = [vector_space[adj],vector_space[noun]]
                # adjective_noun = vector_space[adj] #nur zu testzwecken
                x = True
            except KeyError:
                if verbosity >=2:
                    print("Adjektiv oder Nomen beim Training nicht in WordEmbeddings enthalten: %s,%s" % (adj,noun))
                    
                not_in_embedding_space += inputs


            try:
                target_embedding = vector_space[target.lower()]
                y = True
            except KeyError:
                if verbosity>=2:
                    print("Attribut beim Training nicht in WordEmbeddings enthalten: %s" % (attr))
                if target not in not_in_embedding_space:
                    not_in_embedding_space += [target]



            if x and y:
                tmp_data.append(input_embedding)
                tmp_labels.append(target_embedding)

    tmp_data = np.asarray(tmp_data)
    tmp_labels = np.asarray(tmp_labels)

    if verbosity >= 1 and np.shape(tmp_data) == (0,) or np.shape(tmp_labels) == (0,):
        print("Data oder Labels wurden nicht korrekt erstellt: shape == (0,)")


    return tmp_data,tmp_labels,not_in_embedding_space



def train_model(data_generator, samples_per_epoch, composition_mode, verbosity=2):
    """
    Trains a model that uses certain data, labels and a compositional function.
    :param data: input data
    :param labels: input labels
    :param composition_mode:
    :param verbosity:
    :return:
    """
    if verbosity >= 1:
        print("Trainiere NN mit mode %s..." % composition_mode)

    adj_noun_input = Input(shape=(2,300)) #2 vektoren, je 300 dimensionen

    output = MagicOperation(300, composition_mode = composition_mode)(adj_noun_input)

    model = Model(inputs = adj_noun_input, outputs = output)
    model.compile(optimizer='adam', loss='cosine_proximity', metrics=['accuracy'])

    model.fit_generator(data_generator, samples_per_epoch=samples_per_epoch, verbose=verbosity, nb_epoch=10)

    return model

#TODO: better distinction between function for training and creating the model

def create_model(composition_mode, verbosity=2):
    """
    Create model architecture with a certain compositional function.
    :param composition_mode:
    :param verbosity:
    :return:
    """
    if verbosity >= 1:
        print("Erstelle NN mit mode %s..." % composition_mode)

    adj_noun_input = Input(shape=(2,300)) #2 vektoren, je 300 dimensionen

    output = MagicOperation(300, composition_mode = composition_mode)(adj_noun_input)

    model = Model(inputs = adj_noun_input, outputs = output)
    # model.compile(optimizer='adam', loss='cosine_proximity', metrics=['accuracy'])

    return model

def min_mean_max_weight_matrix(weights):
    """Computes mean min and max for weight matrices."""
    W = weights[0]

    W_adj = W[0].tolist()
    W_noun = W[1].tolist()

    result = [(np.min(W_adj),np.mean(W_adj),np.max(W_adj)), (np.min(W_noun), np.mean(W_noun), np.max(W_noun))]

    return result
