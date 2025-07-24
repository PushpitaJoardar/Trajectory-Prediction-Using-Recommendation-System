'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l1, l2, l1l2
from keras.models import Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Flatten, merge, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import argparse
import GMF, MLP

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m', help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8, help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]', help="MLP layers. First element/2 is embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0, help='Regularization for MF embeddings.')
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]', help="Regularization for each MLP layer.")
    parser.add_argument('--num_neg', type=int, default=4, help='Negatives per positive.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam', help='Optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1, help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1, help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='', help='Pretrain MF model file.')
    parser.add_argument('--mlp_pretrain', nargs='?', default='', help='Pretrain MLP model file.')
    return parser.parse_args()

#################### Initializer ####################
def init_normal(shape, name=None):
    # Ensure shape dimensions are integers
    shape = tuple(int(dim) for dim in shape)
    return initializations.normal(shape, scale=0.01, name=name)

#################### Model Definition ####################
def get_model(num_users, num_items, mf_dim=10, layers=[64,32,16,8], reg_layers=[0,0,0,0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)

    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # MF embeddings
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim,
                                  name='mf_embedding_user', init=init_normal,
                                  W_regularizer=l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim,
                                  name='mf_embedding_item', init=init_normal,
                                  W_regularizer=l2(reg_mf), input_length=1)

    # MLP embeddings: force integer half-dimension
    mlp_emb_dim = layers[0] // 2
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=mlp_emb_dim,
                                   name='mlp_embedding_user', init=init_normal,
                                   W_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=mlp_emb_dim,
                                   name='mlp_embedding_item', init=init_normal,
                                   W_regularizer=l2(reg_layers[0]), input_length=1)

    # Flatten embeddings
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))

    # MF part: element-wise product
    mf_vector = merge([mf_user_latent, mf_item_latent], mode='mul')

    # MLP part: concatenation + hidden layers
    mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode='concat')
    for idx in range(1, num_layer):  # Python 3 range
        layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu',
                      name=f'layer{idx}')
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP vectors
    predict_vector = merge([mf_vector, mlp_vector], mode='concat')

    # Final prediction
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(predict_vector)
    # Use Keras 1.x signature: input=, output=
    model = Model(input=[user_input, item_input], output=prediction)
    return model

#################### Pretraining Loader ####################
def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # Load MF embeddings (try both naming conventions)
    try:
        u_w = gmf_model.get_layer('user_embedding').get_weights()
        i_w = gmf_model.get_layer('item_embedding').get_weights()
    except Exception:
        u_w = gmf_model.get_layer('mf_embedding_user').get_weights()
        i_w = gmf_model.get_layer('mf_embedding_item').get_weights()
    model.get_layer('mf_embedding_user').set_weights(u_w)
    model.get_layer('mf_embedding_item').set_weights(i_w)

        # Load MLP embeddings (try both naming conventions)
    try:
        u_w = mlp_model.get_layer('mlp_embedding_user').get_weights()
        i_w = mlp_model.get_layer('mlp_embedding_item').get_weights()
    except Exception:
        u_w = mlp_model.get_layer('user_embedding').get_weights()
        i_w = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(u_w)
    model.get_layer('mlp_embedding_item').set_weights(i_w)

    # Load MLP hidden layers
    for idx in range(1, num_layers):
        w = mlp_model.get_layer(f'layer{idx}').get_weights()
        model.get_layer(f'layer{idx}').set_weights(w)

    # Merge prediction weights
    g_w, g_b = gmf_model.get_layer('prediction').get_weights()
    m_w, m_b = mlp_model.get_layer('prediction').get_weights()
    new_w = np.concatenate([g_w, m_w], axis=0)
    new_b = g_b + m_b
    model.get_layer('prediction').set_weights([0.5 * new_w, 0.5 * new_b])
    return model

#################### Training Data Generator ####################
def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    train_dok = train.todok()  # DOK membership
    num_items = train.shape[1]
    for (u, i) in train_dok.keys():
        user_input.append(u); item_input.append(i); labels.append(1)
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train_dok:
                j = np.random.randint(num_items)
            user_input.append(u); item_input.append(j); labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    num_epochs    = args.epochs
    batch_size    = args.batch_size
    mf_dim        = args.num_factors
    layers        = eval(args.layers) if isinstance(args.layers, str) else args.layers
    reg_layers    = eval(args.reg_layers) if isinstance(args.reg_layers, str) else args.reg_layers
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner       = args.learner
    verbose       = args.verbose
    mf_pretrain   = args.mf_pretrain
    mlp_pretrain  = args.mlp_pretrain

    # Load data
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape

    # Build and compile
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, args.reg_mf)
    optimizer = {'adagrad': Adagrad, 'rmsprop': RMSprop, 'adam': Adam, 'sgd': SGD}[learner](lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    # Pretrain if provided
    if mf_pretrain and mlp_pretrain:
        gmf_model = GMF.get_model(num_users, num_items, mf_dim)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = MLP.get_model(num_users, num_items, layers, reg_layers)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))

    # Initial eval
    hits, ndcgs = evaluate_model(model, testRatings, testNegatives, 10, 1)
    print(f"Init: HR={np.mean(hits):.4f}, NDCG={np.mean(ndcgs):.4f}")

    # Training loop
    for epoch in range(num_epochs):
        u_in, i_in, labels = get_train_instances(train, num_negatives)
        hist = model.fit([np.array(u_in), np.array(i_in)], np.array(labels),
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        if epoch % verbose == 0:
            hits, ndcgs = evaluate_model(model, testRatings, testNegatives, 10, 1)
            print(f"Iter {epoch}: HR={np.mean(hits):.4f}, NDCG={np.mean(ndcgs):.4f}, loss={hist.history['loss'][0]:.4f}")

    if args.out:
        model.save_weights(f"Pretrain/{args.dataset}_NeuMF.h5")
