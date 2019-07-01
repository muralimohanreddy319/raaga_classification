import cgi

from flask import Flask, request, session, redirect, url_for, abort, \
     render_template, flash
import os
from flask import Flask, flash, request, redirect, url_for
from sklearn.cross_validation import StratifiedShuffleSplit
from werkzeug.utils import secure_filename
import glob
from scipy.io import wavfile
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from os import listdir
import shutil
from os.path import isfile, join
from timeit import default_timer as timer
from tensorflow import keras

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from six.moves import cPickle as pickle
import pickle

import librosa
import soundfile as sf

from python_speech_features import mfcc
from python_speech_features import logfbank
import soundfile as sf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running thiimport numpy as np
import pandas as pd
from pandas import Series, DataFrame
from tensorflow.python.framework import ops

import tensorflow as tf
import matplotlib.pyplot as plt
import math
import sklearn.metrics as skm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from subprocess import check_output





#******auto encoders****
#create random minibathes from the given data
def random_mini_batches(X, Y, mini_batch_size=1024, seed=0):
    np.random.seed( seed )
    m = X.shape[1]
    mini_batches = []

    permutation = list( np.random.permutation( m ) )
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    num_complete_minibatches = math.floor(
        m / mini_batch_size )  # number of mini batches of size mini_batch_size in your partitionning
    for k in range( 0, num_complete_minibatches ):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: mini_batch_size * (k + 1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k: mini_batch_size * (k + 1)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append( mini_batch )
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches: m]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append( mini_batch )

    return mini_batches


tf.reset_default_graph()


def initialize_parameters(f1=193, f2=100, f3=50):
    tf.set_random_seed( 1 )
    W1 = tf.get_variable( "W1", [f2, f1], initializer=tf.contrib.layers.xavier_initializer( seed=1 ) )
    b1 = tf.get_variable( 'b1', [f2, 1], initializer=tf.zeros_initializer() )
    W2 = tf.get_variable( "W2", [f3, f2], initializer=tf.contrib.layers.xavier_initializer( seed=1 ) )
    b2 = tf.get_variable( 'b2', [f3, 1], initializer=tf.zeros_initializer() )
    W3 = tf.get_variable( 'W3', [f2, f3], initializer=tf.contrib.layers.xavier_initializer( seed=1 ) )
    b3 = tf.get_variable( 'b3', [f2, 1], initializer=tf.zeros_initializer() )
    W4 = tf.get_variable( 'W4', [f1, f2], initializer=tf.contrib.layers.xavier_initializer( seed=1 ) )
    b4 = tf.get_variable( 'b4', [f1, 1], initializer=tf.zeros_initializer() )
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W4": W4,
                  "b4": b4,
                  "W3": W3,
                  "b3": b3}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    Z1 = tf.add( tf.matmul( W1, X ), b1 )  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.tanh( Z1 )
    keep_prob = tf.placeholder( "float" )
    A1_dropout = tf.nn.dropout( A1, keep_prob )

    Z2 = tf.add( tf.matmul( W2, A1 ), b2 )  # Z1 = np.dot(W1, X) + b1
    A2 = tf.nn.relu( Z2 )  # A1 = relu(Z1)                                              # A2 = relu(Z2)
    A2_dropout = tf.nn.dropout( A2, keep_prob )

    Z3 = tf.add( tf.matmul( W3, A2 ), b3 )  # Z3 = np.dot(W3,Z2) + b3
    A3 = tf.nn.tanh( Z3 )
    A3_dropout = tf.nn.dropout( A3, keep_prob )

    Z4 = tf.add( tf.matmul( W4, A3 ), b4 )  # Z3 = np.dot(W3,Z2) + b3
    A4 = tf.nn.relu( Z4 )
    return A4, A2


def compute_cost(A4, Y, parameters):
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    #Cost is the the difference of output with input.
    cost = tf.reduce_mean(tf.pow(Y - A4, 2))
    print(cost)
    return cost

def create_placeholders(n_x, n_y):
    X = tf.placeholder(dtype="float", shape=(n_x, None), name='X')
    Y = tf.placeholder(dtype="float", shape=(n_y, None), name='Y')
    return X, Y

def model(X_train, Y_train, f_2, f_3, learning_rate,
          num_epochs, minibatch_size, print_cost):
    ops.reset_default_graph()
    tf.set_random_seed( 1 )
    seed = 3
    print(X_train)
    print(X_train.shape)
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X, Y = create_placeholders( n_x, n_y )
    parameters = initialize_parameters( f2=f_2, f3=f_3 )
    A4, A2 = forward_propagation( X, parameters )
    cost = compute_cost( A4, Y, parameters )
    print( cost )
    # Tensorflow optimizer
    optimizer = tf.train.GradientDescentOptimizer( 0.5 ).minimize( cost )
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run( init )
        # Do the training loop
        for epoch in range( num_epochs ):
            epoch_cost = 0.
            num_minibatches = int( m / minibatch_size )
            seed = seed + 1
            minibatches = random_mini_batches( X_train, Y_train, minibatch_size, seed )
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run( [optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y} )
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True:
                print( "Cost after epoch %i: %f" % (epoch, epoch_cost) )
            if print_cost == True:
                costs.append( epoch_cost )
        parameters = sess.run( parameters )
        return parameters, A2




def forward_propagationout(X, parameters):
    # retrieve parameters
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    Z1 = np.dot( W1, X ) + b1  # Z1 = np.dot(W1, X) + b1
    A1 = np.tanh( Z1 )
    Z2 = np.dot( W2, A1 ) + b2  # Z1 = np.dot(W1, X) + b1
    A2 = relu( Z2 )  # A2 = relu(Z2)
    Z3 = np.dot( W3, A2 ) + b3  # Z3 = np.dot(W3,Z2) + b3
    A3 = np.tanh( Z3 )
    Z4 = np.dot( W4, A3 ) + b4  # Z3 = np.dot(W3,Z2) + b3
    A4 = relu( Z4 )
    return A2
def relu(x):
    s = np.maximum( 0, x )
    return s



##data
df = pd.read_pickle("violin_data_55_split.p")
le = LabelEncoder()
label_num = le.fit_transform(df["label"])
df["label_id"] = label_num

df["is_Karaharapriya"] = np.where(df.label.str.contains("Karaharapriya"), 1, 0)
df['is_kalyani'] =np.where(df.label.str.contains("kalyani"), 1, 0)
df['is_sankarabharanam'] = np.where(df.label.str.contains("sankarabharanam"), 1, 0)
df['is_kambhoji'] =np.where(df.label.str.contains("kambhoji"), 1, 0)
df['is_thodi'] = np.where(df.label.str.contains("thodi"), 1, 0)


temp = list(df['label'].unique())
temp = list(i.strip() for i in temp)
dict2={}
for i in range(len(temp)):
    dict2[i]=df.loc[df['label_id'] == i, 'label'].iloc[0]
dict2


df = df.sample(frac=1,random_state=1).reset_index(drop=True)
stratSplit = StratifiedShuffleSplit(df['label_id'], 1, test_size=0.30,random_state=32)
for train_idx,test_idx in stratSplit:
    train_data = df.iloc[train_idx]
    x_test = df['features'][test_idx]
    y_test = df['label_id'][test_idx]


dict1={}
for raga in temp:
    print(raga)
    dict1[raga+"_y_train"] = train_data["is_"+raga]
    print(dict1[raga+"_y_train"].values)
dict1

train_data = train_data['features'].values
train_data=np.stack(train_data,axis=0)

test_data = x_test.values
test_data=np.stack(test_data,axis=0)


parameters, A2 = model(train_data.T,train_data.T,100,80,minibatch_size=25,num_epochs=250, learning_rate=0.001, print_cost=True)

test_A2 = forward_propagationout(test_data.T.astype('float32'),parameters)
x_test = test_A2.T

train_A2=forward_propagationout(train_data.T.astype('float32'), parameters)
x_train = train_A2.T


clf_karaharapriya = RandomForestClassifier(n_estimators=100,random_state=0)
clf_karaharapriya.fit(x_train,dict1['Karaharapriya_y_train'])

clf_kalyani = RandomForestClassifier(n_estimators=100,random_state=0)
clf_kalyani.fit(x_train,dict1['kalyani_y_train'])

clf_kambhoji = RandomForestClassifier(n_estimators=100,random_state=0)
clf_kambhoji.fit(x_train,dict1['kambhoji_y_train'])

clf_sankarabharanam = RandomForestClassifier(n_estimators=100,random_state=0)
clf_sankarabharanam.fit(x_train,dict1['sankarabharanam_y_train'])

clf_thodi = RandomForestClassifier(n_estimators=100,random_state=0)
clf_thodi.fit(x_train,dict1['thodi_y_train'])





UPLOAD_FOLDER = "E:/carnatic_audio/untitled/upload_files"
ALLOWED_EXTENSIONS = set(['wav'])
app = Flask( __name__ )
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['DEBUG'] = True
app.config['SECRET_KEY'] ='super-secret-key'
app.config['USERNAME'] = 'admin'
app.config['PASSWORD'] = 'default'







def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs( UPLOAD_FOLDER )
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("hello")
            return redirect(url_for('temp'))
    else:
        return render_template('upload_file.html')


def test(file):
    print(file)
    X, sample_rate = librosa.load( file, mono=True )
    stft = np.abs( librosa.stft( X ) )
    mfccs = np.mean( librosa.feature.mfcc( y=X, sr=sample_rate, n_mfcc=40 ).T, axis=0 )
    chroma = np.mean( librosa.feature.chroma_stft( S=stft, sr=sample_rate ).T, axis=0 )
    mel = np.mean( librosa.feature.melspectrogram( X, sr=sample_rate ).T, axis=0 )
    contrast = np.mean( librosa.feature.spectral_contrast( S=stft, sr=sample_rate ).T, axis=0 )
    tonnetz = np.mean( librosa.feature.tonnetz( y=librosa.effects.harmonic( X ), sr=sample_rate ).T, axis=0 )
    features = np.empty( (0, 193) )
    ext_features = np.hstack( [mfccs, chroma, mel, contrast, tonnetz] )
    features = np.vstack( [features, ext_features] )
    features
    features = features - features.mean( axis=1 )
    features = features / np.abs( features ).max( axis=1 )
    return features

# @app.route('/upload',methods = ["GET","POST"])
# def upload():
#     if request.method == "POST":
#         form = cgi.FieldStorage()
#         print (form)
#         fname = form["audio"].filename
#         print("Got filename:", fname)
#         return "upload"
#     return "not_upload"

@app.route('/temp')
def temp():
    print("temp")
    predicted_raga=""
    files = listdir(UPLOAD_FOLDER)
    path = UPLOAD_FOLDER+"/"+files[0]
    test_features = test(path)
    A2 = forward_propagationout( test_features.T.astype( 'float32' ), parameters )

    l2_rf = clf_kalyani.predict_proba( A2.T )
    l3_rf = clf_kambhoji.predict_proba( A2.T )
    l1_rf = clf_karaharapriya.predict_proba( A2.T )
    l5_rf = clf_thodi.predict_proba( A2.T )
    l4_rf = clf_sankarabharanam.predict_proba( A2.T )
    print(l1_rf)
    temp = [l1_rf[0][1], l2_rf[0][1], l3_rf[0][1], l4_rf[0][1], l5_rf[0][1]]
    max1 = temp.index(max(temp))
    predicted_raga = dict2[max1]
    return 'The given file is composed by '+predicted_raga+' raga'


if __name__ == '__main__':
    app.run(debug = True)
