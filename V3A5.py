from V3A4_MLP_ForestClassification import *
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

print("\nLoad data and do preprocessing:")
forestdata  = pd.read_csv('../DATA/ForestTypes/ForestTypesData.csv'); # load data as pandas data frame 
classlabels = ['s','h','d','o'];                                      # possible class labels (C=4) 
classidx    = {classlabels[i]:i for i in range(len(classlabels))}     # dict for mapping classlabel to index 
K           = len(classlabels)               # number of classes 
T_txt = forestdata.values[:,0]               # array of class labels of data vectors (class label is first data attribute)
X = forestdata.values[:,1:]                  # array of feature vectors (features are remaining attributes)
T = [classidx[t.strip()] for t in T_txt]     # transform text labels 's','h','d','o' to numeric lables 0,1,2,3
X,T = np.array(X,'float'),np.array(T,'int')  # explicitely convert to numpy array (otherwise there may occur type confusions/AttributeErrors...)
if flagScaleData>0:
    scaler = DataScaler(X)
    X=scaler.scale(X)
N,D=X.shape                                 # size and dimensionality of data set


print("\nDo MLP witch sklearn")
clfMLP = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5,2), random_state=1, max_iter=6000)
clfMLP.fit(np.round(X),np.round(T))
# print(X[0].reshape(1, -1))
# y_pred_1_MLP = clfMLP.predict(X[0].reshape(1, -1))
# y_pred_2_MLP = clfMLP.predict(X[1].reshape(1, -1))

pipeline = Pipeline([('scaler', StandardScaler()), ('mlpc', clfMLP)])
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, T,scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print("Score: ", n_scores)


# print("\nDo MLP witch Keras")
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# import numpy 
# numpy.random.seed(7)

# X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3)

# model = Sequential()

# model.add(Dense(12, input_dim=27, activation='relu'))
# model.add(Dense(27, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# lr_schedule=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01,decay_steps=100000,decay_rate=0.96)
# optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.0, nesterov=False, name='SGD')

# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# model.fit(X_train, T_train, batch_size=54, epochs=100, validation_split=0.2)

# score = model.evaluate(X_test, T_test, verbose=1)

# print("\n%s: %2.f%%" % (model.metrics_names[1], score[1]*100))