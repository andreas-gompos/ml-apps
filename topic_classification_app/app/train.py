
# coding: utf-8

# In[1]:


from keras.utils.np_utils import to_categorical
import os

from dependencies import *

np.random.seed(7)


# In[2]:


def load_dataset():
    df = pd.DataFrame()
    summ = 0
    classes = os.listdir("../data/")

    print("Loading:")
    for class_ in classes:
        current_class_directory = "../data/{}/".format(class_)
        print(current_class_directory)

        for name in sorted(os.listdir(current_class_directory)):
            path = os.path.join(current_class_directory, name)

            current_text = open(path)
            summ += 1
            df.loc[summ,"text"] = current_text.read()
            df.loc[summ,"class"] = class_

                      
    df["class_meaning"] = df["class"]      
    df["class"].replace({"business":0,                             "entertainment":1,                             "politics":2,                             "sport":3,                             "tech":4},inplace=True)
    return df


data = load_dataset()
data = data.sample(frac=1,random_state=25)


# In[ ]:


#build preprocessing pipeline

preprocessing_pipeline = Pipeline([
    ('tokenizer', DocTokenizer()),
    ('encoder', WordsEncoder()),
    ('padder', Padder())])

X_train = preprocessing_pipeline.fit_transform(data.text)
y_train = to_categorical(data["class"])


# In[ ]:


# build keras model

glove_embeddings = {}
f = open("../glove.6B.300d.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embeddings[word] = coefs
f.close()

word_index = preprocessing_pipeline.named_steps['encoder'].encoder_.word_index
embedding_dim = len(glove_embeddings['yes']) #=300
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = glove_embeddings.get(word)
    
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



model = Sequential()
model.add(Embedding(np.shape(embedding_matrix)[0],                    np.shape(embedding_matrix)[1],                    weights=[embedding_matrix],                    trainable=True))
model.add(LSTM(100))
model.add(Dense(70, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(BatchNormalization())
model.add(Dense(30, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(BatchNormalization())
model.add(Dense(5, activation='softmax'))


model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train, epochs=20, batch_size=128, shuffle=True)


# In[ ]:


model.save("./artifacts/model")
joblib.dump(preprocessing_pipeline, './artifacts/preprocessing_pipeline.pkl') 

