import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from keras.utils import pad_sequences
#  MODEL 
from keras.models import load_model
prediction_model = load_model("Model_weights/model_19.h5")
# PRE REQUISITES
import pickle
with open("Encoded_test.pkl","rb") as f:
    encoding_test = pickle.load(f)

with open('idx_to_word.pkl', 'rb') as f:
    idx_to_word = pickle.load(f)

with open('word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)
# PREDICTION
max = 30
def predict_caption(photo):
    
    input_text = "<s>"
    for w in range(max):
        seq = [word_to_idx[i] for i in input_text.split() if i in word_to_idx]
        seq = pad_sequences([seq],maxlen=max,padding="post")

        ypred = prediction_model.predict([photo,seq])
        ypred = ypred.argmax() #word with max probability, this is called greedy sampling
        word = idx_to_word[ypred]

        input_text += " "+word
        if word == "<e>":
            break
     
    final_caption = input_text.split()[1:-1]
    return " ".join(final_caption)

img_path = "D:\Python\Machine Learning\Image Captioning\Images/"
for i in range(10):
    no = np.random.randint(0,1000)
    img_names = list(encoding_test.keys())
    img = img_names[no]
    photo_2048 = encoding_test[img].reshape((1,2048))

    caption = predict_caption(photo_2048)

    i = plt.imread(img_path+img+".jpg")
    print(caption)
    plt.imshow(i)
    plt.axis("Off")
    plt.show()