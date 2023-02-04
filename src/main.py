import streamlit as st

import numpy as np
from train import PredictionPrototypicalNet
st.title("Prototypical Network prediction")

st.markdown('''This application was created to allow the user to train his own prototype network and on its basis to tag instruments in sound files.

In the application we have a choice of pre-trained networks on datasets such as:
* TinySOL
* GoodSounds
* MedleySolosDb
* IRMAS
''')
network_trained = False
form_valid = False
num_uploaders = st.slider("Number of classes to predict from", min_value=2, max_value=10)
uploaded_files = []
form = st.form(key="submit-form")


n_shot = form.number_input("Number of samples in support set of each class (recommended 5, should be smaller than the smallest number of samples per class )", step=1)
q_set = form.number_input("Number of samples in query set of each class (should be not greater than the sum of the smallest number of samples per class minus the data taken for the support set)", step=1)
n_epoch = form.number_input("Number of epochs to train the model", step=1, min_value = 1)
first_dataset = 'TinySOL'
second_dataset = 'MedleySolosDb'
third_dataset = 'GoodSounds'
fouyh_dataset = 'IRMAS'
model_type = form.selectbox("Select model trained on:", [first_dataset, second_dataset, third_dataset, fouyh_dataset])

for i in range(num_uploaders):
    label_name = form.text_input(f"Enter instrument name", key=i)
    uploaded_file = form.file_uploader("Upload audio data", type=["mp3","wav"], accept_multiple_files=True, key=f'file_uploader_{i}')
    if uploaded_file:
        uploaded_files.append([label_name, uploaded_file])
    st.empty()
uploaded_predict_set = form.file_uploader("Upload data to predict", type=["mp3","wav"], accept_multiple_files=True, key=f'predict')
classes = [x[0] for x in uploaded_files]
num_of_samples = [len(x[1]) for x in uploaded_files]

# check input data
if not label_name.strip():
    st.error("Instrument name field is empty")
elif n_epoch < 0:
    st.error("Number of epochs should be equal to or greater than zero")
elif not all(i >= 0 for i in num_of_samples):
    st.error("Please put examples of each instruments")
elif not all(i >= n_shot for i in num_of_samples):
    st.error("Not enaught examples")
elif not all(i-n_shot >= q_set for i in num_of_samples):
    st.error("Number of samples in query set of each class should be not greater than the sum of the smallest number of samples per class minus the data taken for the support set. Please upload more data or change number of samples in query set")
elif (len(set(classes)) != len(classes)):
    st.error("Instruments are not unique")
elif len(uploaded_predict_set) == 0:
    st.error("Please put data to predict")
elif n_epoch > 50:
    st.warning("Large number of epochs will cause long waiting time for the result")
else:
    form_valid = True

generate = form.form_submit_button("Learn model and predict")

if generate:
    if form_valid:
        # choose neural network
        st.success("Data is valid, start training the model")
        if model_type =='TinySOL':
            model_t = "tinysol.ckpt"
        elif model_type =='MedleySolosDb':
            model_t = "medley_solo_db.ckpt"
        if model_type =='GoodSounds':
            model_t = "good_sounds.ckpt"
        else:
            model_t = "irmas.ckpt"
        checkpoint_path = f".//models//{model_t}"
        # learn and predict
        predict_network = PredictionPrototypicalNet(checkpoint_path, uploaded_files, uploaded_predict_set, int(num_uploaders), int(n_shot), int(q_set), int(n_epoch))
        predict_network.train()
        predict_network.predict()
    else:
        st.warning("Please fill out the form with the correct data")
