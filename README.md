# SubwaySurfer CNN (5 Class Classification)

This repository contains code for capturing and training data for Subway Surfers. It contains code for Capturing data from defined window name, cropping it to deisred length, and saving the taken action along with it inside frames/ directory. The provided files's desciprtion are given below:

## DataCapture.py

This file contains code for capturing the data from a defined window with a provided name, and will crop the image from the top to defined pixels to remove unnecessary jargon like the coin count and player score. It contains two different versions of code. One with 1% automatic 'Nothing' capture and one with manual 'Nothing' Capture with a fixed cap of 2000 per action to balance the dataset. It can be changed if you want more.

## Swipe_Keys.cpp

If your Windows doesn't respond to arrow keys natively, you can compile this file to create a pseudo swipe action at the location of the cursor.
g++ swipe_keys.cpp -o swipe_keys.exe -static -luser32

You need to have g++ installed to be able to compile this. Just install Msys2.

## Data_Augmentation.py

This file will basically flip your left and right images and change the label correspondingly for faster data collection. Make sure to have appropriate amount of data for Up and Down. The model may learn left and right swipes only if data is insufficient.

## Train_Models

There are three versions of the model.
    1. The train_model_color.py uses pretrained imagenet weights for training and also finetunes the copied weights after training. (Transfer Learning)
    2. The train_model_grayscale.py also use pretrained imagenet wieghts but doesnt finetune them. It also uses grayscale instead of RGB for training. (Transfer Learning)
    3. The train_modelv2.py uses simple CNN to train the model without any pretrained weights. (No Transfer Learning)

## Server.py

The server.py will run on wsl2 and run your model. It will take data from client.py and send predicted action to client.py

## CLient.py

The client.py will run on windows and send the captured image to server.py and will execute the action given by the server.
