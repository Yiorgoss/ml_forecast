#!/bin/bash
source /home/ubuntu/anaconda3/bin/activate tensorflow_p36;
for i in {1...100}
do 
    echo $i
    python src/Enc_Dec_LSTM.py
done
