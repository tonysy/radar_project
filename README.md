# Lidar Gesture Recognization

## Requirements
- Keras
- pydot(pip install pydot==1.1.0)
- Tensorflow

## Dataset
We create a lidar gesture image dataset which included 4 classes(backward, forward, static, rotate), and will be release soon.

## Prepare
- `mkdir data` and put the data in this folder
- `mkdir logs` which is used to store logs file for TensorBoard

## Train
- `python net/train.py`, then you will get network weight in `model` folder, where existed model trained by us already. And the network architecture is stored in `network.json`, which can be read directly when test new data.
- After the train process stop, there will be a png file in `model`, which show the `val_acc` and `val_loss` curve.

## Test
Please `from net.testor import Gesture_Testor` when you want to predict the new data without training process.

We have already pack up the test process into a class, and you can directly use
`label = Gesture_Testor().test(test_data)` to get the label belong to the test_data.

The test_data in `test.py` is just the first sample of the train dataset.

## Then
if you have any problem, feel free to contact with me by issue or email.
