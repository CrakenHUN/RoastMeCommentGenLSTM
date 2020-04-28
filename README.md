# r/RoastMe Comment Generator LSTM

This is a neural network used to generate text based on the comments of the subreddit r/RoastMe

## Prequisites

- Python
- Tensorflow
- Numpy
- Keras
- h5py (used to save and load weights)

## Getting started

After cloning the repository, you have the options of training the model again, on my data, (`RoastMe.txt`) or running one of the two models. (`SimpleBot` and `SmarterBot`)

You can change the model in `Roastme_insult_generator.py`. By default, it is set up in the architecture of the `SmarterBot`.

To train the model, run `python Roastme_insult_generator.py -train`. This will run the training, saving the weights and the model every epoch. To load the model, use `python Roastme_insult_generator.py [model.json file path] [weights.h5 file path] [generated length]`.

Default file paths (if you haven't changed the place of anything in the repo):
`python Roastme_insult_generator.py .\SmarterBot\finalModel.json .\SmarterBot\finalWeights.h5 600`
