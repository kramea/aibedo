## Temporal Models:

#### Requirements:

Refer to `../environment_*.yml` (one level up) to check the necessary pre-req. Use it to create a conda envirnment for this project. 


#### Contents:
* `neuralhydrology_examples` contains explicit example for MTS-LSTM available as part of NeuralHydrology. This requires the provided data to train and test the model
* `standalone` contains refactored code to create Aibedo relavant MTS-LSTM models and also code to play around with the initial library
* `rainfall_data` contains preprocessed and extracted data (train, validate, test) for the rainfall dataset used in the original paper. This are in saved as numpy arrays
* `aibedo` the core classes for model, loss and utility function to implement MTS-LSTM model for AiBEDO
* `experiment_mtslstm.ipynb` notebook to test the new aibedo specific classes
* `aibedo_v1.yml` example config file with specification for am MTSLSTM models and other parameters for the simple rainfall data experiment (i.e. `experiment_mtslstm.ipynb`)


