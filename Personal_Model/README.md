# Personal Model
After have running the training and test process, in this folder it will be saved the current Model. To reuse is, there is a specific function load model in model_utils.py
## Models Parameter
#### Model1
    - DATASET PARAMS
    sample_size_train=7500
    sample_size_test=5023
    sample_size_val=2000

    - TRAINING PARAMS
    batch_size = 16 
    test_batch_size = 16
    
    - OPTIMIZER & LOSS PARAMS
    cost_function = get_cost_function()
    learning_rate = 0.001
    weight_decay = 0.000001
    momentum = 0.9

    epochs = 20

    After training:
    Train - LOSS: 0.02013 ACCURACY: 92.5%% RECALL: 5.8%
    Validation - LOSS: 0.06406 ACCURACY: 65.2%% RECALL: 4.1%
    Test - LOSS: 0.06042 ACCURACY: 67.8%% RECALL: 4.2%
#### Model2
    - DATASET PARAMS
    sample_size_train=7500
    sample_size_test=5023
    sample_size_val=2000

    - TRAINING PARAMS
    batch_size = 16 
    test_batch_size = 8
    
    - OPTIMIZER & LOSS PARAMS
    cost_function = get_cost_function()
    learning_rate = 0.001
    weight_decay = 0.000001
    momentum = 0.9

    epochs = 20

    After training:
    Train - LOSS: 0.02063 ACCURACY: 91.8%% RECALL: 5.7%
    Validation - LOSS: 0.08587 ACCURACY: 75.1%% RECALL: 9.4%
    Test - LOSS: 0.07803 ACCURACY: 78.0%% RECALL: 9.8%