################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################

from experiment import Experiment
import sys

# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
# or specify the configuration by running `python main.py custom`
if __name__ == "__main__":
    exp_name = 'default'

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]

    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
    exp.run()
    exp.test()
    
    #     #LSTM lr small
#     exp_name = 'LSTM_v1_lr_small'

#     if len(sys.argv) > 1:
#         exp_name = sys.argv[1]

#     print("Running Experiment: ", exp_name)
#     exp = Experiment(exp_name)
#     exp.run()
#     exp.test()
    
    
#      #LSTM lr large
#     exp_name = 'LSTM_v1_lr_large'

#     if len(sys.argv) > 1:
#         exp_name = sys.argv[1]

#     print("Running Experiment: ", exp_name)
#     exp = Experiment(exp_name)
#     exp.run()
#     exp.test()
    
#     exp_name = 'RNN_lr_big'

#     if len(sys.argv) > 1:
#         exp_name = sys.argv[1]

#     print("Running Experiment: ", exp_name)
#     exp = Experiment(exp_name)
#     exp.run()
#     exp.test()
    

#      #RNN lr 
#     exp_name = 'RNN-regular'

#     if len(sys.argv) > 1:
#         exp_name = sys.argv[1]

#     print("Running Experiment: ", exp_name)
#     exp = Experiment(exp_name)
#     exp.run()
#     exp.test()
    
    
#     #RNN lr small
#     exp_name = 'RNN-small'

#     if len(sys.argv) > 1:
#         exp_name = sys.argv[1]

#     print("Running Experiment: ", exp_name)
#     exp = Experiment(exp_name)
#     exp.run()
#     exp.test()
    
    
    
#     #LSTM change
#     exp_name = 'LSTM_v1_2048——1048'
#     print("Running Experiment: ", exp_name)
#     exp = Experiment(exp_name)
#     exp.run()
#     exp.test()
    
    
    #LSTM change again
    exp_name = 'RNN_temperature'
    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
    exp.run()
    exp.test()