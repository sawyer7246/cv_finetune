import numpy as np
import tensorflow as tf

        
 
if __name__ == '__main__':
    

    res_decison_scheme_array = np.array([1.0000000,0.0008619,0.0005491,0.0330000,1.0000000,0.0000000,0.0000000,0.0000000,1.0000000])
    res_decison_scheme_array.shape = 3,3
    res_pred_array = np.fromfile("pred_array_res.bin",dtype=np.float32)
    res_pred_array.shape = 217,3
    
    alex_decison_scheme_array = np.array([1.0000000,0.0940948,0.0000000,0.0679445,1.0000000,0.0040725,0.0000000,0.0000000,1.0000000])
    alex_decison_scheme_array.shape = 3,3
    alex_pred_array = np.fromfile("pred_array_alex.bin",dtype=np.float32)
    alex_pred_array.shape = 217,3
    
    for train_index in range(217):
        alex_pred_sample = alex_pred_array[train_index]
        res_pred_sample = res_pred_array[train_index]
        
    
    print(res_pred_array)
    tf.app.run()
