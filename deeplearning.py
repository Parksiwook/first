# [1] -> [3], [2] -> [5], [3] ->[7] 
# 4 -> 9 , 5 ->11………

import numpy as np
import tensorflow as tf

train_data = np.array([1,2,3,4,5], dtype = 'float64')
label_data = np.array([3,5,7,9,11], dtype = 'float64')

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))

model.compile('sgd','mse')

model.fit(train_data,label_data,epochs=100)

print(model.predict([1,3,5,6,7,8,9]))


# 100
# [[ 2.5983438]
#  [ 6.9060893]
#  [11.213835 ]
#  [13.367708 ]
#  [15.521581 ]
#  [17.675453 ]
#  [19.829327 ]]