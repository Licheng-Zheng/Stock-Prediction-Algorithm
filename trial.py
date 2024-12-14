import numpy as np

thing1 = [3, 1, 3, 4, 8]
thing2 = [[3, 1, 0, 0, 0], 
          [3, 1, 0, 0, 0], 
          [1, 1, 0, 0, 0],
          [1, 1, 0, 0, 0],
          [1, 1, 0, 0, 0],         
        ]

thing3 = np.dot(thing1, thing2)
print(thing3)

thing3 = np.matmul(thing1, thing2)
print(thing3)
