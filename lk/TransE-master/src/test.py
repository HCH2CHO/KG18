import multiprocessing as mp
import time
import datetime
import numpy as np
# a=np.array([(1,1,1),(2,2,2),(3,3,3)])
# #
# #
# # b=np.array([(4,4,4),(5,5,5),(6,6,6),(7,7,7)])
# # b=b[:,:,np.newaxis]
# # print(b)
# # print(a.shape)
# # print(b.shape)
# #
# # print(a+b)
# # print((a+b).shape)


## dd/mm/yyyy格式
print (time.strftime("%Y/%m/%d_%H:%M:%S"))
ll=datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
print(ll)



