import numpy as np
import pdb

print("Importing data")
train_import = np.load('ecs171train.npy')

num_rows = train_import.shape[0] # 50001
num_cols = len(str(train_import[0]).split(",")) # 771


# transforming data into a 2D list
print("transforming data into a 2D list")
train_list = [str(train_import[i+1]).split(",") for i in range(num_rows - 1)]
  
# pdb.set_trace()
  
  
# replacing the NA's with the column mean
print("replacing the NA's with the column mean")
for j in range(num_cols-1): # the last column is the loss
  sum = 0
  num = 0
  
  for i in range(num_rows - 1):
    if train_list[i][j] != "NA":
      sum += float(train_list[i][j])
      num += 1
      
  mean = sum / num
  
  for i in range(num_rows - 1):
    if train_list[i][j] == "NA":
      train_list[i][j] = str(mean)

# pdb.set_trace()
      
      
# changing everything to floats
print("changing the 2D list elements to floats")
train_floats = np.array(train_list).astype(float)


# removing low variance features
print("removing low variance features")
var = np.var(train_floats, axis=0)
idxs = np.tile(np.array(var > .000001, dtype=bool), (num_rows-1, 1))
train_floats = np.reshape(train_floats[idxs], (num_rows-1, -1))
pdb.set_trace()


# normalize data to between 0 and 1
print("standardizing data")
min = np.amin(train_floats, axis=0)
max = np.amax(train_floats, axis=0)
train_normalized = (train_floats - min) / (max - min)
train_export = train_normalized

# exporting the data
print("exporting the data")
np.save('train_data_sanitized.npy', train_export)
