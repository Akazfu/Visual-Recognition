def knn(x_train, y_train, x_test, n_classes, device):
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 1000 x 784 testing images
    n_classes: no. of classes in the classification task
    device: pytorch device on which to run the code
    return: predicted y_test which is a 1000-sized vector
    """
    #Convert data from numpy arrays to pytorch tensors
    tensor_x_train = torch.from_numpy(x_train).to(device, dtype=torch.float)
    tensor_y_train = torch.from_numpy(y_train).to(device, dtype=torch.float)
    tensor_x_test = torch.from_numpy(x_test).to(device, dtype=torch.float)
    tensor_y_test = torch.zeros((1000), device = device)
    
    #Choose a distance function and a value for k
    #Choose Euclidean using torch.norm and k == 5
    k = 5
    
    #For each test image:
    for i in range (0, tensor_y_test.size()[0]):
      #Calculate the distance between the test image and all training images.
      distance = torch.norm(tensor_x_train - tensor_x_test[i], dim=1)
      
      #Find indices of k training images with the smallest distances
      ind_nn = torch.topk(distance, k, largest = False).indices
      
      #Get classes of the corresponding training images
      classes = torch.gather(tensor_y_train, 0, ind_nn).to(torch.int64)
      
      #Find the most frequent class among these k classes
      #Represent classes as one-hot vectors and stack into a 𝑘 × 10 array
      reshape = torch.reshape(classes, (k,1))
      one_hot = torch.zeros((k, 10), dtype=torch.int64, device = device)
      one_hot.scatter_(1, reshape, 1)
      
      #Compute column-wise sum of this array
      sums = torch.sum(one_hot, 0)
    
      #Take the column with the maximum sum
      maxsum = torch.argmax(sums).float()
      tensor_y_test[i] = maxsum

    #Return the predicted class tensor_y_test 
    return tensor_y_test.cpu().numpy()