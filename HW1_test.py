import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device
# Help run the DNN faster
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 123
# best practice to use learning rate decay after initial training
learning_rate = 1e-3  # Lower learning rate to prevent NaN
#adjusted number of epochs after adding other overfitting preventative techniques
# num_epochs = 35
num_epochs = 15
batch_size_train = 25
batch_size_test = 300
# batch_size = 50
# batch_size = 100
# batch_size = 150
# batch_size = 200
# batch_size = 256
# batch_size = 300
# batch_size = 330
# batch_size = 512

num_features = 5  # 3 forces + 2 angles
num_classes = 3   # Control efforts (one for each thruster)
# num_hidden1 = 256
# num_hidden2 = 512
# num_hidden3 = 1024
# starts to platue at about zero
num_hidden1 = 512
num_hidden2 = 1024
num_hidden3 = 2048
# continues to platue at about zero
# not needed after using batch normalization
# num_hidden1 = 1024
# num_hidden2 = 2048
# num_hidden3 = 4096

# num_hidden1 = 2048
# num_hidden2 = 4096
# num_hidden3 = 8192

# Parameters
# Define the ranges for the forces and angles
# Generate data artifically in order to optimize real-time calculations
# Ranges are found on pg. 3

F1_range = [-10000, 10000]
F2_range = [-5000, 5000]
alpha2_range = [-180, 180]
F3_range = [-5000, 5000]
alpha3_range = [-180, 180]

# Number of samples to generate
num_samples = 5000

# Normalize the force and angle ranges to avoid overfitting
F1 = np.random.uniform(F1_range[0], F1_range[1], num_samples) / 10000  # Normalize to [-1, 1]
F2 = np.random.uniform(F2_range[0], F2_range[1], num_samples) / 5000
F3 = np.random.uniform(F3_range[0], F3_range[1], num_samples) / 5000
alpha2 = np.random.uniform(alpha2_range[0], alpha2_range[1], num_samples) / 180
alpha3 = np.random.uniform(alpha3_range[0], alpha3_range[1], num_samples) / 180 

# Combine the forces and angles into one matrix (inputs for the neural network)
u = np.vstack([F1, F2, F3, alpha2, alpha3]).T 
mu = np.vstack([F1, F2, F3])

# Transformation Matrix

# Define the neural network
# increasing inputs for output nn lowers loss
class DNNModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DNNModel, self).__init__()
        
        # Encoder
        # Uses thruster inputs to produce force and angle output
        self.encoder = nn.Sequential(
            nn.Flatten(), 
            # 1st hidden layer
            nn.Linear(num_features, num_hidden1, bias=False),  
            # Batch norm helps accelerate the training process and correct overfitting
            # With batch norm weight initialization is less important slide 24
            nn.BatchNorm1d(num_hidden1),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 2nd hidden layer
            nn.Linear(num_hidden1, num_hidden2, bias=False), 
            nn.BatchNorm1d(num_hidden2), 
            nn.ReLU(),
            nn.Dropout(0.5),
            # 3rd hidden layer
            nn.Linear(num_hidden2, num_hidden3), 
            nn.ReLU()
        )

        # Decoder
        # Dropout to allow all nodes to be utilized equally
        # Adding dropout corrected overfitting and smoothed the loss curve
        self.decoder = nn.Sequential(
            # 1st hidden layer
            nn.Flatten(),
            nn.Linear(num_hidden3, num_hidden2, bias=False), 
            nn.BatchNorm1d(num_hidden2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 2nd hidden layer
            nn.Linear(num_hidden2, num_hidden1, bias=False),
            nn.BatchNorm1d(num_hidden1),  
            nn.ReLU(),
            nn.Dropout(0.5),
            # 3rd hidden layer
            nn.Linear(num_hidden1, num_features),  
            nn.ReLU()
        )

        # Output (Control efforts)
        self.output_layer = nn.Sequential(
            nn.Linear(num_hidden3, num_classes)  
        )
    
    # Must define forward method so the DNN knows the order to run input data flow
    # DNN is not a prebuilt model 
    # This function does not have to be called manually
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.output_layer(encoded)
        return output, decoded

# create model and use device
model = DNNModel(num_features, num_classes)
model.to(device)

# Weight regularization to prevent overfitting
# slide 10
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
torch.manual_seed(random_seed)

# Define max thrust (for scaling outputs)
max_thrust = torch.tensor([100, 50, 30], dtype=torch.float32).to(device)


# Convert data to torch tensors
# Must be converted before the data can be used in DNN
X_train = torch.tensor(u, dtype=torch.float32).to(device)
Y_train = torch.tensor(u[:, :3], dtype=torch.float32).to(device)

# Split the dataset into training and testing sets (80/20)
split = int(0.8 * num_samples)
# X_train = X_train[:split]
# X_test = X_train[split:]
# Y_train = Y_train[:split]
# Y_test = Y_train[split:]
X_train, X_test = X_train[:split], X_train[split:]
Y_train, Y_test = Y_train[:split], Y_train[split:]

# Training
train_loss = []

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])
    epoch_loss = 0
    
    for i in range(0, X_train.size()[0], batch_size_train):
        optimizer.zero_grad()
        
        indices = permutation[i:i+batch_size_train]
        batch_X = X_train[indices]
        batch_Y = Y_train[indices]
        
        # Forward pass
        outputs, new_inputs = model(batch_X)
        
        # Define loss function
        L0 = nn.MSELoss()(outputs,batch_Y)
        L1 = nn.MSELoss()(outputs, batch_Y) # Force errors
        L2 = nn.L1Loss()(new_inputs, batch_X) # For input reconstruction (L1 loss)
        L3 = torch.mean((outputs[1:] - outputs[:-1]) ** 2)  # L3: Loss for rate of change in thruster commands
        L4 = torch.sum(outputs ** 2) # Loss to penalize high power consumption
        L5 = nn.MSELoss()(batch_X[:, -2:], torch.zeros_like(batch_X[:, -2:])) #Loss to penalize disallowed azimuth angles
        # Compute losses
        # L1 = L1_loss(outputs, batch_Y)
        # L2 = L2_loss(new_inputs, batch_X)
        
        # Combine losses
        #loss = L1 + 0.1 * L2
        loss = (L0 * 0.1) + (L1 * 0.1) + (L2 * 0.1) + (L3 * 0.1) + (L4 * 0.1) + (L5 * 0.1)
        
        # Backpropagation
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / (len(X_train) // batch_size_train)
    train_loss.append(avg_train_loss)
    print(f'Train Loss {epoch}: {avg_train_loss:.4f}')


# Testing using 20% of generated data
test_loss = []

#with torch.no_grad():

for epoch in range(num_epochs):
    model.eval()
    #permutation = torch.randperm(X_test.size()[0])
    total_loss_test = 0
    loss_test = 0
    batch_count = 0
    print("Weights after training:", model.encoder[1].weight[:2])
    for i in range(0, X_test.size()[0], batch_size_test):
        optimizer.zero_grad()
        
        # indices = permutation[i:i+batch_size]
        # batch_X = X_test[indices]
        # batch_Y = Y_test[indices]
        
        batch_X_test = X_test[i:i+batch_size_test]
        batch_Y_test = Y_test[i:i+batch_size_test]
        
        # Forward pass
        outputs_test, new_inputs = model(batch_X_test)
        
        # Compute losses
        # L1 = L1_loss(outputs, batch_Y)
        # L2 = L2_loss(new_inputs, batch_X)
        # L4 = torch.sum(outputs ** 2)
        # L5 = nn.MSELoss(batch_X[:, -2:], torch.zeros_like(batch_X[:, -2:]))
        # loss_function(outputs, batch_X, batch_Y)
        L0 = nn.MSELoss()(outputs_test,batch_Y_test)
        L1 = nn.MSELoss()(outputs_test, batch_Y_test) # Force errors
        L2 = nn.L1Loss()(new_inputs, batch_X_test) # For input reconstruction (L1 loss)
        L3 = torch.mean((outputs_test[1:] - outputs_test[:-1]) ** 2)  # L3: Loss for rate of change in thruster commands
        L4 = torch.sum(outputs_test ** 2) # Loss to penalize high power consumption
        L5 = nn.MSELoss()(batch_X_test[:, -2:], torch.zeros_like(batch_X_test[:, -2:])) #Loss to penalize disallowed azimuth angles
        
        # Combine losses
        # include scaling factors pg. 5
        loss_test = (L0 * 0.1) + (L1 * 0.1) + (L2 * 0.1) + (L3 * 0.1) + (L4 * 0.1) + (L5 * 0.1)
        
        # Backpropagation
        #loss_test.backward()

        # Clip gradients to prevent gradient explosion
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        # loss_test += loss_test.item()
        # batch_count += 1
        
        # Accumulate total loss for this epoch
        total_loss_test += loss_test.item()
        print(f'Test Loss Item {epoch}: {loss_test:.4f}')
        #print(f'Test Loss Item {epoch}: {loss_test.item():.4f}')
        batch_count += 1
        # epoch_loss += loss.item()
        epoch_loss += loss_test.item()

    #avg_test_loss = epoch_loss / (len(X_test) // batch_size_test)
    avg_test_loss = total_loss_test / batch_count
    test_loss.append(avg_test_loss)
    print(f'Test Loss {epoch}: {avg_test_loss:.4f}')

    # Calculate and store average test loss for this epoch
    #avg_test_loss = total_loss_test / batch_count
    #test_loss.append(avg_test_loss)
    # test_loss.append(total_loss_test)
    # print(f'Test Loss {epoch}: {total_loss_test:.4f}')

# Calculate average test loss
#avg_test_loss = epoch_loss_test / (len(X_test) // batch_size_test)
#avg_test_loss = sum(test_loss) / len(test_loss)

# avg_test_loss = epoch_loss_test / batch_count
# test_loss.append(avg_test_loss)
# print(f'Test Loss {epoch}: {avg_test_loss:.4f}')

#epochs = list(range(1, len(train_loss) + 1))  # Epoch numbers
#test_fit = np.polyfit(epochs, test_loss, 1)  # Fit line to testing loss

# Plot the training loss
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Testing Loss')
#plt.plot(test_fit, label='Testing Loss')
plt.title('Loss vs Epoch #')
plt.xlabel('Epoch')
plt.ylabel('Epoch Loss')
plt.grid()
plt.legend()
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(test_loss, label='Testing Loss', color='orange')
plt.title('Testing Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()