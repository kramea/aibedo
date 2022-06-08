import os, random, time, shutil
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

#refered: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append([dataset[i + look_back]])
    return np.array(dataX), np.array(dataY)


def moving_average(x, w):
    x_out = []
    _ , n = np.shape(x)
    for i in range(n):
        x_out.append(np.convolve(x[:,i], np.ones(w), 'valid') / w)
    return np.asarray(x_out)


def load_data(file_path, input_time_length, name_of_variable, average_window):
    ds = xr.open_dataset(file_path)
    data = np.asarray(ds[name_of_variable])  #size = [timesteps: vertexs]
    print(np.shape(data))
    data = moving_average(data, average_window)
    print(np.shape(data))
    input_file, output_file  = create_dataset(data, 3)

    return input_file, output_file



def shuffle_data(d1, d2):
    n=len(d1)
    m=len(d2)
    assert(n == m)
    idx = [ i for i in range(m)]
    random.shuffle(idx)
    d1_out = []
    d2_out = []
    for i in idx:
        d1_out.append(d1[i:i+1])
        d2_out.append(d2[i:i+1])
    d1_out2 = np.concatenate(d1_out, axis=0)
    d2_out2 = np.concatenate(d2_out, axis=0)
    return d1_out2, d2_out2



class LSTM(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Linear(input_size, embedding_dim) # use nn.Linear insead of Embedding as inputs are not words
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        embeds = self.relu(embeds)
        
        lstm_out, hidden = self.lstm(embeds, hidden)
        d1,d2,_ = lstm_out.size()
        lstm_out = lstm_out.contiguous().view(d1*d2, -1)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        _, output_dim  = out.size()
        out = out.view(batch_size, -1, output_dim)
        out = out[:,-1:,:] # only last timestep
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        #hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
        #              weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


def main():
    #(1) input parameters : Soo(Todo): consider to  make it as parser_arg
    file_path = "./data/annual_avg_output.nc"
    input_file_length = 3
    name_of_variable = 'psl_pre' # choose among ['tas', 'psl', 'pr']
    output_path = "./lstm_"+str(name_of_variable)+"_ts_"+str(input_file_length)+"/"
    n_layers = 7
    hidden_dim = 256
    embedding_dim = 164
    learning_rate = 0.001
    n_epochs = 100
    batch_size = 70
    clip = 5
    average_window=20
    #(2) load data: input_file (N, time_length, input_dims), output_file (N, 1, output_dims)
    input_file, output_file = load_data(file_path, input_file_length, name_of_variable, average_window)
    print(np.shape(input_file), np.shape(output_file))
    _, _, input_size = np.shape(input_file)
    _, _, output_size = np.shape(output_file)
    #Split the first 80% of the timesteps into training data, and the rest into test data (in chronological order).
    n = len(input_file)
    input_file_tr,input_file_te = input_file[:int(0.8*n)], input_file[int(0.8*n):]
    output_file_tr,output_file_te = output_file[:int(0.8*n)], output_file[int(0.8*n):]
    #shuffle #Soo(Todo): ask to Kalai. Do we shuffle before we divide train/test or after(the way it is implemented here)?
    dataset_tr, dataset_out_tr  = shuffle_data(input_file_tr, output_file_tr)
    dataset_te, dataset_out_te  = shuffle_data(input_file_te, output_file_te)
    print(name_of_variable)
    print("(1) Train-set \n input size:",  np.shape(dataset_tr), "output size:" ,np.shape(dataset_out_tr))
    print("(2) Test-set \n input size:",  np.shape(dataset_te), "output size:" ,np.shape(dataset_out_te))

    #(3) Define model
    if not os.path.isdir(output_path): 
        os.mkdir(output_path)    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_data, time_steps, input_dims = np.shape(input_file)
    model = LSTM(input_size, output_size, embedding_dim,   hidden_dim, n_layers, drop_prob=0.2)
    lr = learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(model)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #(4) Train Model 
    model.train()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(0, n_epochs+1):
        h = model.init_hidden(batch_size)
        train_loss = 0.0
        for i in range(int(len(dataset_tr)/batch_size)+1):
            if i == int(len(dataset_tr)/batch_size):
                images = torch.tensor(dataset_tr[-batch_size:])
                gt_outputs = torch.tensor(dataset_out_tr[-batch_size:])
            else:
                images = torch.tensor(dataset_tr[i*batch_size: (i+1)*batch_size])
                gt_outputs = torch.tensor(dataset_out_tr[i*batch_size: (i+1)*batch_size])
            optimizer.zero_grad()
            h = tuple([e.data for e in h])
            outputs, h = model(images.float(), h)
            loss = criterion(outputs.float(), gt_outputs.float())
            loss.backward(retain_graph=True)
            #nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            print("Minibatch step "+str(i)+" train loss "+str(loss.item()))
            train_loss += loss.item()*images.size(0)
        train_loss = train_loss/len(dataset_tr)
        # validation: As we don't have much data, use testset as validation set
        with torch.no_grad():
            images = torch.tensor(dataset_te)
            test_size, ch, n =images.size()
            gt_outputs = torch.tensor(dataset_out_te)
            val_h = model.init_hidden(test_size)
            val_h = tuple([each.data for each in val_h])
            outputs, val_h = model(images.float(), val_h)
            validation_loss = criterion(outputs.float(), gt_outputs.float())
        print('Epoch: {} \tTraining Loss: {:.6f}\tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            validation_loss
            ))
        if epoch%50==0 and epoch>0:
            #save model
            torch.save(model.state_dict(), output_path+"/lstm_"+str(epoch)+".pt")
            #test with testset
            with torch.no_grad():
                images = torch.tensor(dataset_te)
                gt_outputs = torch.tensor(dataset_out_te)
                test_size, ch, n =images.size()
                val_h = model.init_hidden(test_size)
                val_h = tuple([each.data for each in val_h])
                outputs, val_h = model(images.float(), val_h)
                test_loss = criterion(outputs.float(), gt_outputs.float())
                prediction = outputs.detach().numpy()
                groundtruth = gt_outputs.detach().numpy()
            np.save(output_path+"/prediction_"+str(epoch)+"_"+str(test_loss)+".npy", prediction)
            np.save(output_path+"/groundtruth_"+str(epoch)+"_"+str(test_loss)+".npy", groundtruth)
        

if __name__ == "__main__":
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()

