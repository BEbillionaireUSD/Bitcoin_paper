import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab


torch.manual_seed(0)
np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)

input_window = 20 # number of input steps
output_window = 1 # number of prediction steps
batch_size = 64 
lr = 0.0005 # Usually an order of magnitude smaller than LSTM
epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'Trans_model.pth'
log_interval = 100
dropout_rate=0.2

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
          

class TransAm(nn.Module):
    def __init__(self,feature_size=250,num_layers=1,dropout=dropout_rate):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)

def get_data(end=400, path='BTC_1h.csv'):
    time        = np.arange(0, end, 0.1)
    amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))
    
    from sklearn.preprocessing import MinMaxScaler
    
    #loading weather data from a file
    from pandas import read_csv
    series = read_csv(path, header=0, index_col=0, parse_dates=True, squeeze=True)
    
    # looks like normalizing input values curtial for the model
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    #amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)   
    
    sampels = int(len(series)*0.8)
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]

    # convert our train data into a pytorch train tensor
    #train_tensor = torch.FloatTensor(train_data).view(-1)
    train_sequence = create_inout_sequences(train_data,input_window)
    train_sequence = train_sequence[:-output_window] #todo: fix hack? -> din't think this through, looks like the last n sequences are to short, so I just remove them. Hackety Hack.. 

    #test_data = torch.FloatTensor(test_data).view(-1) 
    test_data = create_inout_sequences(test_data,input_window)
    test_data = test_data[:-output_window] #todo: fix hack?
    
    return train_sequence.to(device),test_data.to(device)

def get_batch(source, i,batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target


def train(train_data):
    model.train() # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('Epoch {:2d} | {:3d}/{:3d} batches | {:5.2f} ms | loss {:5.5f} '.format(
                    epoch, batch, len(train_data) // batch_size, 
                    elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

#%config InlineBackend.figure_format='svg'
def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i,1)
            output = eval_model(data)            
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            
    # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']
    plt.rc('font',family='Times New Roman')
    plt.title('Transformer Prediction Result', fontdict={'family' : 'Times New Roman', 'size': 16})
    plt.ylabel('Normalized Price', fontdict={'family' : 'Times New Roman', 'size': 16})
    plt.xlabel('Time(hour)', fontdict={'family' : 'Times New Roman', 'size': 16})

    plt.plot(test_result,color="tomato", label='Prediction')
    plt.plot(truth,color="deepskyblue", label='True Data') 

    #show legend
    plt.legend()
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=16)
    plt.tick_params(axis='both', labelsize=16)

    #plt.grid(True, which='both')
    #plt.axhline(y=0, color='k')
    plt.savefig('graph/transformer-epoch%d.png'%epoch, dpi=750, bbox_inches='tight')
    plt.close()
    
    return total_loss / i


def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    lenth = len(data_source)
    with torch.no_grad():
        for i in range(0, lenth - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0])* criterion(output, targets).cpu().item()
    return total_loss / lenth

if __name__ == '__main__':

    train_data, val_data = get_data(3500)
    model = TransAm().to(device)
    criterion = nn.MSELoss()

    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min', 
                                                        factor=0.2, 
                                                        patience=5, 
                                                        verbose=False, 
                                                        cooldown=0, 
                                                        min_lr=0, 
                                                        eps=1e-08)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    best_model = None
    #best_model = model.load_state_dict(torch.load(model_path))
    
    patience = 0
    val_loss0 = 10000
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data)
        val_loss= evaluate(model, val_data)
        print('Epoch {:2d} | time: {:5.2f}s | valid loss {:5.5f}'.format(epoch, (time.time() - epoch_start_time), val_loss))
        print('-' * 70)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            plot_and_loss(model, val_data,epoch)
            
        if val_loss0-val_loss >= 1e-5:
            patience = 0
        else:
            patience += 1

        if patience == 10:
            print('Stop at Epoch', epoch, 'Learning Rate', lr)
            break

        val_loss0 = val_loss
        scheduler.step(val_loss) 

    torch.save(model.state_dict(), model_path)
    
