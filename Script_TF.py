#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from operator import itemgetter
from random import shuffle
import torch
import math
from typing import TypeVar, Optional, Iterator
import torch_geometric
from torch_geometric.data import Dataset,Data
import numpy as np
import zipfile
import os
from random import shuffle
import time
import io
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import argparse
from torch_geometric.loader import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import Dataset as Dt
from torch.nn import Linear, LSTM,LogSoftmax,NLLLoss, Embedding, GRU, Sequential, ReLU
from torch_geometric.nn import GraphConv,global_add_pool,NNConv
T_co = TypeVar('T_co', covariant=True)


# In[2]:





# In[ ]:


parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--max_epochs', type=int, default=1000, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='gloo', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')


def main():
    print("Starting...")

    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    """ This next line is the key to getting DistributedDataParallel working on SLURM:
		SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
 		current process inside a node and is also 0 or 1 in this example."""

    local_rank = int(os.environ.get("SLURM_LOCALID")) 
    rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank

    current_device = local_rank
    print(current_device)

    torch.cuda.set_device(current_device)

    """ this block initializes a process group and initiate communications
		between all processes running on all nodes """

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    #init the process group
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
    print("process group ready!")

    print('From Rank: {}, ==> Making model..'.format(rank))
# In[3]:


    class MoleculeDataset(Dataset):
        def __init__(self,root,test=False, transform=None, pre_transform= None, pre_filter= None):
            self.test = test
            self.archive = zipfile.ZipFile(os.path.join(os.getcwd(),'Graph_Data/processed.zip'),'r')
            super(MoleculeDataset,self).__init__(root, transform, pre_transform, pre_filter)
            
        @property
        def raw_file_names(self):
            if self.test:
                return  'Smiles Test Dataset.csv'
            else:
                return 'Smiles Train Dataset.csv'
              
        
        @property
        def processed_file_names(self):
            
            #self.data = pd.read_csv(self.raw_paths[0]).reset_index()
            
            #if self.test:
                #return [f'data_test_{i}.pt' for i in list(self.data.index)]
            #else:
                #return [f'data_{i}.pt' for i in list(self.data.index)]
            return 'not_implemented.pt'
    
        
        def download(self):
            pass
            
            
        def process(self):
            self.data = pd.read_csv(self.raw_paths[0])
                
        
        def len(self):
            return self.data.shape[0]
        
        def get(self,idx):
            
            if self.test:
                data = torch.load(io.BytesIO(self.archive.read(f'processed/data_test_{idx}.pt')))
            else:
                data = torch.load(io.BytesIO(self.archive.read(f'processed/data_{idx}.pt')))
            # DO THE FOLLOWING STEP ONLY FOR PRE_STORED DATA
            #data= MyData(x=data.x, edge_index= data.edge_index, edge_attr= data.edge_attr, smiles= data.smiles, 
                           #in_seq= data.in_seq, out_seq= data.out_seq)
            return data
       
    class DistributedBucketSampler(Sampler[T_co]):
 
 
        def __init__(self, dataset: Dataset,bucket_boundaries,num_replicas: Optional[int] = None,
                     rank: Optional[int] = None, shuffle: bool = True,
                     seed: int = 0, drop_last: bool = False, batch_size=1024) -> None:
            if num_replicas is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                num_replicas = dist.get_world_size()
            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                rank = dist.get_rank()
            if rank >= num_replicas or rank < 0:
                raise ValueError(
                    "Invalid rank {}, rank should be in the interval"
                    " [0, {}]".format(rank, num_replicas - 1))
            self.dataset = dataset
            self.num_replicas = num_replicas
            self.rank = rank
            self.epoch = 0
            self.drop_last = drop_last
            self.bucket_boundaries = bucket_boundaries
            self.batch_size = batch_size
            self.boundaries = list(self.bucket_boundaries)
            self.buckets_min = torch.tensor([np.iinfo(np.int32).min] + self.boundaries)
            self.buckets_max = torch.tensor(self.boundaries + [np.iinfo(np.int32).max])
            self.boundaries = torch.tensor(self.boundaries)
            # If the dataset length is evenly divisible by # of replicas, then there
            # is no need to drop any data, since the dataset will be split equally.
            if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.num_samples = math.ceil(
                    (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
                )
            else:
                self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
            self.total_size = self.num_samples * self.num_replicas
            self.shuffle = shuffle
            self.seed = seed
    
        def __iter__(self) -> Iterator[T_co]:
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            else:
                indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
    
            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[:self.total_size]
            assert len(indices) == self.total_size
    
            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            ind_n_len = []
            for i, p in enumerate(dataset[indices]):
                ind_n_len.append( (indices[i], p.in_seq.shape[0]) )
    
            self.ind_n_len = ind_n_len
            data_buckets = dict()
            # where p is the id number and seq_len is the length of this id number. 
            for p, seq_len in self.ind_n_len:
                pid = self.element_to_bucket_id(p,seq_len)
                if pid in data_buckets.keys():
                    data_buckets[pid].append(p)
                else:
                    data_buckets[pid] = [p]
    
            for k in data_buckets.keys():
    
                data_buckets[k] = torch.tensor(data_buckets[k])
    
            iter_list = []
            for k in data_buckets.keys():
    
                batch = torch.split(data_buckets[k], self.batch_size, dim=0)
    
                iter_list += batch
    
            shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
            # size
            for i in iter_list: 
                yield i.numpy().tolist() # as it was stored in an array
        
        def element_to_bucket_id(self, x, seq_length):
    
            valid_buckets = (seq_length >= self.buckets_min)*(seq_length < self.buckets_max)
            bucket_id = valid_buckets.nonzero()[0].item()
    
            return bucket_id
    
    
        def __len__(self) -> int:
            return self.num_samples
    
        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch    
        
        
        
        
    class Encoder(torch.nn.Module):
        def __init__(self,latent_dim):
            super(Encoder, self).__init__()
            num_features = 5
            dim =32
            nn1 = Sequential(Linear(3,25),ReLU(),Linear(25,num_features*dim))
            self.conv1 =NNConv(num_features, dim, nn1, aggr= 'mean')
            nn2 = Sequential(Linear(3,25),ReLU(),Linear(25,dim*dim))
            self.conv2 =NNConv(dim, dim, nn2, aggr= 'mean')
            nn3 = Sequential(Linear(3,25),ReLU(),Linear(25,dim*dim))
            self.conv3 =NNConv(dim, dim, nn3, aggr= 'mean')
            nn4 = Sequential(Linear(3,25),ReLU(),Linear(25,dim*dim))
            self.conv4 =NNConv(dim, dim, nn4, aggr= 'mean')
            nn5 = Sequential(Linear(3,25),ReLU(),Linear(25,dim*dim))
            self.conv5 =NNConv(dim, dim, nn5, aggr= 'mean')
            self.lin = Linear(dim,latent_dim)
            
        def forward(self,x,edge_index, edge_weight,batch): 
            x = self.conv1 (x, edge_index, edge_weight).relu()
            x = self.conv2 (x, edge_index, edge_weight).relu()
            x = self.conv3 (x, edge_index, edge_weight).relu()
            x = self.conv4 (x, edge_index, edge_weight).relu()
            x = self.conv5 (x, edge_index, edge_weight).relu()
            x = global_add_pool(x, batch= batch)
            x = self.lin(x).tanh()
            
            return x
        
    class Decoder(torch.nn.Module):
        def __init__(self,latent_dims):
            super(Decoder, self).__init__()
            self.seq_feature = 37
            num_layers =1
            self.expand1 = Linear(512, latent_dims[1])
            #self.expand2 = Linear(512, latent_dims[2])
            #self.expand3 = Linear(512, latent_dims[3])
            self.embedding = Embedding(self.seq_feature, latent_dims[0])
            self.LSTM1 = LSTM(latent_dims[0], latent_dims[1], num_layers, batch_first = True)
            #self.LSTM2 = LSTM(latent_dims[1], latent_dims[2], num_layers, batch_first = True)
            #self.LSTM3 = LSTM(latent_dims[2], latent_dims[3], num_layers, batch_first = True)
            self.lin1 = Linear(latent_dims[-1], 30)
            self.lin2 = Linear(30, self.seq_feature)
            
        def forward(self, in_seq, out_seq, lengths, z, teacher_forcing = False): 
            
            h10 = self.expand1(z)
            #h20 = self.expand2(z)
            #h30 = self.expand3(z)
            c10 = torch.zeros(h10.size()[0], h10.size()[1],requires_grad=True).to(current_device)
            #c20 = torch.zeros(h20.size()[0], h20.size()[1],requires_grad=True).to(current_device)
            #c30 = torch.zeros(h30.size()[0], h30.size()[1],requires_grad=True).to(current_device)
            h10 = h10.unsqueeze(0)
            c10 = c10.unsqueeze(0)
            #h20 = h20.unsqueeze(0)
            #c20 = c20.unsqueeze(0)
            #h30 = h30.unsqueeze(0)
            #c30 = c30.unsqueeze(0)
            
            decoder_output=[]
            #seq_packed= torch.nn.utils.rnn.pack_padded_sequence(seq, lengths= lengths.cpu(), batch_first= True, enforce_sorted= False)
            for i in range(in_seq.shape[1]):
                if (teacher_forcing == True) or (i ==0):
                    Input = in_seq[:,i].unsqueeze(1)
                else:
                    Input = x.detach()
                decoder_input = self.embedding(Input)
                out, (h1,c1) = self.LSTM1(decoder_input, (h10,c10))
                #out, (h2,c2) = self.LSTM2(out, (h20,c20))
                #out, (h3,c3) = self.LSTM3(out, (h30,c30))
                x= self.lin1(out).relu()
                x= self.lin2(x).relu()
                smax= LogSoftmax(dim=2)
                x=smax(x)
                decoder_output.append(x)
                x =torch.argmax(x,2)
                
                h10,c10 = h1,c1
                #h20,c20 = h2,c2
                #h30,c30 = h3,c3
                
            decoder_outputs = torch.cat(decoder_output, dim =1)
            return decoder_outputs
                    

    # In[32]:


    Enc = Encoder(latent_dim = 512)
    Enc.cuda();
    Dec = Decoder(latent_dims = [128, 512])#, 1024, 2048])
    Dec.cuda();
    
    Enc = DistributedDataParallel(Enc, device_ids=[current_device])
    Dec = DistributedDataParallel(Dec, device_ids=[current_device])

    dataset= MoleculeDataset(root=os.path.join(os.getcwd(),'Graph_Data/'))
    dataset.shuffle()
    batch_size=args.batch_size
    bucket_size=[10,20,30,40,50,60,70,80,90,100,200,700]
    train_sampler= DistributedBucketSampler(dataset[:897150],bucket_boundaries= bucket_size,batch_size= batch_size)
    val_sampler= DistributedBucketSampler(dataset[897150:],bucket_boundaries= bucket_size,batch_size= batch_size)
    train_dl= DataLoader(dataset[:897150],batch_sampler= train_sampler)
    val_dl = DataLoader(dataset[897150:],batch_sampler = val_sampler)

    # FOR TEST
    #batch_size = 100
    #bucket_size=[10,20,30,40,50,60,70,80,90,100,200,700]
    #train_sampler= DistributedBucketSampler(dataset[:20000],bucket_boundaries= bucket_size,batch_size= batch_size)
    #val_sampler= DistributedBucketSampler(dataset[20000:30000],bucket_boundaries= bucket_size,batch_size= batch_size)
    #train_dl= DataLoader(dataset[:20000],batch_sampler= train_sampler)
    #val_dl = DataLoader(dataset[20000:],batch_sampler = val_sampler)

    enc_optimizer = torch.optim.Adam(Enc.parameters(), lr=0.1)
    dec_optimizer = torch.optim.Adam(Dec.parameters(), lr=0.1)
    lossfn =NLLLoss()
    tloss=[]
    vloss=[]
    runtime=[]
    forced=[]
    for epoch in range(args.max_epochs):
        start=time.time()
        train_loss = []
        num_train = []
        val_loss = []
        num_val = []
        if (torch.rand(1) >=0.5) and epoch >99:
            tf= True
        else:
            tf= False
        forced.append(tf)
        for i,data in enumerate(train_dl):
            data.to(current_device);
            num_train.append(len(data))
            loss = train(Enc, Dec, data, enc_optimizer, dec_optimizer, lossfn, tf)
            train_loss.append(loss)
        avg_tloss= np.sum(np.multiply(train_loss,num_train))/np.sum(num_train)
        tloss.append(avg_tloss)
        for i,data in enumerate(val_dl):
            data.to(current_device);
            num_val.append(len(data))
            loss = validate(Enc,Dec, data, lossfn)
            val_loss.append(loss)
        avg_vloss= np.sum(np.multiply(val_loss,num_val))/np.sum(num_val)
        vloss.append(avg_vloss)
        end=time.time()
        runtime.append(end-start)
        if rank==0:
          print('Epoch [{}/{}] , Train loss: {:.4f}, Val loss: {:.4f}'.format(epoch+1,args.max_epochs,avg_tloss,avg_vloss))
          pd.DataFrame({'Train Loss':tloss,'Val Loss': vloss, 'Time': runtime, 'Forced': forced}).to_csv('LSTM/Losses.csv')
        
        if ((epoch+1) % 5 ==0) and rank ==0:
            torch.save({ 'epoch': epoch, 'model_state_dict' : Enc.state_dict(),
                       'optimizer_state_dict' : enc_optimizer.state_dict(), 'train_loss': avg_tloss,
                        'val_loss': avg_vloss},f'LSTM/Enc_GLSTM_{epoch+1}.pt')
            torch.save({ 'epoch': epoch, 'model_state_dict' : Dec.state_dict(),
                       'optimizer_state_dict' : dec_optimizer.state_dict(), 'train_loss': avg_tloss,
                        'val_loss': avg_vloss},f'LSTM/Dec_GLSTM_{epoch+1}.pt')
# In[33]:


def train(Enc,Dec,data,enc_optimizer, dec_optimizer,lossfn, tf):
    Enc.train()
    Dec.train()
    in_seq,out_seq, lengths, mask_in, mask_out = collate_func(data)
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()
    z = Enc(data.x, data.edge_index,data.edge_attr,data.batch)
    output = Dec(in_seq, out_seq, lengths, z, tf)
    vocab_size = output.size(-1)
    log_probas = output.view(-1,vocab_size)
    target= out_seq.view(-1)
    mask =mask_out.view(-1)
    loss = lossfn(log_probas[mask], target[mask])
    loss.backward()
    enc_optimizer.step()
    dec_optimizer.step()
    return loss.item()


# In[34]:


def validate(Enc,Dec,data,lossfn):
    with torch.no_grad():
        Enc.eval()
        Dec.eval()
        in_seq,out_seq, lengths, mask_in, mask_out = collate_func(data)
        z = Enc(data.x, data.edge_index,data.edge_attr,data.batch)
        output = Dec(in_seq, out_seq, lengths, z, False)
        vocab_size = output.size(-1)
        log_probas = output.view(-1,vocab_size)
        target= out_seq.view(-1)
        mask =mask_out.view(-1)
        loss = lossfn(log_probas[mask], target[mask])
    return loss.item()
    



# In[ ]:
def collate_func(batch):
    data= batch.to_data_list()
    in_sequence=[]
    out_sequence=[]
    for i in range(len(data)):
        in_sequence.append(data[i].in_seq)
        out_sequence.append(data[i].out_seq)
    lengths = torch.tensor([t.shape[0] for t in in_sequence])#.to(current_device)#add to device
    in_batch = torch.nn.utils.rnn.pad_sequence(in_sequence,batch_first= True, padding_value=0)#.to(current_device);
    out_batch = torch.nn.utils.rnn.pad_sequence(out_sequence,batch_first= True, padding_value=0)#.to(current_device);
    mask_in = (in_batch!=-1)#.to(current_device);
    mask_out = (out_batch!=-1)#.to(current_device);
    #for i in range(in_batch.shape[0]):
        #data[i].in_seq = in_batch[i]
        #data[i].out_seq = out_batch[i]
        #data[i].lengths = lengths[i]
    return in_batch, out_batch, lengths, mask_in,mask_out#,data



if __name__=='__main__':
   main()


# In[ ]:




