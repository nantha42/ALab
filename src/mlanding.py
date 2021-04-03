from icecream import ic
import gym
import torch.nn as nn
import torch


env = gym.make('MountainCarContinuous-v0')
env.reset()


class Self_attention(nn.Module):
    def  __init__(self,dp = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dp)
        
    def forward(self,q,k,v,mask=None):
        matmul_qk = q @ k.transpose(-1,-2)
        scaled_attention_logits = matmul_qk/ torch.sqrt(torch.tensor(k.shape[-1],dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scaled_attention_logits,dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = attention_weights @ v 
        return [output, attention_weights.detach().clone()]

class AttentionHead(nn.Module):
    def __init__(self,d_model,d_features,dp ):
        super().__init__()
        self.attn = Self_attention(dp)
        self.query_tfm = nn.Linear(d_model,d_features)
        self.key_tfm = nn.Linear(d_model,d_features)
        self.values_tfm = nn.Linear(d_model,d_features)
        self.kmemory = None 
        self.vmemory = None 
    
    def forward(self,queries,key,values,mask=None,discard_mem = False ):
        if discard_mem :
            self.kmemory = None 
            self.vmemory = None 
            
        Q = self.query_tfm(queries)
        K = self.key_tfm(key)
        V = self.values_tfm(values)
         
        dK = K.detach().clone()
        dV = V.detach().clone()

        if self.kmemory == None: 
            self.kmemory = dK 
            self.vmemory = dV 
        else:
            K = torch.cat((K,self.kmemory),dim=1)
            V = torch.cat((V,self.vmemory),dim=1)
             
            self.kmemory = torch.cat((dK,self.kmemory),dim=1)# concating in sequence length 
            self.vmemory = torch.cat((dV,self.vmemory),dim=1)
            # print("Memory appended",self.kmemory.shape)

        x,att_weight = self.attn(Q,K,V,None)
        return x,att_weight
    
    def pop_last(self,n):
        if self.kmemory == None:
            return 
        if self.kmemory.shape[1] == n*32:
            self.kmemory = self.kmemory[:,:-32,:]
            self.vmemory = self.vmemory[:,:-32,:]
    
    def print_seq_length(self):
        ic(self.kmemory.shape)
        ic(self.vmemory.shape)
        
class MultiheadAttentionXL(nn.Module):
    def __init__(self,d_model,d_feature,n_heads,dp=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_feature = d_feature
        self.n_heads = n_heads
        assert d_model == d_feature * n_heads
        self.attn_heads = nn.ModuleList([
            AttentionHead(d_model, d_feature, dp) for _ in range(n_heads)
        ])
        self.projection = nn.Linear(d_feature * n_heads, d_model) 
  
    def forward(self,queries,keys,values,discard_mem = False,mask=None):
        comb = [attn(queries, keys, values, mask=mask,discard_mem = discard_mem) # (Batch, Seq, Feature)
             for i, attn in enumerate(self.attn_heads)]


        # log_size(x[0], "output of single head")
        attentions = []
        xs = []
        for u,att in comb:
            xs.append(u)
            attentions.append(att)
        # reconcatenate
        x = torch.cat(xs, dim=-1) # (Batch, Seq, D_Feature * n_heads)
        attentions = torch.cat(attentions,dim=-1)
        # log_size(x, "concatenated output")
        x = self.projection(x) # (Batch, Seq, D_Model)
        # log_size(x, "projected output")
        return x,attentions

    def pop_last(self,n):
        for i,attn in enumerate(self.attn_heads):
            attn.pop_last(n)



    
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Linear(2,16)
        self.atten = MultiheadAttentionXL(16,4,4,0.1)
        self.final_out = 

    def forward(self,x):
        t = self.encode(x)
        self.atten.pop_last(4)
        out = self.atten(t)
        return out

if __name__ == '__main__':

    for _ in range(500):
        env.render()
        o,r,d,i = env.step(env.action_space.sample())
        ic(o,r,d,i)
# 2. To check all env available, uninstalled ones are also shown