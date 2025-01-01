import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_tasks=1):
        super(Attn_Net_Gated, self).__init__()

        # Attention mechanisms for a and b
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh()
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid()
        )
        if dropout:
            self.attention_a.add_module("dropout_a", nn.Dropout(0.25))
            self.attention_b.add_module("dropout_b", nn.Dropout(0.25))

        self.attention_c = nn.Linear(D, n_tasks)  # Linear transformation to get attention scores

    def forward(self, x):
        a = self.attention_a(x)  # Size [N, D]
        b = self.attention_b(x)  # Size [N, D]
        A = a.mul(b)  # Element-wise multiplication, size [N, D]
        
        # Linear layer to output attention scores for each class, size [N, n_classes]
        A = self.attention_c(A)  # Output attention scores [N, n_classes]

        return A, x  # Return the attention scores and original input features


class GMA_multiple(nn.Module):
    def __init__(self, ndim=1024, gate=True, size_arg="big", dropout=False, n_classes=1, n_tasks=1):
        super(GMA_multiple, self).__init__()

        print("intializing GMA...")
        
        self.size_dict = {"small": [ndim, 512, 256], "big": [ndim, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))

        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_tasks=n_tasks)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)

        initialize_weights(self)

    def forward(self, h, attention_only=False):
        # Get the attention scores and features
        A, h = self.attention_net(h)  # A shape is [n_classes, num_tiles], h is [num_tiles, n_embeddings]

        # If not in attention_only mode, continue to compute logits
        A_raw = A.detach().cpu().numpy()  # Raw attention scores before softmax for saving

        if attention_only:
            return A_raw  # Return the attention scores directly if attention_only is True

        A = torch.transpose(A, 1, 0) 
        A = F.softmax(A, dim=1)  
        
        M = torch.mm(A, h)  # M is class-specific weighted embedding, shape [n_classes, n_embeddings]

        logits = torch.diagonal(self.classifier(M)).unsqueeze(0)  # logits shape: [1, n_classes]

        return logits 
    
class GMA(nn.Module):
    def __init__(self, ndim=1024, gate=True, size_arg="big", dropout=False, n_classes=1, n_tasks=1):
        super(GMA, self).__init__()

        print("intializing GMA...")
        
        self.size_dict = {"small": [ndim, 512, 256], "big": [ndim, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))

        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_tasks=n_tasks)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)

        initialize_weights(self)

    def forward(self, h, attention_only=False):
        A, h = self.attention_net(h)  
        A_raw = A.detach().cpu().numpy()

        if attention_only:
            return A_raw  
        
        A = torch.transpose(A, 1, 0) 
        A = F.softmax(A, dim=1)  
        
        M = torch.mm(A, h)  # M is class-specific weighted embedding, shape [n_classes, n_embeddings]

        logits = self.classifier(M)  # logits shape: [1, n_classes]
        return logits 

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()