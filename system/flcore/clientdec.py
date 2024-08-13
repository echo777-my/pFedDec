import copy
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from torchvision import transforms


class PersonalizedVectorGenerator(nn.Module):          
    def __init__(self, input_dim, output_dim):
        super(PersonalizedVectorGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, client_info):
        personalization_vector = self.network(client_info)
        return personalization_vector
    
class GenericVectorGenerator(nn.Module):          
    def __init__(self, input_dim, output_dim):
        super(GenericVectorGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, client_info):
        generic_vector = self.network(client_info)
        return generic_vector


class clientdec:
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.previous_model_state = None

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.lamda = args.lamda

        in_dim = list(args.model.head.parameters())[0].shape[1]
        self.context = torch.rand(1, in_dim).to(self.device)

        self.personalization_dim = in_dim
        self.sample_per_class = torch.zeros(self.num_classes).to(self.device)

        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        self.sample_per_class = self.sample_per_class / torch.sum(self.sample_per_class)

        self.personalized_vector_generator = PersonalizedVectorGenerator(self.personalization_dim, self.personalization_dim).to(self.device)
        self.generic_vector_generator = GenericVectorGenerator(self.personalization_dim, self.personalization_dim).to(self.device)
    
        self.model = Ensemble(
            model=self.model, 
            cs=copy.deepcopy(kwargs['ConditionalSelection']), 
            head_g=copy.deepcopy(self.model.head), 
            feature_extractor=copy.deepcopy(self.model.feature_extractor),
            num_classes=self.num_classes,
            sample_per_class=self.sample_per_class,
            device=self.device,
            personalized_vector_generator = self.personalized_vector_generator,
            generic_vector_generator = self.generic_vector_generator
        )
        self.opt= torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.pm_train = []
        self.pm_test = []

        
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)    

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)
            
    def set_parameters(self, feature_extractor):
        for new_param, old_param in zip(feature_extractor.parameters(), self.model.model.feature_extractor.parameters()):
            old_param.data = new_param.data.clone()
            
        for new_param, old_param in zip(feature_extractor.parameters(), self.model.feature_extractor.parameters()):
            old_param.data = new_param.data.clone()


    def set_head_g(self, head):
        headw_ps = []
        for name, mat in self.model.model.head.named_parameters():
            if 'weight' in name:
                headw_ps.append(mat.data)
        headw_p = headw_ps[-1]
        for mat in headw_ps[-2::-1]:
            headw_p = torch.matmul(headw_p, mat)
        headw_p.detach_()
        self.context = torch.sum(headw_p, dim=0, keepdim=True)
        
        for new_param, old_param in zip(head.parameters(), self.model.head_g.parameters()):
            old_param.data = new_param.data.clone()

    def set_cs(self, cs):
        for new_param, old_param in zip(cs.parameters(), self.model.gate.cs.parameters()):
            old_param.data = new_param.data.clone()
    
    def save_generator_model(self, personalized_path, generic_path):
        torch.save(self.personalized_vector_generator.state_dict(), personalized_path)
        torch.save(self.generic_vector_generator.state_dict(), generic_path)

    def load_generator_model(self, personalized_path, generic_path):
        self.personalized_vector_generator.load_state_dict(torch.load(personalized_path))
        self.generic_vector_generator.load_state_dict(torch.load(generic_path))

    def save_con_items(self, items, tag='', item_path=None):
        self.save_item(self.pm_train, 'pm_train' + '_' + tag, item_path)
        self.save_item(self.pm_test, 'pm_test' + '_' + tag, item_path)
        for idx, it in enumerate(items):
            self.save_item(it, 'item_' + str(idx) + '_' + tag, item_path)

    def generate_upload_head(self):
        for (np, pp), (ng, pg) in zip(self.model.model.head.named_parameters(), self.model.head_g.named_parameters()):
            pg.data = pp * 0.5 + pg * 0.5

    def test_metrics(self):
        testloader = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        self.model.gate.pm_ = []
        self.model.gate.gm_ = []
        self.pm_test = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x, is_rep=False, context=self.context)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        self.pm_test.extend(self.model.gate.pm_)
        
        return test_acc, test_num, auc

    def save_model_state(self):
        self.previous_model_state = copy.deepcopy(self.model.state_dict())
    
    def cos(self, x, y):
        return F.cosine_similarity(x, y, dim=1)

    def train_cs_model(self):
        trainloader = self.load_train_data()
        self.model.train()

        total_loss_accumulated = 0
        class_loss_accumulated = 0
        mmd_loss_accumulated = 0
        contrastive_loss_accumulated = 0
        loss = 0

        personalized_model_path = os.path.join('1',f'client_{self.id}-personalized.pth')
        generic_model_path = os.path.join('1',f'client_{self.id}-generic.pth')
        if os.path.exists(personalized_model_path) and os.path.exists(generic_model_path):
            self.load_generator_model(personalized_model_path, generic_model_path)
        
        for _ in range(self.local_steps):
            self.model.gate.pm = []
            self.model.gate.gm = []
            self.pm_train = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output, rep, rep_base, rep_p, rep_g = self.model(x, is_rep=True, context=self.context)
                class_loss = self.loss(output, y) * 2
                mmd_loss = MMD(rep, rep_base, 'rbf', self.device) * self.lamda            

                posi = self.cos(rep_base, rep_g).reshape(-1, 1)
                nega = self.cos(rep_g, rep_p).reshape(-1, 1)
                logits = torch.cat((posi, nega), dim=1)
                temperature = 0.05  
                logits /= temperature
                labels = torch.zeros(logits.size(0)).cuda().long()

                criterion = torch.nn.CrossEntropyLoss()
                contrastive_loss = criterion(logits, labels) * 0.5

                loss = class_loss + mmd_loss + contrastive_loss

                total_loss_accumulated += loss.item()
                class_loss_accumulated += class_loss.item()
                mmd_loss_accumulated += mmd_loss.item()
                contrastive_loss_accumulated += contrastive_loss.item()

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        print(f"{'Average Total Loss:':<20} {total_loss_accumulated / len(trainloader):<10.5f} "
      f"{'Average Classify Loss:':<20} {class_loss_accumulated / len(trainloader):<10.5f} "
      f"{'Average MMD Loss:':<20} {mmd_loss_accumulated / len(trainloader):<10.5f} "
      f"{'Average Contrastive Loss:':<20} {contrastive_loss_accumulated / len(trainloader):<10.5f}"
      )
        
        self.save_generator_model(personalized_model_path, generic_model_path)

def MMD(x, y, kernel, device='cpu'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
    return torch.mean(XX + YY - 2. * XY)


class Ensemble(nn.Module):
    def __init__(self, model, cs, head_g, feature_extractor, num_classes, sample_per_class, device,personalized_vector_generator,generic_vector_generator) -> None:
        super().__init__()

        self.model = model
        self.head_g = head_g
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.sample_per_class = sample_per_class
        self.device = device
        self.personalized_vector_generator= personalized_vector_generator
        self.generic_vector_generator =  generic_vector_generator
        
        for param in self.head_g.parameters():
            param.requires_grad = False
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.flag = 0
        self.tau = 1
        self.hard = False
        self.context = None

        self.gate = Gate(cs)
        
        self.personalization_dim = list(self.model.head.parameters())[0].shape[1]
        self.embedding_dim = self.personalization_dim
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dim).to(self.device)

    def generate_personalized_fixed_input(self):
        fixed_input = torch.zeros(self.personalization_dim).to(self.device)
        embeddings = self.embedding(torch.tensor(range(self.num_classes), device=self.device))
        for l, emb in enumerate(embeddings):
            fixed_input += emb * self.sample_per_class[l]
        fixed_input /= sum(self.sample_per_class)
        return fixed_input
    
    def generate_generic_fixed_input(self):
        fixed_input = torch.zeros(self.personalization_dim).to(self.device)
        embeddings = self.embedding(torch.tensor(range(self.num_classes), device=self.device))
        for l, emb in enumerate(embeddings):
            fixed_input += emb / self.num_classes
        return fixed_input
    
    def compute_personalized_vector(self, batch_size):
        personalized_fixed_input = self.personalized_fixed_input.unsqueeze(0).expand(batch_size, -1)
        personalized_vector = self.personalized_vector_generator(personalized_fixed_input)
        return personalized_vector

    def compute_generic_vector(self, batch_size):
        generic_fixed_input = self.generic_fixed_input.unsqueeze(0).expand(batch_size, -1)
        generic_vector = self.generic_vector_generator(generic_fixed_input)
        return generic_vector


    def forward(self, x, is_rep=False, context=None):
        rep = self.model.feature_extractor(x)  # 个性化特征提取器提取的特征
        gate_in = rep

        self.personalized_fixed_input = self.generate_personalized_fixed_input()
        self.generic_fixed_input = self.generate_generic_fixed_input()

        personalization_vector = self.compute_personalized_vector(rep.shape[0])
        generic_vector = self.compute_generic_vector(rep.shape[0])

        if context is not None:
            context = F.normalize(context, p=2, dim=1)
            if type(x) == type([]):
                self.context = torch.tile(context, (x[0].shape[0], 1))
            else:
                self.context = torch.tile(context, (x.shape[0], 1))

        if self.context is not None:
            gate_in = rep * self.context

        if self.flag == 0:
            rep_p, rep_g = self.gate(rep, personalization_vector, generic_vector, gate_in, self.flag)
            output_p = self.model.head(rep_p)
            output_g = self.head_g(rep_g)
            output = output_p + output_g
        elif self.flag == 1:
            rep_p = self.gate(rep, personalization_vector, generic_vector, gate_in, self.flag)
            output = self.model.head(rep_p)
        else:
            rep_g = self.gate(rep, personalization_vector, generic_vector, gate_in, self.flag)
            output = self.head_g(rep_g)

        if is_rep:
            return output, rep, self.feature_extractor(x), rep_p, rep_g
        else:
            return output
        

class Gate(nn.Module):
    def __init__(self, cs) -> None:
        super().__init__()

        self.cs = cs
        self.pm = []
        self.gm = []
        self.pm_ = []
        self.gm_ = []

    def forward(self, rep, personalization_vector, generic_vector, flag=0):
        pm, gm = self.cs(rep, personalization_vector, generic_vector)
        if self.training:
            self.pm.extend(pm)
            self.gm.extend(gm)
        else:
            self.pm_.extend(pm)
            self.gm_.extend(gm)

        if flag == 0:
            rep_p = rep * pm
            rep_g = rep * gm
            return rep_p, rep_g
        elif flag == 1:
            return rep * pm
        else:
            return rep * gm