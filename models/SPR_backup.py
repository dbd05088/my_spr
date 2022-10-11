import os
from copy import deepcopy
import tqdm
import torch
import logging
import time
import datetime
import torch.nn.functional as F
import colorful
import numpy as np
import networkx as nx
from tensorboardX import SummaryWriter
from .reservoir import reservoir
from components import Net
from utils import BetaMixture1D
from collections import Counter
logger = logging.getLogger()

class SPR(torch.nn.Module):
    """ Train Continual Model self-supervisedly
        Freeze when required to eval and finetune supervisedly using Purified Buffer.
    """
    def __init__(self, config, writer: SummaryWriter):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.writer = writer
        self.num_updates = 0
        self.purified_buffer = reservoir['purified'](config, config['purified_buffer_size'], config['purified_buffer_q_poa'])
        self.delay_buffer = reservoir['delay'](config, config['delayed_buffer_size'], config['delayed_buffer_q_poa'])
        self.start_time = time.time()

        self.E_max = config['E_max']
        self.total_sample_num = 0
        self.expert_step = 0
        self.base_step = 0
        self.base_ft_step = 0
        self.topk = 1
        self.expert_number = 0

        self.base = self.get_init_base(config)
        self.expert = self.get_init_expert(config)

        self.ssl_dir = os.path.join(os.path.dirname(os.path.dirname(self.config['log_dir'])),
                                    'noiserate_{}'.format(config['corruption_percent']),
                                    'expt_{}'.format(config['expert_train_epochs']),
                                    'randomseed_{}'.format(config['random_seed']))

        if os.path.exists(self.ssl_dir):
            with open(os.path.join(self.ssl_dir, 'idx_sets.npy'), 'rb') as f:
                self.debug_idxs = np.load(f, allow_pickle=True)

        self.total_samples = config['total_samples']

    def get_init_base(self, config):
        """get initialized base model"""

        base = Net['resnet_simclr18_ft'](config)
        optim_config = config['optimizer']
        lr_scheduler_config = deepcopy(config['lr_scheduler'])
        lr_scheduler_config['options'].update({'T_max': config['base_train_epochs']})

        base.setup_optimizer(optim_config)
        base.setup_lr_scheduler(lr_scheduler_config)
        return base

    def get_init_expert(self, config):
        """get initialized expert model"""
        expert = Net[config['net']](config)
        optim_config = config['optimizer2']
        lr_scheduler_config = deepcopy(config['lr_scheduler2'])
        lr_scheduler_config['options'].update({'T_max': config['expert_train_epochs']})

        expert.setup_optimizer(optim_config)
        expert.setup_lr_scheduler(lr_scheduler_config)
        return expert

    def get_init_base_ft(self, config):
        """get initialized eval model"""
        base_ft = Net[config['net'] + '_ft'](config)
        optim_config = config['optimizer_ft']
        lr_scheduler_config = config['lr_scheduler_ft']

        base_ft.setup_optimizer(optim_config)
        base_ft.setup_lr_scheduler(lr_scheduler_config)
        return base_ft

    def learn(self, x, y, corrupt, idx, step=None):
        x, y = x.cuda(), y.cuda()
        #self.expert = self.get_init_expert(self.config)
        for i in range(len(x)):
            self.delay_buffer.update(imgs=x[i: i + 1], cats=y[i: i + 1], corrupts=corrupt[i: i + 1], idxs=idx[i: i + 1])
            self.num_updates+=1
            self.total_sample_num+=1
            if self.num_updates<self.config['base_batch_size']-1:
                continue

            #if self.delay_buffer.is_full():
            if not os.path.exists(os.path.join(self.ssl_dir, 'model{}.ckpt'.format(self.expert_number))):
                #self.expert = self.get_init_expert(self.config)
                self.train_self_expert(self.total_sample_num)
                #print()
            else:
                self.expert.load_state_dict(
                    torch.load(os.path.join(self.ssl_dir, 'model{}.ckpt'.format(self.expert_number)),
                                map_location=self.device))
                ################### data consistency check ######################
                if torch.sum(self.delay_buffer.get('idxs') != torch.Tensor(self.debug_idxs[self.expert_number])) != 0:
                    raise Exception("it seems there is a data consistency problem: exp_num {}".format(self.expert_number))
                ################### data consistency check ######################
            # self.train_self_base()

            clean_idx, clean_p = self.cluster_and_sample()
            self.update_purified_buffer(clean_idx, clean_p, step)
            self.expert_number += 1

            if len(self.purified_buffer) != 0:
                self.train_self_base(self.total_sample_num)

            self.num_updates = 0
            # model update every sample이 들어올때마다
            # 단, purified buffer에서 get_batch하기!

            

    def update_purified_buffer(self, clean_idx, clean_p, step):
        """update purified buffer with the filtered samples"""
        self.purified_buffer.update(
            imgs=self.delay_buffer.get('imgs')[clean_idx],
            cats=self.delay_buffer.get('cats')[clean_idx],
            corrupts=self.delay_buffer.get('corrupts')[clean_idx],
            idxs=self.delay_buffer.get('idxs')[clean_idx],
            clean_ps=clean_p)

        self.delay_buffer.reset()
        if len(self.purified_buffer)!=0:
            print(colorful.bold_yellow(self.purified_buffer.state('corrupts')).styled_string)
            self.writer.add_scalar('buffer_corrupts', torch.sum(self.purified_buffer.get('corrupts')), step)

    def cluster_and_sample(self):
        """filter samples in delay buffer"""
        self.expert.eval()
        with torch.no_grad():
            xs = self.delay_buffer.get('imgs')
            ys = self.delay_buffer.get('cats')
            corrs = self.delay_buffer.get('corrupts')

            features = self.expert(xs)
            #print("features shape", features.shape)
            #print(features)
            features = F.normalize(features, dim=1)
            #print("normalized features")
            #print(features)
            clean_p = list()
            clean_idx = list()
            #print("***********************************************")
            for u_y in torch.unique(ys).tolist():
                # class별로 centrality를 측정하는 것
                y_mask = ys == u_y
                corr = corrs[y_mask]
                #print("corr")
                #print(corr)
                feature = features[y_mask]
                # ignore negative similairties
                _similarity_matrix = torch.relu(F.cosine_similarity(feature.unsqueeze(1), feature.unsqueeze(0), dim=-1))

                # stochastic ensemble
                _clean_ps = torch.zeros((self.E_max, len(feature)), dtype=torch.double)
                for _i in range(self.E_max):
                    similarity_matrix = (_similarity_matrix > torch.rand_like(_similarity_matrix)).type(torch.float32)
                    similarity_matrix[similarity_matrix == 0] = 1e-5  # add small num for ensuring positive matrix

                    g = nx.from_numpy_matrix(similarity_matrix.cpu().numpy())
                    info = nx.eigenvector_centrality(g, max_iter=6000, weight='weight') # index: value
                    centrality = [info[i] for i in range(len(info))]

                    bmm_model = BetaMixture1D(max_iters=10)
                    # fit beta mixture model
                    c = np.asarray(centrality)
                    c, c_min, c_max = bmm_model.outlier_remove(c)
                    #print("c", c , "c_min", c_min, "c_max", c_max)
                    c = bmm_model.normalize(c, c_min, c_max)
                    #print("!!c", c)
                    bmm_model.fit(c)
                    bmm_model.create_lookup(1) # 0: noisy, 1: clean

                    # get posterior
                    c = np.asarray(centrality)
                    #print("before normalize c", c)
                    c = bmm_model.normalize(c, c_min, c_max)
                    #print("after normalize c", c)
                    p = bmm_model.look_lookup(c) # lowest p를 갖는 sample들이 제거됨
                    _clean_ps[_i] = torch.from_numpy(p)
                
                #print("before norm_clean_ps", _clean_ps)
                _clean_ps = torch.mean(_clean_ps, dim=0)
                #print("_clean_ps", _clean_ps)
                m = _clean_ps > torch.rand_like(_clean_ps)
                #print("m", m)

                clean_idx.extend(torch.nonzero(y_mask)[:, -1][m].tolist())
                clean_p.extend(_clean_ps[m].tolist())

                #print("class: {}".format(u_y))
                #print("--- num of selected samples: {}".format(torch.sum(m).item()))
                #print("--- num of selected corrupt samples: {}".format(torch.sum(corr[m]).item()))
            #print("***********************************************")
        return clean_idx, torch.Tensor(clean_p)


    def train_self_base(self, sample_num):
        """Self Replay. train base model with samples from delay and purified buffer"""
        bs = self.config['base_batch_size']
        # If purified buffer is full, train using it also
        '''
        db_bs = (bs // 2) if self.purified_buffer.is_full() else bs
        db_bs = min(db_bs, len(self.delay_buffer))
        pb_bs = min(bs - db_bs, len(self.purified_buffer))
        '''
        bs = min(bs, len(self.purified_buffer))
        self.base.train()
        #self.base.init_ntxent(self.config, batch_size = bs)

        total_loss, correct, num_data = 0.0, 0.0, 0.0

        #TODO 매우 비효율적인 code
        #for epoch_i in tqdm.trange(self.config['base_train_epochs'], desc="base training", leave=False):
        for i in range(self.config['base_train_epochs']):
            #dataloader = self.purified_buffer.get_dataloader(batch_size=bs, shuffle=True, drop_last=True)
            #for inner_step, data in enumerate(dataloader):
            #for i in range():
            replay_data = self.purified_buffer.sample(num = bs)
            x, y = replay_data['imgs'], replay_data['cats']
            y = y.type(torch.LongTensor)
            y = y.to(self.device)
            self.base.zero_grad()

            loss, logit = self.base.get_sup_loss(x, y)
            _, preds = logit.topk(self.topk, 1, True, True)
            loss = loss.mean()
            loss.backward()
            #self.base.clip_grad() # TODO
            self.base.optimizer.step()
            self.base.lr_scheduler.step()

            '''
            self.writer.add_scalar(
                'continual_base_train_loss', loss,
                self.base_step + inner_step + epoch_i * len(dataloader))
            '''
            # warmup for the first 3 epoch
            '''
            if i >= 2:
                self.base.lr_scheduler.step()
            '''

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
        self.purified_buffer.memory_statistics()
        self.report_training(sample_num, total_loss / self.config['base_train_epochs'], correct / num_data)
        self.writer.flush()
        self.base_step += self.config['base_train_epochs'] # * len(dataloader)

    def report_training(self, sample_num, train_loss, train_acc, contra_loss = False):
        self.writer.add_scalar(f"train/loss", train_loss, sample_num)
        self.writer.add_scalar(f"train/acc", train_acc, sample_num)
        if not contra_loss:
            print(
                f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"lr {self.base.optimizer.param_groups[0]['lr']:.6f} | "
                f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
                f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
            )
        else:
            print(
                f"Train | Sample # {sample_num} | contra_loss {train_loss:.4f} | "
                f"lr {self.expert.optimizer.param_groups[0]['lr']:.6f} | "
                f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
                f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
            )

    def train_self_expert(self, sample_num):
        """train expert model with samples from delay"""
        bs = min(self.config['expert_batch_size'], len(self.delay_buffer))
        self.expert.train()
        self.expert.init_ntxent(self.config, batch_size=bs)
    
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        #dataloader = self.delay_buffer.get_dataloader(batch_size=batch_size, shuffle=True, drop_last=True)
        #for epoch_i in tqdm.trange(self.config['expert_train_epochs'], desc='expert training', leave=False):
        #    for inner_step, data in enumerate(dataloader):
        # for i in range(self.config['expert_train_epochs']):
        for i in range(bs):
        #for i in range(32):
            replay_data = self.delay_buffer.sample(num = bs)
            x = replay_data['imgs']
            self.expert.zero_grad()
            loss = self.expert.get_selfsup_loss(x)
            loss.backward()
            self.expert.optimizer.step()
            '''
            self.writer.add_scalar(
                'expert_train_loss', loss,
                self.expert_step + inner_step + len(dataloader) * epoch_i)
            '''
            # warmup for the first 10 epochs
            '''
            if epoch_i >= 10:
                self.expert.lr_scheduler.step()
            '''
            total_loss += loss.item()
            num_data += bs

        self.report_training(sample_num, total_loss / self.config['expert_train_epochs'], 0, True)
        self.writer.flush()
        self.expert_step += self.config['expert_train_epochs'] #* len(dataloader)

    def get_finetuned_model(self):
        """copy the base and fine-tune for evaluation"""
        '''
        base_ft = self.get_init_base_ft(self.config)
        # overwrite entries in the state dict
        ft_dict = base_ft.state_dict()
        ft_dict.update({k: v for k, v in self.base.state_dict().items() if k in ft_dict})
        base_ft.load_state_dict(ft_dict)
        '''
        '''
        base_ft.train()
        dataloader = self.purified_buffer.get_dataloader(batch_size=self.config['ft_batch_size'], shuffle=True, drop_last=True)
        for epoch_i in tqdm.trange(self.config['ft_epochs'], desc='finetuning', leave=False):
            for inner_step, data in enumerate(dataloader):
                x, y = data['imgs'], data['cats']
                base_ft.zero_grad()
                loss = base_ft.get_sup_loss(x, y).mean()
                loss.backward()
                base_ft.clip_grad()
                base_ft.optimizer.step()
                base_ft.lr_scheduler.step()

                self.writer.add_scalar(
                    'ft_train_loss', loss,
                    self.base_ft_step + inner_step + epoch_i * len(dataloader))

        self.writer.flush()
        self.base_ft_step += self.config['ft_epochs'] * len(dataloader)
        base_ft.eval()
        '''

        #return base_ft
        self.base.eval()
        return self.base

    def forward(self, x):
        pass
