# coding: utf-8
import math
from re import L
from time import time
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy
from torch_scatter import scatter_add, scatter_mean
from torch_sparse import coalesce
from sklearn import metrics
# from d2l.torch import tensor
from tqdm import tqdm
import os
import numpy as np
import pickle
import copy

from collections import Counter
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import DataLoader
from torch_scatter import scatter_add

from discrimintor import utils, feats
# from discrimintor.data import dataset
from discrimintor.utils import logger
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
# from torch.utils.tensorboard import SummaryWriter
import yaml
from easydict import EasyDict
import pickle
from discrimintor import model, runner, utils

class DefaultRunner(object):
    def __init__(self, train_set, val_set, test_set, score_model, discriminator_model, optimizer, scheduler, gpus,
                 score_config, dg_config):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.gpus = gpus
        self.device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')
        self.config = score_config
        self.dg_config = dg_config

        self.batch_size = self.dg_config.train.batch_size

        self._score_model = score_model
        self._disc_model = discriminator_model
        self._optimizer = optimizer
        self._scheduler = scheduler
        # self.loss_f = loss

        self.best_loss = 1e20
        self.start_epoch = 0

        if self.device.type == 'cuda':
            if self._score_model:
                self._score_model = self._score_model.cuda(self.device)
            if self._disc_model:
                self._disc_model = self._disc_model.cuda(self.device)

    def save(self, checkpoint, epoch=None, var_list={}):

        state = {
            **var_list,
            "model": self._disc_model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "config": self.dg_config,
        }
        epoch = str(epoch) if epoch is not None else ''
        checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
        torch.save(state, checkpoint)

    def load(self, score_checkpoint, disc_checkpoint, score_epoch=None, disc_epoch=None, load_optimizer=False,
             load_scheduler=False):

        disc_epoch = str(disc_epoch) if disc_epoch is not None else ''
        score_epoch = str(score_epoch) if score_epoch is not None else ''
        # load score network
        score_checkpoint = os.path.join(score_checkpoint, 'checkpoint%s' % score_epoch)
        logger.log("Load score network checkpoint from %s" % score_checkpoint)
        state = torch.load(score_checkpoint, map_location=self.device)
        self._score_model.load_state_dict(state["model"], strict=False)
        # load discriminator
        disc_checkpoint = os.path.join(disc_checkpoint, 'checkpoint%s' % disc_epoch)
        logger.log("Load discriminator checkpoint from %s" % disc_checkpoint)
        state = torch.load(disc_checkpoint, map_location=self.device)

        self._disc_model.load_state_dict(state["model"])
        # self._model.load_state_dict(state["model"], strict=False)
        self.best_loss = state['best_loss']
        self.start_epoch = state['cur_epoch'] + 1

        if load_optimizer:
            self._optimizer.load_state_dict(state["optimizer"])
            if self.device.type == 'cuda':
                for state in self._optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.device)

        if load_scheduler:
            self._scheduler.load_state_dict(state["scheduler"])
    def load_dg(self,disc_checkpoint, disc_epoch=None, load_optimizer=False,
             load_scheduler=False):

        disc_epoch = str(disc_epoch) if disc_epoch is not None else ''
        # load discriminator
        disc_checkpoint = os.path.join(disc_checkpoint, 'checkpoint%s' % disc_epoch)
        logger.log("Load discriminator checkpoint from %s" % disc_checkpoint)
        state = torch.load(disc_checkpoint, map_location=self.device)
        self._disc_model.load_state_dict(state["model"])
        # self._model.load_state_dict(state["model"], strict=False)
        self.best_loss = state['best_loss']
        self.start_epoch = state['cur_epoch'] + 1
        return self._disc_model
    @torch.no_grad()
    def evaluate(self, split, verbose=0):
        """
        Evaluate the model.
        Parameters:
            split (str): split to evaluate. Can be ``train``, ``val`` or ``test``.
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError('split should be either train, val, or test.')

        test_set = getattr(self, "%s_set" % split)
        dataloader = DataLoader(test_set, batch_size=self.dg_config.train.batch_size, \
                                shuffle=True, num_workers=self.dg_config.train.num_workers)
        model = self._model
        loss_function = torch.nn.BCEWithLogitsLoss()

        model.eval()
        # code here
        eval_start = time()
        eval_losses = []
        for batch, labels in dataloader:
            if self.device.type == "cuda":
                batch = batch.to(self.device)
                labels = labels.to(self.device)

            scores = model(batch, self.device)
            loss = loss_function(scores, labels)
            eval_losses.append(loss.item())
        average_loss = sum(eval_losses) / len(eval_losses)

        if verbose:
            logger.log('Evaluate %s Loss: %.5f | Time: %.5f' % (split, average_loss, time() - eval_start))
        return average_loss
    @torch.no_grad()
    def get_distance(self, data):
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1)  # (num_edge, 1)
        data.edge_length = d
        return data

    @torch.no_grad()
    # extend the edge on the fly, second order: angle, third order: dihedral
    def extend_graph(self, data, order=3):

        def binarize(x):
            return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

        def get_higher_order_adj_matrix(adj, order):
            """
            Args:
                adj:        (N, N)
                type_mat:   (N, N)
            """
            adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                        binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

            for i in range(2, order + 1):
                adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
            order_mat = torch.zeros_like(adj)

            for i in range(1, order + 1):
                order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

            return order_mat

        num_types = len(utils.BOND_TYPES)

        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)  # (N, N)
        type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N)  # modify data
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N)  # modify data
        data.is_bond = (data.edge_type < num_types)
        assert (data.edge_index == edge_index_1).all()

        return data
    def train(self, verbose=1):
        # writer = SummaryWriter()
        train_start = time()

        num_epochs = self.dg_config.train.epochs
        dataloader = DataLoader(self.train_set,
                                batch_size=self.dg_config.train.batch_size,
                                shuffle=self.dg_config.train.shuffle,
                                num_workers=self.dg_config.train.num_workers)

        model = self._disc_model
        loss_function = torch.nn.BCEWithLogitsLoss()


        train_losses = []
        val_losses = []
        best_loss = self.best_loss
        start_epoch = self.start_epoch
        logger.log('start training...')
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.dg_config.model.sigma_begin), np.log(self.dg_config.model.sigma_end),
                               self.dg_config.model.num_noise_level)), dtype=torch.float32, device=self.device)
        for epoch in range(num_epochs):
            # train
            model.train()
            epoch_start = time()
            batch_losses = []
            batch_cnt = 0
            for batch, labels in dataloader:
                batch_cnt += 1
                if self.device.type == "cuda":
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)


                # batch = self.extend_graph(batch, self.dg_config.model.order)
                # batch = self.get_distance(batch)
                d = batch.edge_length  # (num_edge, 1)

                node2graph = batch.batch
                edge2graph = node2graph[batch.edge_index[0]]
                if self.dg_config.model.noise_type == 'symmetry':
                    num_nodes = scatter_add(torch.ones(batch.num_nodes, dtype=torch.long, device=self.device),
                                            node2graph)  # (num_graph)
                    num_cum_nodes = num_nodes.cumsum(0)  # (num_graph)
                    node_offset = num_cum_nodes - num_nodes  # (num_graph)
                    edge_offset = node_offset[edge2graph]  # (num_edge)

                    num_nodes_square = num_nodes ** 2  # (num_graph)
                    num_nodes_square_cumsum = num_nodes_square.cumsum(-1)  # (num_graph)
                    edge_start = num_nodes_square_cumsum - num_nodes_square  # (num_graph)
                    edge_start = edge_start[edge2graph]

                    all_len = num_nodes_square_cumsum[-1]

                    node_index = batch.edge_index.t() - edge_offset.unsqueeze(-1)
                    # node_in, node_out = node_index.t()
                    node_large = node_index.max(dim=-1)[0]
                    node_small = node_index.min(dim=-1)[0]
                    undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start

                    symm_noise = torch.zeros(all_len, device=self.device, dtype=torch.float).normal_()
                    d_noise = symm_noise[undirected_edge_id].unsqueeze(-1)  # (num_edge, 1)

                elif self.dg_config.model.noise_type == 'rand':
                    d_noise = torch.randn_like(d)
                else:
                    raise NotImplementedError('noise type must in [distance_symm, distance_rand]')
                assert d_noise.shape == d.shape

                noise_level = torch.randint(0, sigmas.size(0), (batch.num_graphs,),
                                            device=self.device)  # (num_graph)

                used_sigmas = sigmas[noise_level]  # (num_graph)
                used_sigmas = used_sigmas[edge2graph].unsqueeze(-1)  # (num_edge, 1)
                perturbed_d = d + d_noise * used_sigmas
                scores = model(batch, perturbed_d, used_sigmas)
                # print(scores.T, end="out\n")
                # print(labels, end="label\n")

                loss = loss_function(scores, labels)
                if not loss.requires_grad:
                    raise RuntimeError("loss doesn't require grad")

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()


                batch_losses.append(loss.item())

                if batch_cnt % self.dg_config.train.log_interval == 0 or (epoch == 0 and batch_cnt <= 10):
                    logger.log('Epoch: %d | Step: %d | loss: %.8f | Lr: %.7f' % \
                               (
                                   epoch + start_epoch, batch_cnt, batch_losses[-1],
                                   self._optimizer.param_groups[0]['lr']))

            average_loss = sum(batch_losses) / len(batch_losses)
            train_losses.append(average_loss)

            if verbose:
                logger.log('Epoch: %d | Train Loss: %.8f | Time: %.5f' % (
                    epoch + start_epoch, average_loss, time() - epoch_start))

            # evaluate
            if self.dg_config.train.eval:
                average_eval_loss = self.evaluate('val', verbose=1)
                val_losses.append(average_eval_loss)
            else:
                # use train loss as surrogate loss
                average_eval_loss = average_loss
                val_losses.append(average_loss)

            if self.dg_config.train.scheduler.type == "plateau":
                self._scheduler.step(average_eval_loss)
            else:
                self._scheduler.step()

            if self.dg_config.train.save:

                val_list = {
                    'cur_epoch': epoch + start_epoch,
                    'best_loss': best_loss,
                }
                self.save(self.dg_config.train.save_path + str(self.dg_config.train.seed) + "_lr0.0001", epoch + start_epoch, val_list)

        self.best_loss = best_loss
        self.start_epoch = start_epoch + num_epochs
        logger.log('optimization finished.')
        logger.log('Total time elapsed: %.5fs' % (time() - train_start))

    def discriminator_sampeler(self, data, model):
        with torch.no_grad():
            scores = model(data, self.device)
        return scores

    def SDE_PC_sampeler(self, data, SDE_model, discriminator_model, sigma=25, snr=0.16, num_steps=250,
                        eps=1e-3, num_langevin_steps=1):
        '''
        Args:
            data: GEOM-data, usually the batched data
            SDE_model: trained SDEGen model
            sigma: the diffusion coefficient chosen for adding noise
            snr: signal-to-noise ratio
            num_steps: the iteration number of the Euler-Maruyama solver
            eps: a small number used to prevent division by zero
            num_langevin_steps: the number of Langevin-MCMC steps at each time

        Returns: atomic distances

        '''
        d_vecs = []
        num_edge = len(data.edge_type)
        device = data.edge_index.device
        t = torch.ones(1, device=device)
        init_d = torch.randn(num_edge, device=device) * utils.marginal_prob_std(t, sigma, device=device)
        pos_init = torch.randn(data.num_nodes, 3).to(data.pos)
        time_steps = torch.linspace(1., eps, num_steps, device=device)
        step_size = time_steps[0] - time_steps[1]  # Δt
        d = init_d[:, None]
        batch_size = len(data.smiles)

        with torch.no_grad():
            for time_step in time_steps:
                batch_time_step = torch.ones(batch_size, device=device) * time_step
                g = utils.diffusion_coeff(time_step, sigma, device=device)
                # Corretor Step
                for i in range(num_langevin_steps):
                    grad = SDE_model.get_score(data, d, batch_time_step)
                    grad_norm = torch.norm(grad)  # ||grad||
                    noise_norm = np.sqrt(d.shape[0])  # sqrt(d_dim)
                    langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
                    d = d + (langevin_step_size * grad)[:, None] + (torch.sqrt(
                        2 * langevin_step_size) * torch.randn_like(d).squeeze(-1))[:, None]
                # Predictor Step
                mean_d = d + ((g ** 2) * ( SDE_model.get_score(data, d, batch_time_step) + 0.05 * time_step * discriminator_model.judge(data, d, batch_time_step)) * step_size)[:,None]  # 为什么没有z todo：需要修改的地方
                # mean_d = d + ((g ** 2) * ( SDE_model.get_score(data, d, batch_time_step) + 0) * step_size)[:,None]  # 为什么没有z todo：需要修改的地方
                d = mean_d + (torch.sqrt(step_size) * g * torch.randn_like(d).squeeze(-1))[:, None]
        return mean_d

    def sde_generator(self, data, socre_model, disciminator_model, num_steps, num_langevin_steps):
        '''

        Args:
            data: GEOM-MOL data
            model: trained_SDE model
        Returns: data, pos_traj, d_gen, d_recover

        '''

        # get the generated distance
        d_gen = self.SDE_PC_sampeler(data, socre_model, disciminator_model, num_steps=num_steps,
                                     num_langevin_steps=num_langevin_steps)
        (d_gen - data.edge_length).max()

        # discriminator score
        # d = self.discriminator_sampeler(data, model)
        # get the position traj
        num_nodes = len(data.atom_type)
        pos_init = torch.randn(num_nodes, 3).to(data.pos)
        embedder = utils.Embed3D()
        pos_traj, _ = embedder(d_gen.view(-1), data.edge_index, pos_init, data.edge_order)
        pos_gen = pos_traj[-1].cuda()
        d_recover = utils.get_d_from_pos(pos_gen, data.edge_index)

        data.pos_gen = pos_gen.to(data.pos)
        data.d_recover = d_recover.view(-1, 1).to(data.edge_length)
        data.d_gen = d_gen.view(-1, 1).to(data.edge_length)
        return data, pos_traj, d_gen, d_recover

    def sde_generate_samples_from_testset(self, start, end, num_repeat=None, out_path=None,
                                          file_name='sample_from_testset'):
        '''
        >> start&end: suppose the length of testset is 200, choosing the start=0, end=100 means 
            we sample data from testset[0:100].
        >> num_repeat: suppose one of the datas contained in the testset has 61 conformation, 
            num_repeat =2 means we generate 2 conformation for evaluation task, but if num_repeat=None,
            this means we are under the 2*num_pos_ref mode.
        >> out_path is self-explanatory.

        This function we use the packed testset to generate num_repeat times conformations
        than the reference conformations, and this is just because we want to compute the 
        COV and MAT metrics for sde method on a specific testset.
        For user who wants to generate conformations merely through throwing a smiles of a 
        molecule, we recommand he/she to use the sde_generate_samples_from_smiles.
        '''
        # load model

        test_set = self.test_set
        generate_start = time()
        all_data_list = []
        print('len of all data : %d' % len(test_set))
        # SDE_model = self._model
        for i in tqdm(range(len(test_set))):
            if i < start or i >= end:
                continue
            return_data = copy.deepcopy(test_set[i])
            num_repeat_ = num_repeat if num_repeat is not None else 2 * test_set[i].num_pos_ref.item()
            batch = utils.repeat_data(test_set[i], num_repeat_).to(self.device)
            embedder = utils.Embed3D(step_size=self.dg_config.test.gen.dg_step_size, \
                                     num_steps=self.dg_config.test.gen.dg_num_steps, \
                                     verbose=self.dg_config.test.gen.verbose)

            batch, pos_traj, d_gen, d_recover = self.sde_generator(batch, self._score_model, self._disc_model,
                                                                   self.dg_config.test.gen.num_euler_steps,
                                                                   self.dg_config.test.gen.num_langevin_steps)

            batch = batch.to('cpu').to_data_list()

            all_pos = []
            for i in range(len(batch)):
                all_pos.append(batch[i].pos_gen)
            return_data.pos_gen = torch.cat(all_pos, 0)  # (num_repeat * num_node, 3)
            return_data.num_pos_gen = torch.tensor([len(all_pos)], dtype=torch.long)
            all_data_list.append(return_data)
            return_data.d_gen = d_gen
        if out_path is not None:
            with open(os.path.join(out_path, file_name), "wb") as fout:
                pickle.dump(all_data_list, fout)
            print('save generated %s samples to %s done!' % ('sde', out_path))
        print('pos generation[%d-%d] done  |  Time: %.5f' % (start, end, time() - generate_start))

        return all_data_list

    def sde_generate_samples_from_mol(self, mol, num_repeat=2, out_path=None, file_name=None, num_steps=250,
                                      num_langevin_steps=2, useFF=False):
        data = feats.mol_to_data(mol)
        smiles = Chem.MolToSmiles(mol)
        '''
        generate sample based on smiles representation, this module mainly includes processing smiles and 
        '''

        if data is None:
            raise ValueError('invalid smiles: %s' % smiles)
        return_data = copy.deepcopy(data)
        batch = utils.repeat_data(data, num_repeat).to(self.device)
        batch, pos_traj, d_gen, d_recover = self.sde_generator(batch, self._model, num_steps, num_langevin_steps)
        batch = batch.to('cpu').to_data_list()
        all_pos = []
        for i in range(len(batch)):
            all_pos.append(batch[i].pos_gen)
        return_data.pos_gen = torch.cat(all_pos, 0)  # (num_repeat * num_node, 3)
        return_data.num_pos_gen = torch.tensor([len(all_pos)], dtype=torch.long)

        return_data.d_gen = d_gen

        if out_path is not None:

            m2 = return_data.rdmol
            AllChem.EmbedMolecule(m2)
            return_data.rdmol = m2

            pos_gen = return_data.pos_gen.view(-1, return_data.num_nodes, 3)
            num_gen = pos_gen.size(0)
            gen_mol_list = []
            for i in range(num_gen):
                gen_mol = utils.set_mol_positions(return_data.rdmol, pos_gen[i])
                if useFF == True:
                    MMFFOptimizeMolecule(gen_mol)
                gen_mol_list.append(gen_mol)
            file_name = os.path.join(out_path, file_name)
            if file_name[-4:] == '.sdf':
                writer = Chem.SDWriter(file_name)
                for i in range(len(gen_mol_list)):
                    writer.write(gen_mol_list[i])
                writer.close()
            else:
                file = open(file_name, 'wb')
                pickle.dump(gen_mol_list, file)
                file.close()
        return return_data

    @torch.no_grad()
    def test(self, model):

        test_set = self.test_set
        dataloader = DataLoader(test_set, batch_size=self.dg_config.train.batch_size, \
                                shuffle=True, num_workers=self.dg_config.train.num_workers)
        loss_function = torch.nn.BCEWithLogitsLoss()
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.dg_config.model.sigma_begin), np.log(self.dg_config.model.sigma_end),
                               self.dg_config.model.num_noise_level)), dtype=torch.float32, device=self.device)
        model.eval()
        # code here
        eval_losses = []
        accuracy = []
        for batch, labels in dataloader:
            if self.device.type == "cuda":
                batch = batch.to(self.device)
                labels = labels.to(self.device)
            # batch = self.extend_graph(batch, self.dg_config.model.order)
            # batch = self.get_distance(batch)
            d = batch.edge_length  # (num_edge, 1)

            node2graph = batch.batch
            edge2graph = node2graph[batch.edge_index[0]]
            if self.dg_config.model.noise_type == 'symmetry':
                num_nodes = scatter_add(torch.ones(batch.num_nodes, dtype=torch.long, device=self.device),
                                        node2graph)  # (num_graph)
                num_cum_nodes = num_nodes.cumsum(0)  # (num_graph)
                node_offset = num_cum_nodes - num_nodes  # (num_graph)
                edge_offset = node_offset[edge2graph]  # (num_edge)

                num_nodes_square = num_nodes ** 2  # (num_graph)
                num_nodes_square_cumsum = num_nodes_square.cumsum(-1)  # (num_graph)
                edge_start = num_nodes_square_cumsum - num_nodes_square  # (num_graph)
                edge_start = edge_start[edge2graph]

                all_len = num_nodes_square_cumsum[-1]

                node_index = batch.edge_index.t() - edge_offset.unsqueeze(-1)
                # node_in, node_out = node_index.t()
                node_large = node_index.max(dim=-1)[0]
                node_small = node_index.min(dim=-1)[0]
                undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start

                symm_noise = torch.zeros(all_len, device=self.device, dtype=torch.float).normal_()
                d_noise = symm_noise[undirected_edge_id].unsqueeze(-1)  # (num_edge, 1)

            elif self.dg_config.model.noise_type == 'rand':
                d_noise = torch.randn_like(d)
            else:
                raise NotImplementedError('noise type must in [distance_symm, distance_rand]')
            assert d_noise.shape == d.shape

            noise_level = torch.randint(0, sigmas.size(0), (batch.num_graphs,),
                                        device=self.device)  # (num_graph)
            # noise_level = torch.zeros(noise_level.size())

            used_sigmas = sigmas[noise_level]  # (num_graph)
            used_sigmas = used_sigmas[edge2graph].unsqueeze(-1)  # (num_edge, 1)
            perturbed_d = d + d_noise * used_sigmas
            out = model(batch, perturbed_d, used_sigmas)


            loss = loss_function(out, labels)
            eval_losses.append(loss.item())
            out = torch.sigmoid(out)
            # for item in out:
            #     print(item)
                # if 0.2 < item < 0.8:
                #     print(item, end='---')

            pred_label = [1 if prob >= 0.5 else 0 for prob in out]
            # out_numpy = [sigmoid(item.item()) for item in out]
            # out_numpy = sigmoid(out_numpy)
            # pred_label = pred_label.cpu().numpy()
            # labels = labels.cpu().numpy()
            # for i in range(labels.shape[0]):
            #     # print(pred_label[i], labels[i].item())
            #     # print(batch[i].smiles)
            #     if pred_label[i] != labels[i].item():
            #         print(pred_label[i], labels[i].item())
            #         print(batch[i].smiles)

                # else:
                #     print("--")
            # close_to_zero_count = np.sum(np.abs(pred_label) < 0.1)
            # # 计算比例
            # percentage_close_to_zero = close_to_zero_count / len(pred_label) * 100
            # print(percentage_close_to_zero)
            labels = [t.cpu().numpy() for t in labels]

            accuracy.append(metrics.accuracy_score(pred_label, labels))

        average_loss = sum(eval_losses) / len(eval_losses)
        average_accuracy = sum(accuracy) / len(accuracy)
        print('Average_loss = %.8f, average accuracy = %.8f' % (average_loss, average_accuracy))
        return average_loss, average_accuracy


    def sde_generate_samples_demo(self, mol, out_path):
        return_data = copy.deepcopy(mol)
        num_repeat_ = 1
        batch = utils.repeat_data(mol, num_repeat_).to(self.device)

        batch, pos_traj, d_gen, d_recover = self.sde_generator(batch, self._score_model, self._disc_model,
                                                               self.dg_config.test.gen.num_euler_steps,
                                                               self.dg_config.test.gen.num_langevin_steps)

        batch = batch.to('cpu').to_data_list()
        all_data_list = []
        all_pos = []
        all_traj = []
        for i in range(len(batch)):
            all_pos.append(batch[i].pos_gen)
        for i in range(len(batch)):
            all_traj.append(batch[i].pos_traj)
        return_data.pos_gen = torch.cat(all_pos, 0)  # (num_repeat * num_node, 3)
        return_data.pos_traj = torch.cat(all_traj, 0)  # (num_repeat * num_node, 3)
        return_data.num_pos_gen = torch.tensor([len(all_pos)], dtype=torch.long)
        all_data_list.append(return_data)
        return_data.d_gen = d_gen
        if out_path is not None:
            with open(os.path.join(out_path, file_name), "wb") as fout:
                pickle.dump(all_data_list, fout)
            print('save generated %s samples to %s done!' % ('sde', out_path))
        # print('pos generation[%d-%d] done  |  Time: %.5f' % (start, end, time() - generate_start))

        return all_data_list, all_traj


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

