# coding: utf-8
from time import time
from tqdm import tqdm
import os
import numpy as np
import pickle
import copy

import rdkit
from rdkit import Chem

import torch
from torch_geometric.data import DataLoader
from torch_scatter import scatter_add

from confgf import utils, dataset


class DefaultRunner(object):
    def __init__(self, train_set, val_set, test_set, model, optimizer, scheduler, gpus, config):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.gpus = gpus
        self.device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')
        self.config = config

        self.batch_size = self.config.train.batch_size

        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler

        self.best_loss = 100.0
        self.start_epoch = 0

        if self.device.type == 'cuda':
            self._model = self._model.cuda(self.device)

    def save(self, checkpoint, epoch=None, var_list={}):

        state = {
            **var_list,
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "config": self.config
        }
        epoch = str(epoch) if epoch is not None else ''
        checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
        torch.save(state, checkpoint)

    def load(self, checkpoint, epoch=None, load_optimizer=False, load_scheduler=False):

        epoch = str(epoch) if epoch is not None else ''
        checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
        print("Load checkpoint from %s" % checkpoint)

        state = torch.load(checkpoint, map_location=self.device)
        self._model.load_state_dict(state["model"])
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
        dataloader = DataLoader(test_set, batch_size=self.config.train.batch_size, \
                                shuffle=False, num_workers=self.config.train.num_workers)
        model = self._model
        model.eval()
        # code here
        eval_start = time()
        eval_losses = []
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = batch.to(self.device)

            loss = model(batch)
            loss = loss.mean()
            eval_losses.append(loss.item())
        average_loss = sum(eval_losses) / len(eval_losses)

        if verbose:
            print('Evaluate %s Loss: %.5f | Time: %.5f' % (split, average_loss, time() - eval_start))
        return average_loss

    def train(self, verbose=1):
        train_start = time()

        num_epochs = self.config.train.epochs
        dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size,
                                shuffle=self.config.train.shuffle, num_workers=self.config.train.num_workers)

        model = self._model
        train_losses = []
        val_losses = []
        best_loss = self.best_loss
        start_epoch = self.start_epoch
        print('start training...')

        for epoch in range(num_epochs):
            # train
            model.train()
            epoch_start = time()
            batch_losses = []
            batch_cnt = 0
            for batch in dataloader:
                batch_cnt += 1
                if self.device.type == "cuda":
                    batch = batch.to(self.device)

                loss = model(batch)
                loss = loss.mean()
                if not loss.requires_grad:
                    raise RuntimeError("loss doesn't require grad")
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                batch_losses.append(loss.item())

                if batch_cnt % self.config.train.log_interval == 0 or (epoch == 0 and batch_cnt <= 10):
                    # if batch_cnt % self.config.train.log_interval == 0:
                    print('Epoch: %d | Step: %d | loss: %.5f | Lr: %.5f' % \
                          (epoch + start_epoch, batch_cnt, batch_losses[-1], self._optimizer.param_groups[0]['lr']))

            average_loss = sum(batch_losses) / len(batch_losses)
            train_losses.append(average_loss)

            if verbose:
                print('Epoch: %d | Train Loss: %.5f | Time: %.5f' % (
                epoch + start_epoch, average_loss, time() - epoch_start))

            # evaluate
            if self.config.train.eval:
                average_eval_loss = self.evaluate('val', verbose=1)
                val_losses.append(average_eval_loss)
            else:
                # use train loss as surrogate loss
                average_eval_loss = average_loss
                val_losses.append(average_loss)

            if self.config.train.scheduler.type == "plateau":
                self._scheduler.step(average_eval_loss)
            else:
                self._scheduler.step()

            if val_losses[-1] < best_loss:
                best_loss = val_losses[-1]
                if self.config.train.save:
                    val_list = {
                        'cur_epoch': epoch + start_epoch,
                        'best_loss': best_loss,
                    }
                    self.save(self.config.train.save_path, epoch + start_epoch, val_list)
        self.best_loss = best_loss
        self.start_epoch = start_epoch + num_epochs
        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))

    @torch.no_grad()
    def convert_score_d(self, score_d, pos, edge_index, edge_length):
        dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])  # (num_edge, 3)
        score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0)

        return score_pos

    @torch.no_grad()
    def distance_Langevin_Dynamics(self, data, d_mod, scorenet, sigmas,
                                   n_steps_each=100, step_lr=0.00002,
                                   clip=1000, min_sigma=0):
        """
        d_mod: initial distance vector. (num_edge, 1)
        """
        scorenet.eval()
        d_vecs = []
        cnt_sigma = 0

        for i, sigma in tqdm(enumerate(sigmas), total=sigmas.size(0), desc="Sampling distances"):
            if sigma < min_sigma:
                break
            cnt_sigma += 1
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for step in range(n_steps_each):
                noise = torch.randn_like(d_mod) * torch.sqrt(step_size * 2)
                score_d = scorenet.get_score(data, d_mod, sigma)  # (num_edge, 1)
                score_d = utils.clip_norm(score_d, limit=clip)
                d_mod = d_mod + step_size * score_d + noise
                d_vecs.append(d_mod)
        d_vecs = torch.stack(d_vecs, dim=0).view(cnt_sigma, n_steps_each, -1, 1)  # (sigams, 100, num_edge, 1)

        return data, d_vecs

    @torch.no_grad()
    def position_Langevin_Dynamics(self, data, pos_init, scorenet, sigmas,
                                   n_steps_each=100, step_lr=0.00002,
                                   clip=1000, min_sigma=0):
        """
        # 1. initial pos: (N, 3) 
        # 2. get d: (num_edge, 1)
        # 3. get score of d: score_d = self.get_grad(d).view(-1) (num_edge)
        # 4. get score of pos:
        #        dd_dr = (1/d) * (pos[edge_index[0]] - pos[edge_index[1]]) (num_edge, 3)
        #        edge2node = edge_index[0] (num_edge)
        #        score_pos = scatter_add(dd_dr * score_d, edge2node) (num_node, 3)
        # 5. update pos:
        #    pos = pos + step_size * score_pos + noise
        """
        scorenet.eval()
        pos_vecs = []
        pos = pos_init
        cnt_sigma = 0
        for i, sigma in tqdm(enumerate(sigmas), total=sigmas.size(0), desc="Sampling positions"):
            if sigma < min_sigma:
                break
            cnt_sigma += 1
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for step in range(n_steps_each):
                d = utils.get_d_from_pos(pos, data.edge_index).unsqueeze(-1)  # (num_edge, 1)

                noise = torch.randn_like(pos) * torch.sqrt(step_size * 2)
                score_d = scorenet.get_score(data, d, sigma)  # (num_edge, 1)
                score_pos = self.convert_score_d(score_d, pos, data.edge_index, d)
                score_pos = utils.clip_norm(score_pos, limit=clip)

                pos = pos + step_size * score_pos + noise  # (num_node, 3)
                pos_vecs.append(pos)

        pos_vecs = torch.stack(pos_vecs, dim=0).view(cnt_sigma, n_steps_each, -1, 3)  # (sigams, 100, num_node, 3)

        return data, pos_vecs

    def ConfGF_generator(self, data, config, pos_init=None):

        """
        The ConfGF generator that generates conformations using the score of atomic coordinates
        Return: 
            The generated conformation (pos_gen)
            Distance of the generated conformation (d_recover)
        """

        if pos_init is None:
            pos_init = torch.randn(data.num_nodes, 3).to(data.pos)
        data, pos_traj = self.position_Langevin_Dynamics(data, pos_init, self._model, self._model.sigmas.data.clone(), \
                                                         n_steps_each=config.steps_pos, step_lr=config.step_lr_pos, \
                                                         clip=config.clip, min_sigma=config.min_sigma)
        pos_gen = pos_traj[-1, -1]  # (num_node, 3) fetch the lastest pos

        d_recover = utils.get_d_from_pos(pos_gen, data.edge_index)  # (num_edges)

        data.pos_gen = pos_gen.to(data.pos)
        data.d_recover = d_recover.view(-1, 1).to(data.edge_length)
        return pos_gen, d_recover.view(-1), data, pos_traj

    def ConfGFDist_generator(self, data, config, embedder=utils.Embed3D(), pos_init=None):

        d = torch.rand(data.edge_index.size(1), 1, device=self.device)  # (num_edge, 1)
        data, d_traj = self.distance_Langevin_Dynamics(data, d, self._model, self._model.sigmas.data.clone(), \
                                                       n_steps_each=config.steps_d, step_lr=config.step_lr_d, \
                                                       clip=config.clip, min_sigma=config.min_sigma)

        d_gen = d_traj[-1, -1].view(-1)  # fetch the lastest d (num_edge, )
        if pos_init is None:
            pos_init = torch.randn(data.num_nodes, 3).to(data.pos)

        pos_traj, _ = embedder(d_gen.view(-1),
                               data.edge_index,
                               pos_init,
                               data.edge_order)  # (num_steps, num_node, 3)
        pos_gen = pos_traj[-1]  # (num_nodes, 3)
        d_recover = utils.get_d_from_pos(pos_gen, data.edge_index)  # (num_edges)

        data.pos_gen = pos_gen.to(data.pos)
        data.d_gen = d_gen.view(-1, 1).to(data.edge_length)
        data.d_recover = d_recover.view(-1, 1).to(data.edge_length)
        return pos_gen, d_recover.view(-1), data, pos_traj

    def generate_samples_from_smiles(self, smiles, generator, num_repeat=1, keep_traj=False, out_path=None):

        if keep_traj:
            assert num_repeat == 1, "to generate the trajectory of conformations, you must set num_repeat to 1"

        data = dataset.smiles_to_data(smiles)

        if data is None:
            raise ValueError('invalid smiles: %s' % smiles)

        return_data = copy.deepcopy(data)
        batch = utils.repeat_data(data, num_repeat).to(self.device)

        if generator == 'ConfGF':
            _, _, batch, pos_traj = self.ConfGF_generator(batch, self.config.test.gen)  # (sigams, 100, num_node, 3)
        elif generator == 'ConfGFDist':
            embedder = utils.Embed3D(step_size=self.config.test.gen.dg_step_size, \
                                     num_steps=self.config.test.gen.dg_num_steps, \
                                     verbose=self.config.test.gen.verbose)
            _, _, batch, pos_traj = self.ConfGFDist_generator(batch, self.config.test.gen,
                                                              embedder)  # (num_steps, num_node, 3)
        else:
            raise NotImplementedError

        batch = batch.to('cpu').to_data_list()
        pos_traj = pos_traj.view(-1, 3).to('cpu')
        pos_traj_step = pos_traj.size(0) // return_data.num_nodes

        all_pos = []
        for i in range(len(batch)):
            all_pos.append(batch[i].pos_gen)
        return_data.pos_gen = torch.cat(all_pos, 0)  # (num_repeat * num_node, 3)
        return_data.num_pos_gen = torch.tensor([len(all_pos)], dtype=torch.long)
        if keep_traj:
            return_data.pos_traj = pos_traj
            return_data.num_pos_traj = torch.tensor([pos_traj_step], dtype=torch.long)

        if out_path is not None:
            with open(os.path.join(out_path, '%s_%s.pkl' % (generator, return_data.smiles)), "wb") as fout:
                pickle.dump(return_data, fout)
            print('save generated %s samples to %s done!' % (generator, out_path))
            print('the file is :', os.path.join(out_path, '%s_%s.pkl' % (generator, return_data.smiles)))

        print('pos generation of %s done' % return_data.smiles)

        return return_data

    def generate_samples_from_testset(self, start, end, generator, num_repeat=None, out_path=None):

        test_set = self.test_set

        generate_start = time()

        all_data_list = []
        print('len of all data: %d' % len(test_set))

        for i in tqdm(range(len(test_set))):
            if i < start or i >= end:
                continue
            return_data = copy.deepcopy(test_set[i])
            num_repeat_ = num_repeat if num_repeat is not None else 2 * test_set[i].num_pos_ref.item()
            batch = utils.repeat_data(test_set[i], num_repeat_).to(self.device)

            if generator == 'ConfGF':
                _, _, batch, _ = self.ConfGF_generator(batch, self.config.test.gen)
            elif generator == 'ConfGFDist':
                embedder = utils.Embed3D(step_size=self.config.test.gen.dg_step_size, \
                                         num_steps=self.config.test.gen.dg_num_steps, \
                                         verbose=self.config.test.gen.verbose)
                _, _, batch, _ = self.ConfGFDist_generator(batch, self.config.test.gen, embedder)
            else:
                raise NotImplementedError

            batch = batch.to('cpu').to_data_list()

            all_pos = []
            for i in range(len(batch)):
                all_pos.append(batch[i].pos_gen)
            return_data.pos_gen = torch.cat(all_pos, 0)  # (num_repeat * num_node, 3)
            return_data.num_pos_gen = torch.tensor([len(all_pos)], dtype=torch.long)
            all_data_list.append(return_data)

        if out_path is not None:
            with open(os.path.join(out_path, 'val_gen%s_s%de%depoch%dmin_sig%.3f.pkl' % (
            generator, start, end, self.config.test.epoch, self.config.test.gen.min_sigma)), "wb") as fout:
                pickle.dump(all_data_list, fout)
            print('save generated %s samples to %s done!' % (generator, out_path))
            print('the file is :', os.path.join(out_path, 'val_gen%s_s%de%depoch%dmin_sig%.3f.pkl' % (
            generator, start, end, self.config.test.epoch, self.config.test.gen.min_sigma)))
        print('pos generation[%d-%d] done  |  Time: %.5f' % (start, end, time() - generate_start))

        return all_data_list

    def dg_generate_samples_from_testset(self, start, end, generator, num_repeat=None, out_path=None, dg_model=None,
                                         w_dg=0.5, seed=2021, tag=""):


        # 指定文件夹路径
        folder_path = os.path.join(out_path, '%s%s_epoch%dmin_sig%.3f_dg_%.3f_seed%d' % (
                    tag, generator, self.config.test.epoch, self.config.test.gen.min_sigma, w_dg, seed,
            ))


        # 或者，如果你想要检查文件夹是否存在，可以使用os.path.exists()函数
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        test_set = self.test_set

        generate_start = time()

        all_data_list = []
        print('len of all data: %d' % len(test_set))
        number = 0
        for i in tqdm(range(len(test_set))):
            if i < start or i >= end:
                continue
            return_data = copy.deepcopy(test_set[i])
            num_repeat_ = num_repeat if num_repeat is not None else 2 * test_set[i].num_pos_ref.item()
            batch = utils.repeat_data(test_set[i], num_repeat_).to(self.device)

            if generator == 'ConfGF':
                _, _, batch, _ = self.dg_ConfGF_generator(batch, self.config.test.gen, dg_model, w_dg)
            elif generator == 'ConfGFDist':
                embedder = utils.Embed3D(step_size=self.config.test.gen.dg_step_size, \
                                         num_steps=self.config.test.gen.dg_num_steps, \
                                         verbose=self.config.test.gen.verbose)
                _, _, batch, _ = self.ConfGFDist_generator(batch, self.config.test.gen, embedder)
            else:
                raise NotImplementedError

            batch = batch.to('cpu').to_data_list()

            all_pos = []
            for i in range(len(batch)):
                all_pos.append(batch[i].pos_gen)
            return_data.pos_gen = torch.cat(all_pos, 0)  # (num_repeat * num_node, 3)
            return_data.num_pos_gen = torch.tensor([len(all_pos)], dtype=torch.long)
            all_data_list.append(return_data)
            # 生成一个存储一个
            with open(os.path.join(folder_path, '%d.pkl' %(number)), "wb") as fout:
                pickle.dump(all_data_list, fout)
            number += 1
            # print('save generated %s samples to %s done!' % (
            # generator, os.path.join(out_path, '%s_qm9_epoch%dmin_sig%.3f_dg_%.3f_%d_reversenums%d.pkl' % (
            #     generator, self.config.test.epoch, self.config.test.gen.min_sigma, w_dg, seed,
            #     self.config.test.gen.dg_num_steps))))

        if folder_path is not None:
            with open(os.path.join(folder_path, 'new_%s_epoch%dmin_sig%.3f_dg_%.3f_seed%d.pkl' % (
                    generator, self.config.test.epoch, self.config.test.gen.min_sigma, w_dg, seed,
            )), "wb") as fout:
                pickle.dump(all_data_list, fout)
            print('save generated %s samples to %s done!' % (
            generator, os.path.join(folder_path, '%s_qm9_epoch%dmin_sig%.3f_dg_%.3f_%d_reversenums%d.pkl' % (
                generator, self.config.test.epoch, self.config.test.gen.min_sigma, w_dg, seed,
                self.config.test.gen.dg_num_steps))))
        print('pos generation[%d-%d] done  |  Time: %.5f' % (start, end, time() - generate_start))

        return all_data_list

    def dg_ConfGF_generator(self, data, config, dg_model, w_dg, pos_init=None):

        """
        The ConfGF generator that generates conformations using the score of atomic coordinates
        Return:
            The generated conformation (pos_gen)
            Distance of the generated conformation (d_recover)
        """

        if pos_init is None:
            pos_init = torch.randn(data.num_nodes, 3).to(data.pos)
        data, pos_traj = self.dg_position_Langevin_Dynamics(dg_model, w_dg, data, pos_init, self._model,
                                                            self._model.sigmas.data.clone(), \
                                                            n_steps_each=config.steps_pos, step_lr=config.step_lr_pos, \
                                                            clip=config.clip, min_sigma=config.min_sigma)
        pos_gen = pos_traj[-1, -1]  # (num_node, 3) fetch the lastest pos

        d_recover = utils.get_d_from_pos(pos_gen, data.edge_index)  # (num_edges)

        data.pos_gen = pos_gen.to(data.pos)
        data.d_recover = d_recover.view(-1, 1).to(data.edge_length)
        return pos_gen, d_recover.view(-1), data, pos_traj

    @torch.no_grad()
    def dg_position_Langevin_Dynamics(self, dg_model, w_dg, data, pos_init, scorenet, sigmas,
                                      n_steps_each=100, step_lr=0.00002,
                                      clip=1000, min_sigma=0):
        """
        # 1. initial pos: (N, 3)
        # 2. get d: (num_edge, 1)
        # 3. get score of d: score_d = self.get_grad(d).view(-1) (num_edge)
        # 4. get score of pos:
        #        dd_dr = (1/d) * (pos[edge_index[0]] - pos[edge_index[1]]) (num_edge, 3)
        #        edge2node = edge_index[0] (num_edge)
        #        score_pos = scatter_add(dd_dr * score_d, edge2node) (num_node, 3)
        # 5. update pos:
        #    pos = pos + step_size * score_pos + noise
        """
        scorenet.eval()
        pos_vecs = []
        pos = pos_init
        cnt_sigma = 0
        res = []
        for i, sigma in tqdm(enumerate(sigmas), total=sigmas.size(0), desc="Sampling positions"):
            if sigma < min_sigma:
                break
            cnt_sigma += 1
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for step in range(n_steps_each):
                d = utils.get_d_from_pos(pos, data.edge_index).unsqueeze(-1)  # (num_edge, 1)
                noise = torch.randn_like(pos) * torch.sqrt(step_size * 2)

                score_d = scorenet.get_score(data, d, sigma)  # (num_edge, 1)
                # batch_time = torch.ones(d.shape[0], device=pos.device) * sigma

                dg_score = get_ratio(dg_model, data, d, sigma, res)
                # print(dg_score)

                score_d = score_d + w_dg * dg_score
                score_pos = self.convert_score_d(score_d, pos, data.edge_index, d)
                score_pos = utils.clip_norm(score_pos, limit=clip)

                pos = pos + step_size * score_pos + noise  # (num_node, 3)
                pos_vecs.append(pos)

        pos_vecs = torch.stack(pos_vecs, dim=0).view(cnt_sigma, n_steps_each, -1, 3)  # (sigams, 100, num_node, 3)
        for i in range(0,len(res)):
            tensor_str = str(res[i])

            # 写入或追加到文本文件
            with open("log_dg_score.txt", 'a' if i > 0 else 'w') as file:
                file.write(f'Iteration {i + 1}:\n')
                file.write(tensor_str + '\n\n')
        return data, pos_vecs


def get_ratio(dg_model, data, d, batch_time, res):
    with torch.enable_grad():

        x_ = d.float().clone().detach().requires_grad_(True)
        # print(x_)
        # y = torch.randn(x_.size(),device=x_.device)

        dg_score = dg_model.judge(data, x_, batch_time).requires_grad_(True)
        # print(min(dg_score), max(dg_score))
        dg_score.requires_grad_(True)
        dg_score = dg_score.unsqueeze(1)
        # res.append(dg_score)
        # if i < 5:

        # print(dg_score.T)
        # print(type(dg_score))
        # print(input)
        # x_ = input.requires_grad_(True)
        prediction = torch.clip(dg_score, 1e-5, 1. - 1e-5)
        log_ratio = torch.log(prediction / (1. - prediction))
        log_ratio.requires_grad_(True)

        dg = torch.autograd.grad(outputs=log_ratio.sum(), inputs=x_, retain_graph=False)[0]
        # print(dg_score[0], dg[0])
        # res = dg.cpu().clone().numpy()
        # close_to_zero_count = np.sum(np.abs(res) < 0.1)
        # 计算比例
        # percentage_close_to_zero = close_to_zero_count / res.size * 100
        # if percentage_close_to_zero != 100.0:
        #     print("c_phi输出：", end=" ")
        #     print(percentage_close_to_zero, close_to_zero_count, res.size)
        # res = dg_score.cpu().detach().numpy()
        # close_to_zero_count = np.sum(np.abs(res) < 0.1)
        # 计算比例
        # percentage_close_to_zero = close_to_zero_count / res.size * 100
        # if percentage_close_to_zero != 100.0:
        #     print("判别器输出：", end=" ")
        #     print(percentage_close_to_zero, close_to_zero_count, res.size)
        # dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])  # (num_edge, 3)
        # score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0)
        # c_phi =

    return dg
