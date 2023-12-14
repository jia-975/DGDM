def sde_generate_samples_demo(self, mol, start, end, num_repeat=None, out_path=None,
                                      file_name='sample_from_testset'):


    return_data = copy.deepcopy(mol)
    num_repeat_ = 1
    batch = utils.repeat_data(mol, num_repeat_).to(self.device)

    batch, pos_traj, d_gen, d_recover = self.sde_generator(batch, self._score_model, self._disc_model,
                                                           self.config.test.gen.num_euler_steps,
                                                           self.config.test.gen.num_langevin_steps)

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
    print('pos generation[%d-%d] done  |  Time: %.5f' % (start, end, time() - generate_start))

    return all_data_list, all_traj
