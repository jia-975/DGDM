import copy
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import PyMol
from rdkit import Chem
from rdkit.Geometry import rdGeometry


def updateConf(mol, pos):
    new_mol = copy.deepcopy(mol)

    conf = new_mol.GetConformer(0)
    for i, item in enumerate(pos):
        p = rdGeometry.Point3D(item[0].item(), item[1].item(), item[2].item())
        conf.SetAtomPosition(i, p)  # 设置第i个原子的坐标
    return mol


if __name__ == '__main__':
    v = PyMol.MolViewer()
    show_num = 5
    gen_conf = 3
    with open("samples_0.pkl", "rb") as fin:
        data = pickle.load(fin)

    data = data[:show_num]
    mols = []
    for i, mol in enumerate(data):
        rdmol = mol.rdmol
        print('有%d个构象'%rdmol.GetNumConformers())
        # print(rdmol)
        pos = mol.pos_gen
        mols.append(rdmol)
        # atom_num = mol.num_nodes
        # print(type)
        atom_num = rdmol.GetNumAtoms()
        v.ShowMol(rdmol, name="mol_%d" % i)
        pos_list = np.array_split(pos, len(pos) // atom_num)
        for j, p in enumerate(pos_list):
            gen_mol = updateConf(rdmol, p)
            mols.append(gen_mol)
            v.ShowMol(gen_mol, name="mol_%d_gen_%d" % (i, j))

    with open("conf_vis/qm9", "wb") as f:
        pickle.dump(mols, f)
