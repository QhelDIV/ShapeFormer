# *** THIS FILE IS FROM DFAUST DATSET ***
# -*- coding: utf-8 -*-
# Script to write registrations as obj files
# Copyright (c) [2015] [Gerard Pons-Moll]

from argparse import ArgumentParser
from os import mkdir
from os.path import join, exists
import h5py
import sys


def write_mesh_as_obj(fname, verts, faces):
    with open(fname, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


if __name__ == '__main__':

    # Subject ids
    sids = ['50002', '50004', '50007', '50009', '50020',
            '50021', '50022', '50025', '50026', '50027']
    # Sequences available for each subject id are listed in scripts/subjects_and_sequences.txt

    parser = ArgumentParser(description='Save sequence registrations as obj')
    parser.add_argument('--path', type=str, default='../registrations_f.hdf5',
                        help='dataset path in hdf5 format')
    parser.add_argument('--seq', type=str, default='jiggle_on_toes',
                        help='sequence name')                         
    parser.add_argument('--sid', type=str, default='50004', choices=sids,
                        help='subject id')
    parser.add_argument('--tdir', type=str, default='./',
                        help='target directory')
    args = parser.parse_args()

    sidseq = args.sid + '_' + args.seq
    with h5py.File(args.path, 'r') as f:
        if sidseq not in f:
            print('Sequence %s from subject %s not in %s' % (args.seq, args.sid, args.path))
            f.close()
            sys.exit(1)
        verts = f[sidseq][...].transpose([2, 0, 1])
        faces = f['faces'][...]

    tdir = join(args.tdir, sidseq)
    if not exists(tdir):
        mkdir(tdir)

    # Write to an obj file
    for iv, v in enumerate(verts):
        fname = join(tdir, '%05d.obj' % iv)
        print('Saving mesh %s' % fname)
        write_mesh_as_obj(fname, v, faces)
