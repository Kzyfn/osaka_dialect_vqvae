from models import BinaryFileSource
from glob import glob
from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from os.path import join
import numpy as np

def create_loader(valid=True, batch_size=1):
    DATA_ROOT = "data"
    X = {"acoustic": {}}
    Y = {"acoustic": {}}
    utt_lengths = {"acoustic": {}}
    for ty in ["acoustic"]:
        for phase in ["train", "test"]:
            train = phase == "train"
            x_dim = 442
        
            X[ty][phase] = FileSourceDataset(
                BinaryFileSource(
                    join(DATA_ROOT, "X_{}".format(ty)), dim=x_dim, train=train, valid=valid
                )
            )
            utt_lengths[ty][phase] = np.array(
                [len(x) for x in X[ty][phase]], dtype=np.int
            )

            np.savetxt('data/shapes_{}.csv'.format(phase), utt_lengths[ty][phase])


create_loader()
