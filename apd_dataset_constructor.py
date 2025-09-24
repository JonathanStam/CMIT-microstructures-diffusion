import PyAPD
import torch
from pathlib import Path

import sys
import os
import torch.nn.functional as F

# Define a null terminal to take the prints we are not interested in.
class NullIO():
    def write(self, txt):
       pass
NullTerminal = NullIO()


def build_apd_dataset(sampler, sample_size: int, folder_name=None, status_interval=250) -> torch.Tensor:
    """
    Construct a dataset of images sampled from apd_system objects produced by sampler:
     -  sampler(seed=-1): function which produces apds. This is the object which contains the selected distribution of APDs in addition to the specified pixel parameters which
        determine the size of the images produced.
     -  sample_size: how many images are sampled
     -  folder_name=None: should be either string or NoneType. If string, saves the images to directory with name folder_name, otherwise the function simply returns a batch of tensors.
     -  status_interval: the number of samples after which the method prints the current progress to the console.
    """


    # Check if folder exists and, if not, create it.
    if not(folder_name == None):
        assert type(folder_name) == str
        path = Path(folder_name)
        if path.is_dir():
            print("Directory already exists")
        else:
            print("Creating directory")
            path.mkdir(parents=True, exist_ok=True)

    sample = []
    for i in range(sample_size):
        apd = sampler()
        img_tens = apd.assemble_apd().reshape(apd.pixel_params)

        # Send values to [0,1]
        img_tens = img_tens/img_tens.max()

        # Then to [-1,1]
        img_tens = 2 * (img_tens - 0.5)
        if not( folder_name == None):
            torch.save(img_tens, f'{folder_name}/{i}.pt')
        else:
            sample.append(img_tens)


        # Status update:
        if (i + 1) % status_interval == 0:
            print(f"Samples constructed: {i+1} / {sample_size}")

    if folder_name == None:
        return torch.stack(sample)
    else:
        pass

def basic_sampler(seed=-1):
    """
    Sampler based on the example used in lectures.
    """
    # The PyAPD calls print a lot, so when sampling many times we don't want all of this to appear in our terminal.
    sys.stdout = NullTerminal

    # Sample as in the lecture examples.
    apd1 = PyAPD.apd_system(N=15, ani_thres=0.5, seed=seed)
    apd1.check_optimality()
    apd1.Lloyds_algorithm(K=10, verbosity_level = 0 )


    sys.stdout = sys.__stdout__ # reset old stdout, so terminal still works.

    return apd1


def realistic_sampler(seed=-1):
    """
    Sampler calibrated to the parameters suggested in the question. 
    100x100 pixels; 
    ani_thres ~ Uniform(0.05, 0.6);
    N ~ Uniform{5, ..., 35}
    log(target_masses) ~ N(mu, ∑), target masses sum to one.
    """
    ani_thres = 0.05 + 0.55 * torch.rand((1,))
    N = torch.randint(low=5, high=36, size=(1,))

    target_masses = torch.exp(torch.randn((N,))).squeeze()
    target_masses /= target_masses.sum()


    apd = PyAPD.apd_system(N=int(N), ani_thres=int(ani_thres), target_masses=target_masses, pixel_params=(100,100))

    
    sys.stdout = NullTerminal
    apd.check_optimality()
    apd.Lloyds_algorithm(K = 10, verbosity_level=0)
    sys.stdout = sys.__stdout__

    return apd


#build_apd_dataset(
#    sampler = realistic_sampler,
#    sample_size=30000,
#    folder_name='test_dataset'
#)

class APDDataset(torch.utils.data.Dataset):
    def __init__(self, root, size=None, device='cpu'):
        super().__init__()
        self.device = device
        self.size = size
        self.root = root
        self.tensor_paths = [t for t in os.listdir(root) if t.endswith('.pt')]

    def __len__(self):
        return len(self.tensor_paths)
    
    def __getitem__(self, idx):
        tensor = torch.load(f"{self.root}/{self.tensor_paths[idx]}", weights_only=True, map_location=torch.device(self.device)).to(self.device)
        if self.size == None:
            return tensor
        else:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            tensor = F.interpolate(tensor, size=self.size)
            tensor = tensor.squeeze(0).squeeze(0)
            return tensor

##dataset = APDDataset(root="test_dataset")

#print(dataset.__getitem__(0))