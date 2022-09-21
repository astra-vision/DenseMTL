import datetime, logging, os

from matplotlib import pyplot as plt
import numpy as np

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames
                if filename.endswith(suffix)]


def colorize(img, cmap='plasma', mask_zero=False, max_percentile=100): #? refactor to torch
    img = img.detach().cpu().numpy()
    # img = torch.log(img.to(torch.float32) + 1).detach().cpu().numpy()
    vmin = np.min(img)
    if max_percentile == 100:
        vmax = np.max(img)
    else:
        vmax = np.percentile(img, max_percentile)
    mask = (img <= 0).squeeze()
    cm = plt.get_cmap(cmap)
    colored_image = cm(np.clip(img.squeeze(), vmin, vmax) / vmax)[:, :, :3]
    # Use white if no depth is available (<= 0)
    if mask_zero:
        colored_image[mask, :] = [1, 1, 1]
    return colored_image

def ref_items(d, keys):
    return {k: d[k] for k in keys}

def now():
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_checkpoint(base_dir):
    """Looks for last experiment checkpoint in the same directory."""
    assert os.path.isdir(base_dir), f'no previous experiments found in {base_dir}'
    prev_exps = glob.glob(os.path.join(base_dir, '**', 'resume.pkl'))
    resume_path = prev_exps[-1]
    assert os.path.isfile(resume_path), f'Could not find a checkpoint for exp {base_dir}'