from typing import Dict, List
import torch
import numpy as np
import time

def collate_fn_epic_kitchen(batch: List) -> Dict:
    """ EPIC-KITCHEN collate function used for dataloader.
    """
    img_height = batch[0][1]
    img_width = batch[0][2]
    class_names = batch[0][3]
    img_stack = np.stack([b[0] for b in batch], axis=0)
    img_stack = torch.as_tensor(img_stack.astype("float32").transpose(0, 3, 1, 2))

    batch_res = [{"image": img_stack[i], "height": img_height, "width": img_width, "class_names": class_names} 
                 for i in range(img_stack.shape[0])]
    return batch_res

def collate_fn_general(batch: List) -> Dict:
    """ General collate function used for dataloader.
    """
    return batch
