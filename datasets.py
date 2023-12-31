import os
import pandas as pd
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        print("base dataset")
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        print("before opening")
        self.ann = json.load(open(self.ann_path))
        print("after opening")  
        self.examples = self.ann[self.split]
        self.masks = []
        self.reports = []

        for each in self.examples.keys():
            #self.reports.append(self.examples[each]['En_Report'][:self.max_seq_length])
            #self.masks.append([1]*len(self.reports[-1]))
             tokenized_report = tokenizer(self.examples[each]['En_Report'])[:self.max_seq_length]
             self.reports.append(tokenized_report)
             self.masks.append([1]*len(tokenized_report))
             
    def __len__(self):
        return len(self.examples)

class FFAIRDataset(BaseDataset):
    def __getitem__(self, idx):
        print("function called FFAIRDataset")
        case_ids = list(self.examples.keys())
        #case_ids = self.examples[idx]
        print("got keys")
        case_id = case_ids[idx]
        image_id = case_id
        image_path = self.examples[case_id]['Image_path'] 
        images = []
        for ind in range(len(image_path)):
            image = Image.open(os.path.join(self.image_dir, image_path[ind])).convert('RGB')
            if self.transform is not None:
                images.append(self.transform(image))
        print("got images")
        images = torch.stack(images, 0)
        report_ids = self.reports[idx]
        report_masks = self.masks[idx]
        seq_length = len(report_ids)
        sample = (image_id, images, report_ids, report_masks, seq_length)
        return sample


