import copy
import json
import logging
import os
import random

import lmdb
import numpy as np
import tensorpack.dataflow as td

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import sys
import pdb
from random import shuffle

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


import json
from typing import Any, Dict, List
import random
import os

import torch
from torch.utils.data import Dataset
import numpy as np

from pytorch_pretrained_bert.tokenization import BertTokenizer
from ._image_features_reader import ImageFeaturesH5ReaderWithObjClasses
import _pickle as cPickle

import sys
import re

def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

# class BiasDataset(Dataset):
#     def __init__(
#         self,
#         captions : list,
#         images : list,
#         category : str,
#         dataset_type : str,
#         fpath_stopwords: str,
#         image_features_reader: ImageFeaturesH5Reader,
#         tokenizer: BertTokenizer,
#         padding_index: int = 0,
#         max_seq_length: int = 20,
#         max_region_num: int = 60
#     ):
#         self.dataset_type = dataset_type
#         self.category = category
#         if self.dataset_type == 'custom':
#             self.imageid2filepath = {}
        
#         self.num_labels = 1
#         self._image_features_reader = image_features_reader
#         self.fpath_stopwords = fpath_stopwords
#         self.tokenizer = tokenizer

#         self._padding_index = padding_index
#         self._max_seq_length = max_seq_length
#         self.entries = self._load_annotations(images, captions)
        
#         self.max_region_num = max_region_num
#         self.tokenize()
#         self.tensorize()

#     def getNumUniqueImages(self):
#         return self._num_unique
        
#     def _load_annotations(self, images, captions):
#         # Build an index which maps image id with a list of caption annotations.
#         entries = []
#         unique_images = set()

#         for image_fp, corresponding_caps in images.items():
#             for c in corresponding_caps:
#                 cap = captions[str(c)]
#                 if self.dataset_type == 'coco': # e.g. COCO, Conceptual Captions
#                     image_id = int(re.sub('^0*', '', image_fp.strip('.jpg|.png').split('_')[-1]))
#                 elif self.dataset_type == 'concap':
#                     image_id = int(image_fp)
#                 else:
#                     image_id = len(entries)
#                     self.imageid2filepath[image_id] = image_fp
                    
#                 entries.append({'caption' : cap,
#                                 'image_id' : image_id})
#                 unique_images.add(image_fp)
#         print(f'{len(unique_images)} unique images for test')
#         self._num_unique = len(unique_images)
#         return entries

#     def tokenize(self):
#         """Tokenizes the captions.

#         This will add caption_tokens in each entry of the dataset.
#         -1 represents nil, and should be treated as padding_idx in embedding.
#         """
#         for entry in self.entries:
            
#             sentence_tokens = self.tokenizer.tokenize(entry["caption"])
#             sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]

#             tokens = [
#                 self.tokenizer.vocab.get(w, self.tokenizer.vocab["[UNK]"])
#                 for w in sentence_tokens
#             ]

#             tokens = tokens[:self._max_seq_length]
#             segment_ids = [0] * len(tokens)
#             input_mask = [1] * len(tokens)

#             if len(tokens) < self._max_seq_length:
#                 # Note here we pad in front of the sentence
#                 padding = [self._padding_index] * (self._max_seq_length - len(tokens))
#                 tokens = tokens + padding
#                 input_mask += padding
#                 segment_ids += padding

#             assert_eq(len(tokens), self._max_seq_length)
#             entry["token"] = tokens
#             entry["input_mask"] = input_mask
#             entry["segment_ids"] = segment_ids

#     def tensorize(self):

        
#         for entry in self.entries:
#             token = torch.from_numpy(np.array(entry["token"]))
#             entry["token"] = token

#             input_mask = torch.from_numpy(np.array(entry["input_mask"]))
#             entry["input_mask"] = input_mask

#             segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
#             entry["segment_ids"] = segment_ids


#     def __getitem__(self, index):
#         entry = self.entries[index]
#         image_id = entry["image_id"]

#         #ref_box = entry["refBox"]

#         #ref_box = [ref_box[0], ref_box[1], ref_box[0]+ref_box[2], ref_box[1]+ref_box[3]]
#         if not self.dataset_type == 'custom':
#             features, num_boxes, boxes, boxes_ori = self._image_features_reader[image_id]
#         else:
#             image_fp = self.imageid2filepath[image_id]
#             features, num_boxes, boxes, boxes_ori = self._image_features_reader[image_fp]

#         boxes_ori = boxes_ori[:num_boxes]
#         boxes = boxes[:num_boxes]
#         features = features[:num_boxes]

#         # if self.split == 'train':
#         #     gt_features, gt_num_boxes, gt_boxes, gt_boxes_ori = self._gt_image_features_reader[image_id]

#         #     # merge two boxes, and assign the labels. 
#         #     gt_boxes_ori = gt_boxes_ori[1:gt_num_boxes]
#         #     gt_boxes = gt_boxes[1:gt_num_boxes]
#         #     gt_features = gt_features[1:gt_num_boxes]

#         #     # concatenate the boxes
#         #     mix_boxes_ori = np.concatenate((boxes_ori, gt_boxes_ori), axis=0)
#         #     mix_boxes = np.concatenate((boxes, gt_boxes), axis=0)
#         #     mix_features = np.concatenate((features, gt_features), axis=0)
#         #     mix_num_boxes = min(int(num_boxes + int(gt_num_boxes) - 1), self.max_region_num)
#         #     # given the mix boxes, and ref_box, calculate the overlap. 
#         #     mix_target = iou(torch.tensor(mix_boxes_ori[:,:4]).float(), torch.tensor([ref_box]).float())
#         #     mix_target[mix_target<0.5] = 0

#         # else:
#         #     mix_boxes_ori = boxes_ori
#         #     mix_boxes = boxes
#         #     mix_features = features
#         #     mix_num_boxes = min(int(num_boxes), self.max_region_num)
#         #     mix_target = iou(torch.tensor(mix_boxes_ori[:,:4]).float(), torch.tensor([ref_box]).float())

#         mix_boxes_ori = boxes_ori
#         mix_boxes = boxes
#         mix_features = features
#         mix_num_boxes = min(int(num_boxes), self.max_region_num)
#         #mix_target = iou(torch.tensor(mix_boxes_ori[:,:4]).float(), torch.tensor([ref_box]).float())
        
#         image_mask = [1] * (mix_num_boxes)
#         while len(image_mask) < self.max_region_num:
#             image_mask.append(0)

#         mix_boxes_pad = np.zeros((self.max_region_num, 5))
#         mix_features_pad = np.zeros((self.max_region_num, 2048))

#         mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
#         mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]

#         # appending the target feature.
#         features = torch.tensor(mix_features_pad).float()
#         image_mask = torch.tensor(image_mask).long()
#         spatials = torch.tensor(mix_boxes_pad).float()

#         #target = torch.zeros((self.max_region_num,1)).float()
#         #target[:mix_num_boxes] = mix_target

#         spatials_ori = torch.tensor(mix_boxes_ori).float()
#         co_attention_mask = torch.zeros((self.max_region_num, self._max_seq_length))

#         caption = entry["token"]
#         input_mask = entry["input_mask"]
#         segment_ids = entry["segment_ids"]
#         return features, spatials, image_mask, caption, input_mask, segment_ids, co_attention_mask, image_id
#         #return features, spatials, image_mask, caption, target, input_mask, segment_ids, co_attention_mask, image_id

#     def __len__(self):
#         return len(self.entries)




class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
        self, image_feat=None, image_target=None, caption=None, lm_labels=None, image_loc=None,
            num_boxes=None, cls_indices=None
    ):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.image_feat = image_feat
        self.caption = caption
        self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        self.image_target = image_target
        self.num_boxes = num_boxes
        self.cls_indices = cls_indices

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids=None,
        input_mask=None,
        segment_ids=None,
        lm_label_ids=None,
        image_feat=None,
        image_loc=None,
        image_label=None,
        image_mask=None,
        coattention_mask=None,
        masked_image_feat=None,
        masked_image_label=None
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_mask = image_mask
        self.coattention_mask = coattention_mask
        self.masked_image_feat = masked_image_feat
        self.masked_image_label = masked_image_label
        
class BiasLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(
            self,
            bert_model_name,
            captions,
            images,
            dataset_type,
            category,
            image_features_fp,
            obj_list,
            seq_len,
            encoding="utf-8",
            hard_negative=False,
            batch_size=512,
            shuffle=False,
            num_workers=25,
            cache=50000,
            drop_last=False,
            cuda=False,
            distributed=False,
            visualization=False,
    ):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
        self.dataset_type = dataset_type
        self.category = category
        self.imageid2filepath = {}

        entries = self._load_entries(images, captions, dataset_type)
        image_feature_reader = ImageFeaturesH5ReaderWithObjClasses(image_features_fp)
        
        preprocess_function = BertPreprocessBatch(
            captions=entries,
            tokenizer=self.tokenizer,
            obj_list=obj_list,
            seq_len=seq_len,
            region_len=36,
            data_size=len(entries),
            image_feature_reader=image_feature_reader,
            imageid2filepath=self.imageid2filepath,
            encoding="utf-8",
            predict_feature=False,
        )

        batch_size = min(len(entries), batch_size)
        ds = td.MapData(entries, preprocess_function)
        self.ds = td.BatchData(ds, batch_size)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def _load_entries(self, images, captions, dataset_type):
        entries = []
        #unique_images = set()

        for image_fp, corresponding_caps in images.items():
            for c in corresponding_caps:
                cap = captions[str(c)]
                if dataset_type == 'coco': # e.g. COCO, Conceptual Captions
                    image_id = int(re.sub('^0*', '', image_fp.strip('.jpg|.png').split('_')[-1]))
                    self.imageid2filepath[image_id] = image_fp
                elif dataset_type == 'concap':
                    image_id = int(image_fp)
                    self.imageid2filepath[image_id] = image_fp
                else:
                    image_id = len(entries)
                    self.imageid2filepath[image_id] = image_fp
                entries.append({'caption' : cap,
                                'image_id' : image_id})
                #unique_images.add(image_fp)
        return entries

        
    def __iter__(self):
        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, lm_label_ids, image_feat, \
                image_loc, image_label, image_mask, image_id, coattention_mask,\
                masked_image_feat, masked_image_label = batch

            batch_size = input_ids.shape[0]
            g_image_feat = np.sum(image_feat, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(np.array([[0,0,1,1,1]], dtype=np.float32), batch_size, axis=0)
            image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)
            
            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            masked_g_image_feat = np.sum(masked_image_feat, axis=1) / \
                                    np.sum(image_mask, axis=1, keepdims=True)
            masked_image_feat = np.concatenate([np.expand_dims(masked_g_image_feat, axis=1),
                                                masked_image_feat], axis=1)
            masked_image_feat = np.array(masked_image_feat, dtype=np.float32)

            batch = (input_ids, input_mask, segment_ids, lm_label_ids, image_feat, \
                     image_loc, image_label, image_mask, image_id, coattention_mask,
                     masked_image_feat, masked_image_label)

            yield tuple(torch.tensor(data) for data in batch)

    def __len__(self):
        return self.ds.size()


class BertPreprocessBatch(object):
    def __init__(
        self,
        captions,
        tokenizer,
        obj_list,
        seq_len,
        region_len, 
        data_size,
        image_feature_reader,
        imageid2filepath,
        encoding="utf-8",
        predict_feature=False,
        visualization=False
    ):
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.obj_list = obj_list
        self.image_feature_reader = image_feature_reader
        self.imageid2filepath = imageid2filepath
        self.predict_feature = predict_feature
        self.num_caps = data_size
        self.captions = captions
        self.visualization = visualization

    def __call__(self, data):
        caption = data['caption']
        image_id = data['image_id']
        
        #if not hasattr(self, 'imageid2filepath'):
        image_fp = self.imageid2filepath[image_id]
        image_feature, num_boxes, image_location, image_location_ori, cls_indices =\
                                    self.image_feature_reader[image_fp]

        # else:
        #     image_fp = self.imageid2filepath[image_id]
        #     image_feature, num_boxes, image_location, image_location_ori, cls_indices =\
        #                             self.image_feature_reader[image_fp]

        num_boxes = min(self.region_len, num_boxes) # TODO
        image_feature = image_feature[:self.region_len]
        image_location = image_location[:self.region_len]
            
        tokens_caption = self.tokenizer.tokenize(caption)
        cur_example = InputExample(
            image_feat=image_feature,
            caption=tokens_caption,
            image_loc=image_location,
            num_boxes=num_boxes,
            cls_indices=cls_indices)
        
        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer,
                                                        self.obj_list, self.region_len)
        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_label,
            cur_features.image_mask,
            image_id,
            cur_features.coattention_mask,
            cur_features.masked_image_feat,
            cur_features.masked_image_label
        )
        return cur_tensors
        
    def convert_example_to_features(self, example, max_seq_length, tokenizer, obj_list, max_region_length):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
        image_feat = example.image_feat
        caption = example.caption
        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)
        cls_indices = example.cls_indices
        self._truncate_seq_pair(caption, max_seq_length - 2)
        caption_label = self.label_caption(caption, tokenizer)
        image_label = [-1] * len(image_feat)
        masked_image_feat, masked_image_label =\
                self.mask_region(image_feat, num_boxes, cls_indices, obj_list)

        # concatenate lm labels and account for CLS, SEP, SEP
        # lm_label_ids = ([-1] + caption_label + [-1] + image_label + [-1])
        lm_label_ids = [-1] + caption_label + [-1]
        # image_label = ([-1] + image_label)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        # for i in range(36):
        #     # tokens.append(0)
        #     segment_ids.append(0)

        # tokens.append("[SEP]")
        # segment_ids.append(0)
        for token in caption:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length


        coattention_mask = np.zeros((max_region_length, max_seq_length))
        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            lm_label_ids=np.array(lm_label_ids),
            image_feat=image_feat,
            image_loc=image_loc,
            image_label=np.array(image_label),
            image_mask = np.array(image_mask),
            coattention_mask=coattention_mask,
            masked_image_feat=masked_image_feat,
            masked_image_label=np.array(masked_image_label)
        )
        return features

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()

    def label_caption(self, tokens, tokenizer):
        output_label = []
        for token in tokens:
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning(
                    "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                )
        return output_label

    def mask_region(self, image_feat, num_boxes, cls_indices, obj_list):        
        """
        """
        output_label = []
        for i in range(num_boxes):
            cls_idx = cls_indices[i]
            cls = obj_list[cls_idx]
            
            if cls == 'man' or cls == 'woman' or cls == 'person':
                i1 = image_feat[i].sum().item()
                image_feat[i] = 0
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return image_feat, output_label

    
# class BertPreprocessRetrieval(object):
#     def __init__(
#         self,
#         caption_path,
#         tokenizer,
#         seq_len,
#         region_len, 
#         data_size,
#         encoding="utf-8",
#         predict_feature=False,
#     ):

#         self.seq_len = seq_len
#         self.region_len = region_len
#         self.tokenizer = tokenizer
#         self.predict_feature = predict_feature
#         self.num_caps = data_size
#         self.captions = list(json.load(open(caption_path, 'r')).values())[:data_size]

#     def __call__(self, data):
#         raise Exception('not yet implemented')
#         image_feature_wp, image_target_wp, image_location_wp, num_boxes,  image_h, image_w, image_id, caption = data
        
#         image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
#         image_target = np.zeros((self.region_len, 1601), dtype=np.float32)
#         image_location = np.zeros((self.region_len, 5), dtype=np.float32)

#         num_boxes = int(num_boxes)
#         image_feature[:num_boxes] = image_feature_wp
#         image_target[:num_boxes] = image_target_wp
#         image_location[:num_boxes,:4] = image_location_wp

#         image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))
        
#         image_location[:,0] = image_location[:,0] / float(image_w)
#         image_location[:,1] = image_location[:,1] / float(image_h)
#         image_location[:,2] = image_location[:,2] / float(image_w)
#         image_location[:,3] = image_location[:,3] / float(image_h)

#         label = 0
        
#         tokens_caption = self.tokenizer.tokenize(caption)
#         cur_example = InputExample(
#             image_feat=image_feature,
#             image_target=image_target,
#             caption=tokens_caption,
#             is_next=label,
#             image_loc=image_location,
#             num_boxes=num_boxes
#         )

#         # transform sample to features
#         cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.region_len)
        
#         cur_tensors = (
#             cur_features.input_ids,
#             cur_features.input_mask,
#             cur_features.segment_ids,
#             cur_features.is_next,
#             cur_features.image_feat,
#             cur_features.image_loc,
#             cur_features.image_mask,
#             float(image_id),
#             caption,
#         )
#         return cur_tensors


#     def convert_example_to_features(self, example, max_seq_length, tokenizer, max_region_length):
#         """
#         Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
#         IDs, LM labels, input_mask, CLS and SEP tokens etc.
#         :param example: InputExample, containing sentence input as strings and is_next label
#         :param max_seq_length: int, maximum length of sequence.
#         :param tokenizer: Tokenizer
#         :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
#         """
#         image_feat = example.image_feat
#         caption = example.caption
#         image_loc = example.image_loc
#         # image_target = example.image_target
#         num_boxes = int(example.num_boxes)
#         self._truncate_seq_pair(caption, max_seq_length - 2)
#         # caption, caption_label = self.random_word(caption, tokenizer)
#         caption_label = None
#         # image_feat, image_loc, image_label = self.random_region(image_feat, image_loc, num_boxes)
#         image_label = None

#         tokens = []
#         segment_ids = []

#         tokens.append("[CLS]")
#         segment_ids.append(0)

#         for token in caption:
#             tokens.append(token)
#             segment_ids.append(0)
#         tokens.append("[SEP]")
#         segment_ids.append(0)

#         input_ids = tokenizer.convert_tokens_to_ids(tokens)

#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.
#         # input_ids = input_ids[:1] input_ids[1:]
#         input_mask = [1] * (len(input_ids))
#         image_mask = [1] * (num_boxes)
#         # Zero-pad up to the visual sequence length.
#         while len(image_mask) < max_region_length:
#             image_mask.append(0)

#         # Zero-pad up to the sequence length.
#         while len(input_ids) < max_seq_length:
#             input_ids.append(0)
#             input_mask.append(0)
#             segment_ids.append(0)

#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length
#         assert len(image_mask) == max_region_length

#         features = InputFeatures(
#             input_ids=np.array(input_ids),
#             input_mask=np.array(input_mask),
#             segment_ids=np.array(segment_ids),
#             is_next=np.array(example.is_next),
#             image_feat=image_feat,
#             image_loc=image_loc,
#             image_mask = np.array(image_mask),
#         )
#         return features

#     def _truncate_seq_pair(self, tokens_b, max_length):
#         """Truncates a sequence pair in place to the maximum length."""

#         # This is a simple heuristic which will always truncate the longer sequence
#         # one token at a time. This makes more sense than truncating an equal percent
#         # of tokens from each, since if one sequence is very short then each token
#         # that's truncated likely contains more information than a longer sequence.
#         while True:
#             total_length = len(tokens_b)
#             if total_length <= max_length:
#                 break

#             tokens_b.pop()
