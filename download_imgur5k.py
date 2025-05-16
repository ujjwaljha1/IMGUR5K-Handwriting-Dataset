'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
IMGUR5K is shared as a set of image urls with annotations. This code downloads
the images and verifies the hash to the image to avoid data contamination.

Usage:
      python download_imgur5k.py --dataset_info_dir <dir_with_annotation_and_hashes> --output_dir <path_to_store_images>

Output:
     Images downloaded to output_dir
     imgur5k_annotations.json : json file with image annotation mappings -> downloaded to dataset_info_dir
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import argparse
import hashlib
import json
import numpy as np
import os
import requests

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Processing imgur5K dataset download...")
    parser.add_argument(
        "--dataset_info_dir",
        type=str,
        default="dataset_info",
        required=False,
        help="Directory with dataset information",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="images",
        required=False,
        help="Directory path to download the image",
    )
    args = parser.parse_args()
    return args


# Image hash computed for image using md5..
def compute_image_hash(img_path):
    with open(img_path, 'rb') as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()

# Create a sub json based on split idx
def _create_split_json(anno_json, _split_idx):

    split_json = {}

    split_json['index_id'] = {}
    split_json['index_to_ann_map'] = {}
    split_json['ann_id'] = {}

    for _idx in _split_idx:
        # Check if the idx is not bad
        if _idx not in anno_json['index_id']:
            continue

        split_json['index_id'][_idx] = anno_json['index_id'][_idx]
        split_json['index_to_ann_map'][_idx] = anno_json['index_to_ann_map'][_idx]

        for ann_id in split_json['index_to_ann_map'][_idx]:
            split_json['ann_id'][ann_id] = anno_json['ann_id'][ann_id]

    return split_json

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # User-Agent header to mimic browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/113 Safari/537.36"
    }

    # Create a hash dictionary with image index and its corresponding gt hash
    with open(f"{args.dataset_info_dir}/imgur5k_hashes.lst", "r", encoding="utf-8") as _H:
        hashes = _H.readlines()
        hash_dict = {}

        for line in hashes:
            parts = line.strip().split()
            if len(parts) == 2:
                idx, hsh = parts
                hash_dict[idx] = hsh

    tot_evals = 0
    num_match = 0
    invalid_urls = []

    # Download the urls and save only the ones with valid hash to ensure underlying image has not changed
    for index in list(hash_dict.keys()):
        urls_to_try = [
            f'https://i.imgur.com/{index}.jpeg',
            f'https://i.imgur.com/{index}.jpg',
        ]

        img_data = None
        for image_url in urls_to_try:
            try:
                response = requests.get(image_url, headers=headers, timeout=10)
                if response.status_code == 200 and len(response.content) > 100:
                    img_data = response.content
                    break
                else:
                    print(f"Failed to download {image_url}: Status {response.status_code}")
            except requests.RequestException as e:
                print(f"Exception downloading {image_url}: {e}")

        if img_data is None:
            print(f"URL retrieval for {index} failed!!\n")
            invalid_urls.append(index)
            continue

        # Save the image
        image_path = f'{args.output_dir}/{index}.jpg'
        with open(image_path, 'wb') as handler:
            handler.write(img_data)

        # Compute hash once
        cur_hash = compute_image_hash(image_path)
        tot_evals += 1

        if hash_dict[index] != cur_hash:
            print(f"Hash mismatch for IMG {index} (ref: {hash_dict[index]} != cur: {cur_hash})")
            os.remove(image_path)
            invalid_urls.append(index)
            continue
        else:
            num_match += 1
            print(f"Downloaded and verified {index}")

    # Generate the final annotations file
    # Format: { "index_id" : {indexes}, "index_to_annotation_map" : { annotations ids for an index}, "annotation_id": { each annotation's info } }
    # Bounding boxes with '.' mean the annotations were not done for various reasons

    _F = np.loadtxt(f'{args.dataset_info_dir}/imgur5k_data.lst', delimiter="\t", dtype=str, encoding="utf-8")
    anno_json = {}

    anno_json['index_id'] = {}
    anno_json['index_to_ann_map'] = {}
    anno_json['ann_id'] = {}

    cur_index = ''
    for cnt, image_url in enumerate(_F[:,0]):
        # image_url format example: https://i.imgur.com/YsaVkzl.jpg
        index = image_url.split('/')[-1][:-4]  # remove extension

        if index in invalid_urls:
            continue

        if index != cur_index:
            anno_json['index_id'][index] = {
                'image_url': image_url,
                'image_path': f'{args.output_dir}/{index}.jpg',
                'image_hash': hash_dict[index]
            }
            anno_json['index_to_ann_map'][index] = []

        ann_id = f"{index}_{len(anno_json['index_to_ann_map'][index])}"
        anno_json['index_to_ann_map'][index].append(ann_id)
        anno_json['ann_id'][ann_id] = {'word': _F[cnt,2], 'bounding_box': _F[cnt,1]}

        cur_index = index

    with open(f'{args.dataset_info_dir}/imgur5k_annotations.json', 'w') as f:
        json.dump(anno_json, f, indent=4)

    # Now split the annotations json in train, validation and test jsons
    splits = ['train', 'val', 'test']
    for split in splits:
        split_file = f'{args.dataset_info_dir}/{split}_index_ids.lst'
        if not os.path.exists(split_file):
            print(f"Warning: Split file {split_file} not found, skipping split {split}")
            continue
        _split_idx = np.loadtxt(split_file, delimiter="\n", dtype=str)
        split_json = _create_split_json(anno_json, _split_idx)
        with open(f'{args.dataset_info_dir}/imgur5k_annotations_{split}.json', 'w') as f:
            json.dump(split_json, f, indent=4)

    print(f"Downloaded and verified images: {num_match}/{tot_evals}")

if __name__ == '__main__':
    main()
