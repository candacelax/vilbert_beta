import pandas as pd
import numpy as np
import requests
import zlib
import os
import shelve
import magic #pip install python-magic
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import re

# python download_data.py --train --train_path /storage/ccross/vilbert_beta/data/conceptual-captions/Train_GCC-training.tsv --format_for_bash --train

headers = {
    #'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    'User-Agent':'Googlebot-Image/1.0', # Pretend to be googlebot
    'X-Forwarded-For': '64.18.15.200'
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format_for_bash', action='store_true', help='just write list of files')
    parser.add_argument('--train', action='store_true', help='Download training data')
    parser.add_argument('--val', dest='validation', action='store_true', help='Download validation data')
    parser.add_argument('--train_path', type=str, help='path to training TSV file')
    parser.add_argument('--val_path', type=str, help='path to validation TSV file')
    parser.add_argument('--num', dest='num_processes', default=32, type=int,
                        help='number of processes in the pool can be larger than cores')
    parser.add_argument('--images_per_part', default=100, type=int,
                        help='chunk_size is how many images per chunk per process'
                        'changing this resets progress when restarting.')
    return parser.parse_args()

def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    return (split_ind, r)

def df_multiprocess(df, processes, chunk_size, func, dataset_name):
    print("Generating parts...")
    with shelve.open('%s_%s_%s_results.tmp' % (dataset_name, func.__name__, chunk_size)) as results:
 
        pbar = tqdm(total=len(df), position=0)
        # Resume:
        finished_chunks = set([int(k) for k in results.keys()])
        pbar.desc = "Resuming"
        for k in results.keys():
            pbar.update(len(results[str(k)][1]))

        pool_data = ((index, df[i:i + chunk_size], func) for index, i in enumerate(range(0, len(df), chunk_size)) if index not in finished_chunks)
        print(int(len(df) / chunk_size), "parts.", chunk_size, "per part.", "Using", processes, "processes")
 
        pbar.desc = "Downloading"
        with Pool(processes) as pool:
            for i, result in enumerate(pool.imap_unordered(_df_split_apply, pool_data, 2)):
                results[str(result[0])] = result
                pbar.update(len(result[1]))
        pbar.close()

    print("Finished Downloading.")
    return

# Unique name based on url
def _file_name(row):
    return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

# For checking mimetypes separately without download
def check_mimetype(row):
    if os.path.isfile(str(row['file'])):
        row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
    return row

# Don't download image, just check with a HEAD request, can't resume.
# Can use this instead of download_image to get HTTP status codes.
def check_download(row):
    fname = _file_name(row)
    try:
        # not all sites will support HEAD
        response = requests.head(row['url'], stream=False, timeout=5, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        row['headers'] = dict(response.headers)
    except:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row
    if response.ok:
        row['file'] = fname

    return row

def download_image(row):
    fname = _file_name(row)
    # Skip Already downloaded, retry others later

    if os.path.isfile(fname):
        row['status'] = 200
        row['file'] = fname
        row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
        return row

    try:
        # use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(row['url'], stream=False, timeout=10, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        #row['headers'] = dict(response.headers)
    except Exception as e:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row

    if response.ok:
        try:
            with open(fname, 'wb') as out_file:
                # some sites respond with gzip transport encoding
                response.raw.decode_content = True
                out_file.write(response.content)
            row['mimetype'] = magic.from_file(row['file'], mime=True)
            row['size'] = os.stat(row['file']).st_size
        except:
            # This is if it times out during a download or decode
            row['status'] = 408
            return row
        row['file'] = fname
    return row

def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption","url"], usecols=range(1,2))
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df

def df_from_shelve(chunk_size, func, dataset_name):
    print("Generating Dataframe from results...")
    with shelve.open('%s_%s_%s_results.tmp' % (dataset_name, func.__name__, chunk_size)) as results:
        keylist = sorted([int(k) for k in results.keys()])
        df = pd.concat([results[str(k)][1] for k in keylist], sort=True)
    return df


if __name__ == '__main__':
    args = parse_args()

    if args.format_for_bash:
        splits = []
        os.makedirs('download-scripts', exist_ok=True)
        
        if args.validation:
            assert args.val_path, 'Path to validation file required!'
            splits.append( (args.val_path, 'validation') )
            os.makedirs('validation', exist_ok=True)
            
        if args.train:
            assert args.train_path, 'Path to training file required!'
            splits.append( (args.train_path, 'training') )
            os.makedirs('training', exist_ok=True)

        for fpath, data_dir in splits:
            df = open_tsv(fpath, data_dir)
            num_items_per_process = int(len(df) / args.num_processes)

            iterator = df.iterrows()
            for pid in range(args.num_processes):
                print(f'on {pid} script')
                with open(f'download-scripts/{data_dir}_{pid}.sh', 'w') as f:
                    for _ in range(num_items_per_process):
                        if _ % 50000 == 0:
                            print(f'on {_} item')
                        try:
                            idx, row = next(iterator)
                            fname = _file_name(row)
                            url = row.url
                            for char in ['?', '(', ')', '&']:
                                url = re.sub(f'\{char}', f'\{char}', url)
                                
                            if idx % 10000 == 0:
                                f.write(f'echo "on item {idx} of script {pid}"\n')
                            
                            f.write(f'wget -nc --timeout=3 -O {fname} {url}\n')
                        except StopIteration:
                            break
        
        
    else:
        if args.validation:
            assert args.val_path, 'Path to validation file required!'
            print('Downloading validation data..')
            df = open_tsv(args.val_path, 'validation')
            df_multiprocess(df=df, processes=args.num_processes, chunk_size=args.images_per_part,
                            func=download_image, dataset_name=data_name)
            df = df_from_shelve(chunk_size=args.images_per_part, func=download_image, dataset_name=data_name)
            df.to_csv("downloaded_%s_report.tsv.gz" % data_name, compression='gzip', sep='\t', header=False, index=False)
            print("Saved.")

        if args.train:
            assert args.train_path, 'Path to training file required!'
            print('Downloading training data..')
            df = open_tsv(args.train_path, 'training')
            df_multiprocess(df=df, processes=args.num_processes, chunk_size=args.images_per_part,
                            func=download_image, dataset_name=data_name)
            df = df_from_shelve(chunk_size=args.images_per_part, func=download_image, dataset_name=data_name)
            df.to_csv("downloaded_%s_report.tsv.gz" % data_name, compression='gzip', sep='\t', header=False, index=False)
            print("Saved.")



