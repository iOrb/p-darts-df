import random
import argparse
import os
import json

# URLs and filenames
USAGES = ['train','test','val']

def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-d', '--data', type=str, default='data/face_forensics', help='data dir')
    parser.add_argument('-si','--splits_dir_in', type=str, default='splits', help='splits dir in')
    parser.add_argument('-so','--splits_dir_out', type=str, default='splits_new', help='splits dir out')
    parser.add_argument('-n', '--num_split_versions', type=int, default=3,
                        help='number of different splits directories')
    parser.add_argument('-ntr', '--num_videos_train', type=int, default=None,
                        help='Number of videos for training, if None pick All')
    args = parser.parse_args()
    return args


def main(args):
    splits_list = []
    for use in USAGES:
        json_split_fn = os.path.join(args.data, args.splits_dir_in, f'{use}.json')
        with open(json_split_fn) as json_file:
            splits_list += json.load(json_file)

    os.mkdir(os.path.join(args.data, args.splits_dir_out))
    for i in range(args.num_split_versions):
        os.mkdir(os.path.join(args.data, args.splits_dir_out, str(i)))
        random.shuffle(splits_list)
        for use in USAGES:
            json_split_fn_out = os.path.join(args.data, args.splits_dir_out, str(i), f'{use}.json')
            with open(json_split_fn_out, 'w') as json_file:
                if use == 'train':
                    if args.num_videos_train and  args.num_videos_train < 360:
                        json.dump(splits_list[:args.num_videos_train],json_file)
                    else:
                        json.dump(splits_list[:360], json_file)
                elif use == 'val':
                    json.dump(splits_list[360:430], json_file)
                elif use == 'test':
                    json.dump(splits_list[430:], json_file)

def convert_list_to_string(list):
    string = '[\n'
    for e in list:
        string += "\t[\n\t\t\"" + str(e[0]) + "\",\n\t\t\"" + str(e[1]) + "\"\n\t],\n"
    string += "]"
    return string

if __name__ == '__main__':
    args = parse_args()
    main(args)
