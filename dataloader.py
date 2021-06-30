from __future__ import print_function, division

import os
from typing import List, Tuple

data_dir = '/home/shane/PycharmProjects/MedNLP/drive-download-20210621T204018Z-001'
subdirs = ['Groenies', 'Heath', 'Juan', 'Nick', 'Pieter', 'Santa']

col_headings = ['MRN',
                'NGT',
                'ET',
                'Orthosis',
                'Alignment',
                'Soft tissue Swelling',
                'Listhesis',
                'BDI',
                'ADI',
                'Fracture',
                'Dislocation',
                'Pedicle/Pars #',
                'TP #',
                'Facet #',
                'Lamina #',
                'Spinous #',
                'Osteophyte #',
                'Sclerosis',
                'Osteophyte',
                'Bridging osteophyte',
                'Disc height loss',
                'OPLL',
                'DISH',
                'Lytic lesion',
                'Bony destruction',
                'End-plate erosion',
                'Osteopaenia',
                'Scalloping',
                'Ankylosis',
                'Os odontoideum',
                'Klippel Feil',
                ]


def get_labels(superdir, subdirs, verbose=False) -> Tuple[List[str], List[List[str]]]:
    """
    finds all valid labels within the subdirs

    :returns: tuple(labels, label_files)
    """
    print("loading med text data for processing")

    expected_items = 31

    corrects = 0
    total = 0
    label_files = []
    labels = []
    all_text: List[List[str]] = []

    for subdir in subdirs:
        files = set(os.listdir(os.path.join(superdir, subdir)))
        for file in files:
            # skip if not label file or if label file has no corresponding image file in folder
            if not file.endswith('.txt') or not file[:-4].isdigit(): continue

            raw_label = ''
            text_lines = []
            with open(f'{os.path.join(superdir, subdir)}/{file}', 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    line = line[:-1].strip()
                    parsed = line.split('\t')
                    if len(parsed) == expected_items:
                        raw_label = parsed
                    elif len(line) > 0 and line != "DATA":
                        if i == 0 and ":" in line:
                            continue
                        text_lines.append(line)

                if not raw_label:
                    total += 1
                    continue

                labels += [raw_label[1:]]  # cut off id at the start
                all_text.append(text_lines)
                label_files += [os.path.join(superdir, subdir, file)]
                corrects += 1

                total += 1

    print(f'{corrects}/{total} correct')
    return labels, all_text


label_text, comment_text = get_labels(data_dir, subdirs)
 # = pd.DataFrame(, columns=col_headings)

# for i in range(len(labels)):
#     print("label:", labels[i], "text:", all_text[i])