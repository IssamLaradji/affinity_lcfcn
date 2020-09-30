import sys, os

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)



hash_dct = {'b04090f27c7c52bcec65f6ba455ed2d8': 'Fully_Supervised',
            '6d4af38d64b23586e71a198de2608333': 'LCFCN',
            '84ced18cf5c1fb3ad5820cc1b55a38fa': 'LCFCN+Affinity_(ours)',
            '63f29eec3dbe1e03364f198ed7d4b414': 'Point-level_Loss ',
            '017e7441c2f581b6fee9e3ac6f574edc': 'Cross_entropy_Loss+pseudo-mask'}

def make_latex_count(csv_path, column_name):
    import os
    import pandas as pd
    from tqdm import tqdm
    all_df = []
    paths = (os.listdir(csv_path))
    for file in tqdm(paths):
        if file.endswith('_latex.csv'):
            continue
        if file.endswith('.csv'):
            file_path = '{}/{}'.format(csv_path, file)
            print(file_path)
            org_DF = pd.read_csv(file_path).round(decimals=3)
            org_DF = org_DF[column_name]
            org_DF['Loss Function'] = hash_dct[file.split("_")[0]]
            all_df.append(org_DF)
    concat_df = pd.concat(all_df, axis=1)
    concat_df = concat_df.transpose()
    concat_df = concat_df[['Loss Function', 0, 1, 2]]
    concat_df.to_csv(os.path.join(csv_path , "%s_latex.csv"%column_name), index=False)
    concat_df.to_latex(os.path.join(csv_path , "%s_latex.tex"%column_name),
                         index=False, caption=column_name, label=column_name)
    print(concat_df)


def make_latex_habitat(csv_path, column_name):
    import os
    import pandas as pd
    from tqdm import tqdm
    all_df = []
    habitats = []
    paths = (os.listdir(csv_path))
    for file in tqdm(paths):
        if file.endswith('_latex.csv'):
            continue
        if file.endswith('.csv'):
            file_path = '{}/{}'.format(csv_path, file)
            print(file_path)
            org_DF = pd.read_csv(file_path).round(decimals=3)
            habitats = org_DF["Habitat"]
            org_DF = org_DF[column_name]
            org_DF['Loss Function'] = hash_dct[file.split("_")[0]]
            all_df.append(org_DF)
    concat_df = pd.concat(all_df, axis=1, ignore_index= True )
    concat_df.insert(0, "habitat", habitats, True)
    concat_df = concat_df.transpose()
    cols = list(concat_df.columns)
    cols = [cols[-1]] + cols[:-1]
    concat_df = concat_df[cols]
    concat_df.to_csv(os.path.join(csv_path , "%s_latex.csv"%column_name), index=False)
    concat_df.to_latex(os.path.join(csv_path , "%s_latex.tex"%column_name),
                         index=False, caption=column_name, label=column_name)
    print(concat_df)


if __name__ == '__main__':
    fish = '/mnt/public/predictions/fish/'
    habitat = '/mnt/public/predictions/habitat/'
    make_latex_count(fish, "IoU class 1")
    make_latex_habitat(habitat, "IoU class 1")
