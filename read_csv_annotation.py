import pandas as pd
import glob
import xml.etree.ElementTree as ET

xml_folder = r'G:\Books\Hands On Computer Vision\code\chapter6\BCCD_Dataset-master\BCCD\Annotations/'
xml_path_list = glob.glob(xml_folder + '*.xml')


def xml2dataframe(xmlpath):
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    df = pd.DataFrame()
    image_name = root.find('filename').text
    for object_itr in root.findall('object'):
        class_name = object_itr.find('name').text
        bndbox = object_itr.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        series = {'image_name': image_name, 'class_name': class_name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax,
                  'ymax': ymax}
        df = df.append(series, ignore_index=True)
    return df


master_df = pd.DataFrame()
for xml in xml_path_list:
    df = xml2dataframe(xml)
    master_df = master_df.append(df)

print("Number of Images:", master_df.image_name.nunique())
print("Number of Bounding Boxes:", len(master_df))


master_df.to_csv('full_dataset.csv')

from sklearn.model_selection import train_test_split
image_list = master_df.image_name.unique()
train, test = train_test_split(image_list, test_size=0.30)

train_df = master_df[master_df.image_name.isin(train)]
test_df = master_df[master_df.image_name.isin(test)]


def nameappend(image_name):
    return 'JPEGImages/' + image_name


def dataset_to_required_format(in_df):
    out_df = pd.DataFrame()
    out_df["filepath"] = in_df['image_name'].apply(nameappend)
    out_df["x1"] = in_df["xmin"]
    out_df["y1"] = in_df["ymin"]
    out_df["x2"] = in_df["xmax"]
    out_df["y2"] = in_df["ymax"]
    out_df["class_name"] = in_df["class_name"]

    return out_df

train_df_mod = dataset_to_required_format(train_df)
test_df_mod = dataset_to_required_format(test_df)

train_df_mod.to_csv("train.csv",header=False) # ,cols=["filepath","x1","y1","x2", "y2", "class_name"]
test_df_mod.to_csv("test.csv",header=False)