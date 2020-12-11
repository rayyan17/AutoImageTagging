import json
import pandas as pd


def fetch_image_annotations_obj(file_path):
    with open(file_path) as f:
        data_instance = json.load(f)

    return data_instance


def get_image_id_and_paths(data):
    image_id = list()
    image_filename = list()
    for val in data['images']:
        image_id.append(val['id'])
        image_filename.append(val['file_name'])

    image_df = pd.DataFrame()
    image_df["image_id"] = image_id
    image_df["image_path"] = image_filename

    return image_df


def get_categories_df(data):
    category_id = list()
    super_category = list()
    category_name = list()
    for val in data['categories']:
        category_id.append(val["id"])
        super_category.append(val["supercategory"])
        category_name.append(val["name"])

    category_df = pd.DataFrame()
    category_df["category_id"] = category_id
    category_df["super_category"] = super_category
    category_df["category_name"] = category_name

    return category_df


def get_image_category_mapping(data):
    image_id = list()
    category_id = list()
    for val in data['annotations']:
        image_id.append(val['image_id'])
        category_id.append(val["category_id"])

    cat_image_df = pd.DataFrame()
    cat_image_df["image_id"] = image_id
    cat_image_df["category_id"] = category_id

    return cat_image_df


if __name__ == '__main__':
    annotations_file_path = "annotations/instances_val2014.json"
    annotation_data = fetch_image_annotations_obj(annotations_file_path)

    img_df = get_image_id_and_paths(annotation_data)
    cat_df = get_categories_df(annotation_data)
    img_cat_df = get_image_category_mapping(annotation_data)

    main_df = pd.merge(img_cat_df.drop_duplicates(), cat_df, on="category_id")
    main_df = pd.merge(main_df, img_df, on="image_id")

    images_all_super_cat = main_df.groupby("image_path")["super_category"].apply(lambda x: ','.join(x)).reset_index()
    images_all_sub_cat = main_df.groupby("image_path")["category_name"].apply(lambda x: ','.join(x)).reset_index()

    main_df.to_csv("image_labels_2014val.csv")
    images_all_super_cat.to_csv("image_super_cat_2014val.csv")
    images_all_sub_cat.to_csv("image_sub_cat_2014val.csv")
