from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from copy import deepcopy
from typing import List
from collections import Counter
from tqdm import tqdm

import seaborn as sns
import pandas as pd
import numpy as np

import json
import cv2
import os

from statistics import mean, mode, median

all_columns_names = ['Type', 'Name', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
                     'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity',
                     'Fee', 'State', 'RescuerID', 'VideoAmt', 'Description', 'PetID', 'PhotoAmt', 'AdoptionSpeed']


def plot_histogram(values, label, show, nbins=100):
    fig = plt.figure(figsize=(9, 6))
    plt.hist(values, bins=nbins)
    plt.title(label)
    plt.legend()
    plt.savefig(os.path.join("plots", f"histogram_{label}.png"))
    if show:
        plt.show()


def plot_scatterplot(xvalues, yvalues, show, label):
    fig = plt.figure(figsize=(9, 6))
    plt.scatter(xvalues, yvalues)
    plt.title(label)
    plt.xlabel(label.split("_by_")[0])
    plt.ylabel(label.split("_by_")[1])
    plt.savefig(os.path.join("plots", f"scatterplot_{label}.png"))
    if show:
        plt.show()


def plot_scatterplot_3D(xvalues, yvalues, zvalues, label, show):
    fig = plt.figure(figsize=(9, 6))
    plt.scatter(xvalues, yvalues, zvalues)
    plt.title(label)
    plt.xlabel(label.split("_by_")[0])
    plt.ylabel(label.split("_by_")[1])
    plt.legend()
    plt.savefig(os.path.join("plots", f"scatterplot_{label}.png"))
    if show:
        plt.show()


nbins = 100


def plot_barchart(labels, values, label: str, show: bool, aggregation_strategy: str = "mean", ):
    fig = plt.figure(figsize=(9, 6))

    label_to_average_value = dict()
    for (label_, value) in zip(labels, values):
        if label_ not in label_to_average_value:
            label_to_average_value[label_] = [value]
        else:
            label_to_average_value[label_].append(value)
    for (label_, value) in label_to_average_value.items():
        if aggregation_strategy == "mean":
            label_to_average_value[label_] = mean(value)
        elif aggregation_strategy == "sum":
            label_to_average_value[label_] = mean(value)
        else:
            raise Exception(f"Wrong aggregation_strategy given: {aggregation_strategy}!")

    keys = label_to_average_value.keys()
    values = label_to_average_value.values()
    plt.bar(keys, values, color="red", width=0.4)
    plt.xlabel(label.split("_by_")[0])
    plt.ylabel(label.split("_by_")[1])
    plt.title(label)
    plt.legend()
    plt.savefig(os.path.join("plots", f"barchart_{label}.png"))
    if show:
        plt.show()


def statistical_analysis(values, label, show):
    fig = plt.figure(figsize=(9, 6))
    s = ""
    s += "*" * 50 + "\n"
    s += f"label: {label}" + "\n"
    s += f"mean: {round(mean(values), 2)}" + "\n"
    s += f"mode: {round(mode(values), 2)}" + "\n"
    s += f"median: {round(median(values), 2)}" + "\n"

    plt.boxplot(values)
    plt.title(label)
    plt.legend()
    plt.savefig(f"plots/statistical_analysis_boxplot_{label}.png")
    if show:
        plt.show()
    with open(os.path.join("results", f"statistical_analysis_{label}.txt"), "w") as f:
        f.write(s)
    print(s)


def read_jsons(filepaths: List[str]):
    datapoints = []
    for filepath in filepaths:
        with open(filepath, "r") as f:
            datapoints.append(json.load(f))
    return datapoints


from sentence_transformers import SentenceTransformer, util

sent_transf_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def encode_text(text: str):
    return sent_transf_model.encode(text, convert_to_tensor=True)


def read_data(load_jsons: bool = False, load_images: bool = False, load_all: bool = False, max_n: int = None,
              do_prints: bool = True, use_test: bool = False):
    # sentiment related data
    texts_list = []
    magnitudes_list = []
    scores_list = []
    languages_list = []
    categories_list = []

    read_all_csvs = False
    final_data = []
    if load_all is True:
        load_jsons = True
        load_images = True
    train = pd.read_csv('data/train/train.csv')

    print(f"Number of nan values: {train.isnull().sum()}")
    show = False

    def describe_nan(df):
        return pd.DataFrame([(i, df[df[i].isna()].shape[0], df[df[i].isna()].shape[0] / df.shape[0]) for i in df.columns],
                            columns=['column', 'nan_counts', 'nan_rate'])

    nan_df = describe_nan(train)
    nan_df.to_csv("results/nan_df.csv")
    print(nan_df)

    # replace nan values
    train = train.fillna(method="ffill")

    fig = plt.figure(figsize=(18, 12))
    corr = train.corr()
    sns.heatmap(corr, cmap="Blues", annot=True)
    plt.legend()
    plt.savefig("plots/heatmap_correlations_between_variables.png")
    if show:
        plt.show()
    df = train
    AdoptionSpeed_list = df["AdoptionSpeed"].to_list()
    Type_list = df['Type'].to_list()
    Name_list = df['Name'].to_list()
    Age_list = df['Age'].to_list()
    Breed1_list = df['Breed1'].to_list()
    Breed2_list = df['Breed2'].to_list()
    Gender_list = df['Gender'].to_list()
    Color1_list = df['Color1'].to_list()
    Color2_list = df['Color2'].to_list()
    Color3_list = df['Color3'].to_list()
    MaturitySize_list = df['MaturitySize'].to_list()
    FurLength_list = df['FurLength'].to_list()
    Vaccinated_list = df['Vaccinated'].to_list()
    Dewormed_list = df['Dewormed'].to_list()
    Sterilized_list = df['Sterilized'].to_list()
    Health_list = df['Health'].to_list()
    Quantity_list = df['Quantity'].to_list()
    Fee_list = df['Fee'].to_list()
    State_list = df['State'].to_list()
    RescuerID_list = df['RescuerID'].to_list()
    VideoAmt_list = df['VideoAmt'].to_list()
    Description_list = df['Description'].to_list()
    PetID_list = df['PetID'].to_list()
    PhotoAmt_list = df['PhotoAmt'].to_list()

    if do_prints:
        print("*" * 50)
        print("train")
        print(train.columns)
        print(train.dtypes)
        print(train.describe(), len(train))
        print(train.head(n=1))

    inspect_categorical = True
    categorical_variables = []
    if inspect_categorical:
        for column in all_columns_names:
            if len(train) > len(set(train[column].to_list())):
                categorical_variables.append(column)
        print(categorical_variables)

    train_metadata = [os.path.join('data/train_metadata', file) for file in os.listdir('data/train_metadata')]  # jsons
    initial_n = len(train_metadata)
    if max_n:
        train_metadata = train_metadata[:max_n]
    if load_jsons:
        train_metadata = read_jsons(train_metadata)
        if do_prints:
            print("*" * 50)
            print("test")
            print(train_metadata[0].keys(), initial_n)
            # print(train_metadata[0])

    train_images = [os.path.join('data/train_images', file) for file in os.listdir('data/train_images')]  # jpgs
    initial_n = len(train_images)
    if max_n:
        train_images = train_images[:max_n]
    if load_images:
        train_images = [cv2.imread(filepath) for filepath in train_images]
        if do_prints:
            print("*" * 50)
            print("train_images")
            print(train_images[0].shape, initial_n)

    train_sentiment = [os.path.join("data/train_sentiment", file) for file in os.listdir('data/train_sentiment')]  # jsons
    initial_n = len(train_sentiment)
    if max_n:
        train_sentiment = train_sentiment[:max_n]
    if load_jsons:
        train_sentiment = read_jsons(train_sentiment)
        for datapoint in train_sentiment:
            sentences = datapoint["sentences"]
            text = " ".join([sentence["text"]["content"] for sentence in sentences])
            texts_list.append(text)
            magnitudes_list.append(datapoint["documentSentiment"]["magnitude"])
            scores_list.append(datapoint["documentSentiment"]["score"])
            languages_list.append(datapoint["language"])
            categories_list.append(datapoint["categories"])
        if do_prints:
            print("*" * 50)
            print("train_sentiment")
            print(train_sentiment[0].keys(), initial_n)
            # print(train_sentiment[0])

    if use_test:
        test = pd.read_csv('data/test/test.csv')
        Type_list.extend(df['Type'].to_list())
        Name_list.extend(df['Name'].to_list())
        Age_list.extend(df['Age'].to_list())
        Breed1_list.extend(df['Breed1'].to_list())
        Breed2_list.extend(df['Breed2'].to_list())
        Gender_list.extend(df['Gender'].to_list())
        Color1_list.extend(df['Color1'].to_list())
        Color2_list.extend(df['Color2'].to_list())
        Color3_list.extend(df['Color3'].to_list())
        MaturitySize_list.extend(df['MaturitySize'].to_list())
        FurLength_list.extend(df['FurLength'].to_list())
        Vaccinated_list.extend(df['Vaccinated'].to_list())
        Dewormed_list.extend(df['Dewormed'].to_list())
        Sterilized_list.extend(df['Sterilized'].to_list())
        Health_list.extend(df['Health'].to_list())
        Quantity_list.extend(df['Quantity'].to_list())
        Fee_list.extend(df['Fee'].to_list())
        State_list.extend(df['State'].to_list())
        RescuerID_list.extend(df['RescuerID'].to_list())
        VideoAmt_list.extend(df['VideoAmt'].to_list())
        Description_list.extend(df['Description'].to_list())
        PetID_list.extend(df['PetID'].to_list())
        PhotoAmt_list.extend(df['PhotoAmt'].to_list())

    if use_test:
        if do_prints:
            print("*" * 50)
            print("test")
            print(test.columns, len(test))
            print(test.head(n=1))

        test_metadata = [os.path.join("data/test_metadata", file) for file in os.listdir('data/test_metadata')]  # jsons
        initial_n = len(test_metadata)
        if max_n:
            test_metadata = test_metadata[:max_n]
        if load_jsons:
            test_metadata = read_jsons(test_metadata)
            if do_prints:
                print("*" * 50)
                print("test_metadata")
                print(test_metadata[0].keys(), initial_n)
                # print(test_metadata[0])

        test_images = [os.path.join("data/test_images", file) for file in os.listdir('data/test_images')]  # jpgs
        initial_n = len(test_images)
        if max_n:
            test_images = test_images[:max_n]
        if load_images:
            test_images = [cv2.imread(filepath) for filepath in test_images]
            if do_prints:
                print("*" * 50)
                print("test_images")
                print(test_images[0].shape, initial_n)

        test_sentiment = [os.path.join("data/test_sentiment", file) for file in os.listdir('data/test_sentiment')]
        initial_n = len(test_sentiment)
        if max_n:
            test_sentiment = test_sentiment[:max_n]
        if load_jsons:
            test_sentiment = read_jsons(test_sentiment)
            for datapoint in test_sentiment:
                sentences = datapoint["sentences"]
                text = " ".join([sentence["text"]["content"] for sentence in sentences])
                texts_list.append(text)
                magnitudes_list.append(datapoint["documentSentiment"]["magnitude"])
                scores_list.append(datapoint["documentSentiment"]["score"])
                languages_list.append(datapoint["language"])
                categories_list.append(datapoint["categories"])
            if do_prints:
                print("*" * 50)
                print("test_sentiment")
                print(test_sentiment[0].keys(), initial_n)
                # print(test_sentiment[0])

    if read_all_csvs:
        pet_color_labels = pd.read_csv('data/PetFinder-ColorLabels.csv')
        pet_breed_labels = pd.read_csv('data/PetFinder-BreedLabels.csv')
        pet_state_labels = pd.read_csv('data/PetFinder-StateLabels.csv')
        if do_prints:
            print("*" * 50)
            print("pet_color_labels")
            print(pet_color_labels.columns, len(pet_color_labels))
            print("pet_breed_labels")
            print(pet_breed_labels.columns, len(pet_breed_labels))
            print("pet_state_labels")
            print(pet_state_labels.columns, len(pet_state_labels))

        color_labels_camel = pd.read_csv('data/ColorLabels.csv')
        breed_labels_camel = pd.read_csv('data/BreedLabels.csv')
        state_labels_camel = pd.read_csv('data/StateLabels.csv')
        if do_prints:
            print("*" * 50)
            print("color_labels_camel")
            print(color_labels_camel.columns, len(color_labels_camel))
            print("breed_labels_camel")
            print(breed_labels_camel.columns, len(breed_labels_camel))
            print("state_labels_camel")
            print(state_labels_camel.columns, len(state_labels_camel))

    color_labels_snake = pd.read_csv('data/color_labels.csv')
    state_labels_snake = pd.read_csv('data/state_labels.csv')
    breed_labels_snake = pd.read_csv('data/breed_labels.csv')

    if read_all_csvs:
        assert color_labels_camel.equals(color_labels_snake)
        assert breed_labels_camel.equals(breed_labels_snake)
        assert state_labels_camel.equals(state_labels_snake)

        assert color_labels_camel.equals(pet_color_labels)
        assert breed_labels_camel.equals(pet_breed_labels)
        assert state_labels_camel.equals(pet_state_labels)

    if do_prints:
        print("*" * 50)
        print("color_labels_snake")
        print(color_labels_snake.columns, len(color_labels_snake))
        color_id_to_color_name = {id: name for (id, name) in zip(color_labels_snake["ColorID"].to_list(),
                                                                 color_labels_snake["ColorName"].to_list())}

        print("breed_labels_snake")
        print(breed_labels_snake.columns, len(breed_labels_snake))
        breed_id_to_breed_name = {id: name for (id, name) in zip(breed_labels_snake["BreedID"].to_list(),
                                                                 breed_labels_snake["BreedName"].to_list())}

        breed_id_to_type = {id: name for (id, name) in zip(breed_labels_snake["BreedID"].to_list(),
                                                           breed_labels_snake["Type"].to_list())}

        print("state_labels_snake")
        print(state_labels_snake.columns, len(state_labels_snake))
        state_id_to_state_name = {id: name for (id, name) in zip(state_labels_snake["StateID"].to_list(),
                                                                 state_labels_snake["StateName"].to_list())}

    if use_test:
        print(len(train.columns), len(test.columns))

        print(set(list(train.columns)).difference(set(list(test.columns))))
        print(set(list(test.columns)).difference(set(list(train.columns))))

    return (Type_list, Name_list, Age_list, Breed1_list, Breed2_list, Gender_list, Color1_list, Color2_list, \
            Color3_list, MaturitySize_list, FurLength_list, Vaccinated_list, Dewormed_list, Sterilized_list, \
            Health_list, Quantity_list, Fee_list, State_list, RescuerID_list, VideoAmt_list, Description_list, \
            PetID_list, PhotoAmt_list, AdoptionSpeed_list, texts_list, magnitudes_list, scores_list, languages_list, \
            categories_list, color_id_to_color_name, breed_id_to_breed_name, breed_id_to_type, state_id_to_state_name)


from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_scaler(scaling_option):
    if scaling_option == "standard":
        return StandardScaler()
    elif scaling_option == "minmax":
        return MinMaxScaler()
    else:
        raise Exception(f"Wrong scaling option give: {scaling_option}!")


def plot_vanilla_barchart(data, label):
    fig = plt.figure(figsize=(9, 6))
    # creating the bar plot
    plt.bar(data.keys(), data.values(), color='maroon', width=0.4)
    plt.title(label)
    plt.legend()
    # plt.savefig(f"plots/barchart_{label}.png")
    plt.show()


def main():
    show = False
    do_ml = False
    show_at_the_end = False
    max_n = None
    load_images = False
    load_jsons = True
    final_data = read_data(load_jsons=load_jsons, load_images=load_images, max_n=max_n)

    Type_list_, Name_list_, Age_list_, Breed1_list_, Breed2_list_, Gender_list_, Color1_list_, Color2_list_, \
        Color3_list_, MaturitySize_list_, FurLength_list_, Vaccinated_list_, Dewormed_list_, Sterilized_list_, \
        Health_list_, Quantity_list_, Fee_list_, State_list_, RescuerID_list_, VideoAmt_list_, Description_list_, \
        PetID_list_, PhotoAmt_list_, AdoptionSpeed_list_, texts_list_, magnitudes_list_, scores_list_, languages_list_, \
        categories_list_, color_id_to_color_name_, breed_id_to_breed_name_, breed_id_to_type_, state_id_to_state_name_ = final_data

    """
    SUMMARY
    DESCRIPTION
    
    Type <class 'int'> 2
    Name <class 'str'> 9060
    Age <class 'int'> 106
    Breed1 <class 'int'> 176
    Breed2 <class 'int'> 135
    Gender <class 'int'> 3
    Color1 <class 'int'> 7
    Color2 <class 'int'> 7
    Color3 <class 'int'> 6
    MaturitySize <class 'int'> 4
    FurLength <class 'int'> 3
    Vaccinated <class 'int'> 3
    Dewormed <class 'int'> 3
    Sterilized <class 'int'> 3
    Health <class 'int'> 3
    Quantity <class 'int'> 19
    Fee <class 'int'> 74
    State <class 'int'> 14
    RescuerID <class 'str'> 5595
    VideoAmt <class 'int'> 9
    
    Description <class 'str'> 14032  # unique
    PetID <class 'str'> 14993   # unique
    
    PhotoAmt <class 'float'> 31
    AdoptionSpeed <class 'int'> 5
    """
    Type_list = Type_list_
    Name_list = Name_list_
    Age_list = Age_list_
    Breed1_list = Breed1_list_
    Breed2_list = Breed2_list_
    Gender_list = Gender_list_
    Color1_list = Color1_list_
    Color2_list = Color2_list_
    Color3_list = Color3_list_
    MaturitySize_list = MaturitySize_list_
    FurLength_list = FurLength_list_

    Vaccinated_list = Vaccinated_list_
    Dewormed_list = Dewormed_list_
    Sterilized_list = Sterilized_list_
    Health_list = Health_list_

    Quantity_list = Quantity_list_
    Fee_list = Fee_list_
    State_list = State_list_
    RescuerID_list = RescuerID_list_
    VideoAmt_list = VideoAmt_list_
    Description_list = Description_list_
    PetID_list = PetID_list_
    PhotoAmt_list = PhotoAmt_list_
    AdoptionSpeed_list = AdoptionSpeed_list_

    print("*" * 50)
    print(VideoAmt_list[:4])
    print(Description_list[:4])
    print(PhotoAmt_list[:4])

    print(len(RescuerID_list), len(set(RescuerID_list_)))
    print(len(PetID_list), len(set(PetID_list)))
    rescue_id_to_pet_id = dict()
    for (rescuer_id, pet_id) in zip(RescuerID_list, PetID_list):
        if rescuer_id not in rescue_id_to_pet_id:
            rescue_id_to_pet_id[rescuer_id] = [pet_id]
        else:
            rescue_id_to_pet_id[rescuer_id].append(pet_id)

    texts_list = texts_list_
    embeddings = []
    filepath = "data/texts_embeddings.npy"
    if os.path.exists(filepath):
        embeddings = np.load(file=filepath, allow_pickle=True)
    else:
        print(f"Number of texts: {len(texts_list)}")
        for i, text in tqdm(enumerate(texts_list)):
            embeddings.append(encode_text(text))
            if i % 1000 == 0:
                # print(embeddings[0].shape)
                # print(np.array(embeddings).shape)
                np.save(file=filepath, arr=np.array([emb.numpy() for emb in embeddings]), allow_pickle=True)
        np.save(file=filepath, arr=np.array(embeddings), allow_pickle=True)

    print("Saved texts embeddings")

    magnitudes_list = magnitudes_list_
    scores_list = scores_list_

    print(scores_list[:5])
    print(magnitudes_list[:5])

    scores_list = [float(elem) for elem in scores_list]
    magnitudes_list = [float(elem) for elem in magnitudes_list]

    for elem in scores_list:
        if isinstance(elem, float) is False and isinstance(elem, int) is False:
            print(elem, "scores")
    for elem in magnitudes_list:
        if isinstance(elem, float) is False and isinstance(elem, int) is False:
            print(elem, "magnitudes")

    plot_histogram(magnitudes_list, "magnitudes_list", nbins=20, show=show)
    plot_histogram(scores_list, "scores_list", nbins=20, show=show)

    languages_list = languages_list_  # 4 languages, but english is 99%+
    categories_list = categories_list_  # all are empty

    categories_list = [cat for cat in categories_list if cat != []]
    print(f"There are {len(categories_list)} given categories")

    label = "Language distribution among the descriptions of the pets"
    data = Counter(languages_list)
    # plot_vanilla_barchart(data, label)
    print(data, label)

    color_id_to_color_name = color_id_to_color_name_
    breed_id_to_breed_name = breed_id_to_breed_name_
    breed_id_to_type = breed_id_to_type_
    state_id_to_state_name = state_id_to_state_name_

    features_list = [Type_list, Name_list, Age_list, Breed1_list, Breed2_list, Gender_list, Color1_list, Color2_list,
                     Color3_list, MaturitySize_list, FurLength_list, Vaccinated_list, Dewormed_list, Sterilized_list,
                     Health_list, Quantity_list, Fee_list, State_list, RescuerID_list, VideoAmt_list, Description_list,
                     PetID_list, PhotoAmt_list, AdoptionSpeed_list]

    ideal_animal = dict()
    for (feature_name, feature_type, length, set_length, feature_mode, features_types) in zip(all_columns_names,
                                                                                              [type(features[0]) for features in features_list],
                                                                                              [len(features) for features in features_list],
                                                                                              [len(set(features)) for features in features_list],
                                                                                              [mode(features) for features in features_list],
                                                                                              [[type(feat) for feat in features] for features in features_list]):
        # print(feature_name, feature_type, set_length, length, feature_mode)
        if set_length < length:
            ideal_animal[feature_name] = feature_mode
        for feat in features_types:
            if feat != feature_type:
                print("Feature mismatch: ", feature_name)
                break

    # exit(0)
    X, y = [], []
    X_train, X_test, y_train, y_test = [], [], [], []
    train = pd.read_csv("data/train/train.csv")

    for i in range(len(train)):
        datapoint = []
        for column in list(train.columns):
            if column not in ["RescuerID", "Description", "PetID", "Name"]:
                if column == "AdoptionSpeed":
                    y.append(train.at[i, column])
                else:
                    datapoint.append(train.at[i, column])
        X.append(datapoint)

    X = np.array(X)
    y = np.array(y)

    statistical_analysis(values=AdoptionSpeed_list, label="AdoptionSpeed", show=show)

    # print("All the names: ", set(Name_list))

    plot_scatterplot(Age_list, AdoptionSpeed_list, label="Age_by_AdoptionSpeed", show=show)
    plot_scatterplot(MaturitySize_list, AdoptionSpeed_list, label="MaturitySize_by_AdoptionSpeed", show=show)
    plot_scatterplot(Age_list, Color1_list, label="Age_by_Color", show=show)
    plot_scatterplot(Breed1_list, AdoptionSpeed_list, label="Breed1_by_AdoptionSpeed", show=show)
    plot_scatterplot(Breed2_list, AdoptionSpeed_list, label="Breed2_by_AdoptionSpeed", show=show)
    plot_scatterplot(State_list, AdoptionSpeed_list, label="State_by_AdoptionSpeed", show=show)

    plot_barchart(labels=Gender_list, values=AdoptionSpeed_list, label="Gender_by_AdoptionSpeed",
                  aggregation_strategy="mean", show=show)
    plot_barchart(Color1_list, Type_list, label="Color_by_Type", aggregation_strategy="sum", show=show)
    plot_barchart(Color1_list, AdoptionSpeed_list, label="Color_by_AdoptionSpeed", aggregation_strategy="sum", show=show)
    plot_barchart(Type_list, AdoptionSpeed_list, label="Type_by_AdoptionSpeed", aggregation_strategy="mean", show=show)

    plot_scatterplot_3D(Gender_list, Age_list, AdoptionSpeed_list, label="Gender_by_Age_by_AdoptionSpeed", show=show)
    plot_scatterplot_3D(FurLength_list, Age_list, AdoptionSpeed_list, label="FurLength_by_Age_by_AdoptionSpeed", show=show)

    plot_histogram(Fee_list, "Fee", nbins=nbins, show=show)
    plot_histogram(Age_list, "Age", nbins=nbins, show=show)
    plot_histogram(Quantity_list, "Quantity", nbins=nbins, show=show)
    plot_histogram(AdoptionSpeed_list, "AdoptionSpeed", nbins=nbins, show=show)

    # for (feature_name, features) in zip(all_columns_names, features_list):

    scaling_options = ["standard", "minmax"][:1]
    if do_ml:
        for scaling_option in scaling_options:
            scaler = load_scaler(scaling_option)
            if len(X) and len(y):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            task = "classification"
            if task == "classification":
                model = SVC()
            elif task == "regression":
                model = SVR()
            else:
                raise Exception(f"Wrong task given: {task}!")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            print("*" * 100)
            print(f"precision_score: {precision_score(y_pred, y_test, average='weighted')}")
            print(f"recall_score: {recall_score(y_pred, y_test, average='weighted')}")
            print(f"f1_score: {f1_score(y_pred, y_test, average='weighted')}")
            print(f"accuracy_score: {accuracy_score(y_pred, y_test)}")
            print(f"confusion_matrix: {confusion_matrix(y_pred, y_test)}")
            print("*" * 50)

            if task == "classification":
                model = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            elif task == "regression":
                model = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
            else:
                raise Exception(f"Wrong task given: {task}!")

            models, predictions = model.fit(X_train, X_test, y_train, y_test)
            print(models)
            print(predictions)
            print("*" * 50)

    print(f"ideal_animal: {ideal_animal}")

    print("Loaded data.")
    print(f"color_id_to_color_name: {color_id_to_color_name}")
    # print(f"breed_id_to_breed_name: {breed_id_to_breed_name}")
    # print(breed_id_to_type.values())
    # print(list(breed_id_to_type.values())[0])
    print(f"state_id_to_state_name: {state_id_to_state_name}")

    health_related_features_lists = np.array([Vaccinated_list, Dewormed_list, Sterilized_list, Health_list])
    health_related_features_names = ["Vaccinated", "Dewormed", "Sterilized", "Health"]
    for (health, health_features) in zip(health_related_features_names, [Vaccinated_list, Dewormed_list, Sterilized_list, Health_list]):
        plot_scatterplot(health_features, AdoptionSpeed_list, label=f"{health}_by_AdoptionSpeed", show=show)

    do_heatmap = False
    if do_heatmap:
        # plot the heatmap
        import seaborn as sns
        corr = health_related_features_lists.corr()
        sns.heatmap(corr, xticklabels=health_related_features_names, yticklabels=health_related_features_names)

    # print(set(Color1_list))
    #
    # print(set(Color2_list))
    # print(set(Color3_list))

    use_replacement_dicts = False

    if use_replacement_dicts:
        Breed1_list = [breed_id_to_breed_name[breed_id_to_type[breed]] for breed in Breed1_list]
        Breed2_list = [breed_id_to_breed_name[breed_id_to_type[breed]] for breed in Breed2_list]

        State_list = [state_id_to_state_name[state] for state in State_list]

        Color1_list = [color_id_to_color_name[color] for color in Color1_list]
        Color2_list = [color_id_to_color_name[color] for color in Color2_list]
        Color3_list = [color_id_to_color_name[color] for color in Color3_list]

        # print(f"Breed1_list: {Breed1_list}")
        # print(f"Breed2_list: {Breed2_list}")
        #
        # print(f"State_list {State_list}")
        #
        # print(f"Color1_list: {Color1_list}")
        # print(f"Color2_list: {Color2_list}")
        # print(f"Color3_list: {Color3_list}")

    if show_at_the_end:
        plt.show()


if __name__ == "__main__":
    main()
