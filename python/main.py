from matplotlib import pyplot as plt
from copy import deepcopy
from typing import List

import pandas as pd
import numpy as np

import json
import cv2
import os

from statistics import mean, mode, median


def statistical_analysis(values, label):
    s = ""
    s += "*" * 50 + "\n"
    s += f"label: {label}" + "\n"
    s += f"mean: {round(mean(values), 2)}" + "\n"
    s += f"mode: {round(mode(values), 2)}" + "\n"
    s += f"median: {round(median(values), 2)}" + "\n"
    with open(os.path.join("results", f"statistical_analysis_{label}.txt"), "w") as f:
        f.write(s)
    print(s)


def read_jsons(filepaths: List[str]):
    datapoints = []
    for filepath in filepaths:
        with open(filepath, "r") as f:
            datapoints.append(json.load(f))
    return datapoints


def read_data(load_jsons: bool = False, load_images: bool = False, load_all: bool = False, max_n: int = None,
              do_prints: bool = True, use_test: bool=False):
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
        Health_list, Quantity_list, Fee_list, State_list, RescuerID_list, VideoAmt_list, Description_list,\
        PetID_list, PhotoAmt_list, AdoptionSpeed_list, texts_list, magnitudes_list, scores_list, languages_list, \
        categories_list, color_id_to_color_name, breed_id_to_breed_name, breed_id_to_type, state_id_to_state_name)


def main():
    max_n = 15
    load_images = False
    load_jsons = True
    final_data = read_data(load_jsons=load_jsons, load_images=load_images, max_n=max_n)

    Type_list_, Name_list_, Age_list_, Breed1_list_, Breed2_list_, Gender_list_, Color1_list_, Color2_list_, \
        Color3_list_, MaturitySize_list_, FurLength_list_, Vaccinated_list_, Dewormed_list_, Sterilized_list_, \
        Health_list_, Quantity_list_, Fee_list_, State_list_, RescuerID_list_, VideoAmt_list_, Description_list_, \
        PetID_list_, PhotoAmt_list_, AdoptionSpeed_list_, texts_list_, magnitudes_list_, scores_list_, languages_list_, \
        categories_list_, color_id_to_color_name_, breed_id_to_breed_name_, breed_id_to_type_, state_id_to_state_name_ = final_data

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
    texts_list = texts_list_
    magnitudes_list = magnitudes_list_
    scores_list = scores_list_
    languages_list = languages_list_
    categories_list = categories_list_
    color_id_to_color_name = color_id_to_color_name_
    breed_id_to_breed_name = breed_id_to_breed_name_
    breed_id_to_type = breed_id_to_type_
    state_id_to_state_name = state_id_to_state_name_

    print("Loaded data.")
    show = False

    statistical_analysis(AdoptionSpeed_list, "AdoptionSpeed")


    def plot_histogram(values, label, nbins=100, show=True):
        plt.hist(values, bins=nbins)
        plt.title(label)
        plt.savefig(os.path.join("plots", f"{label}.png"))
        if show:
            plt.show()


    def plot_scatterplot(xvalues, yvalues, label, show=True):
        plt.scatter(xvalues, yvalues)
        plt.title(label)
        plt.xlabel(label.split("_by_")[0])
        plt.ylabel(label.split("_by_")[1])
        plt.savefig(os.path.join("plots", f"{label}.png"))
        if show:
            plt.show()

    nbins = 100

    plot_scatterplot(Age_list, AdoptionSpeed_list, label="Age_by_AdoptionSpeed", show=show)
    plot_scatterplot(MaturitySize_list, AdoptionSpeed_list, label="MaturitySize_by_AdoptionSpeed", show=show)


    plot_histogram(Age_list, "Age", nbins=nbins, show=show)
    plot_histogram(AdoptionSpeed_list, "AdoptionSpeed", nbins=nbins, show=show)

    print(color_id_to_color_name.values())
    print(breed_id_to_breed_name.values())
    # print(breed_id_to_type.values())
    # print(list(breed_id_to_type.values())[0])
    print(state_id_to_state_name.values())

    print(set(Color1_list))
    print(set(Color2_list))
    print(set(Color3_list))

    use_replacement_dicts = False

    if use_replacement_dicts:
        Breed1_list = [breed_id_to_breed_name[breed_id_to_type[breed]] for breed in Breed1_list]
        Breed2_list = [breed_id_to_breed_name[breed_id_to_type[breed]] for breed in Breed2_list]

        State_list = [state_id_to_state_name[state] for state in State_list]

        Color1_list = [color_id_to_color_name[color] for color in Color1_list]
        Color2_list = [color_id_to_color_name[color] for color in Color2_list]
        Color3_list = [color_id_to_color_name[color] for color in Color3_list]


if __name__ == "__main__":
    main()
