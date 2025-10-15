import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

animals = [
    "ant",
    "bear",
    "bee",
    "bird",
    "butterfly",
    "camel",
    "cat",
    "cow",
    "crab",
    "crocodile",
    "dog",
    "dolphin",
    "duck",
    "elephant",
    "flamingo",
    "frog",
    "giraffe",
    "hedgehog",
    "horse",
    "kangaroo",
    "lion",
    "lobster",
    "monkey",
    "mosquito",
    "mouse",
    "octopus",
    "owl",
    "panda",
    "parrot",
    "penguin",
    "pig",
    "rabbit",
    "raccoon",
    "rhinoceros",
    "scorpion",
    "sea turtle",
    "shark",
    "sheep",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "swan",
    "tiger",
    "whale",
    "zebra",
]

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "ashishjangra27/doodle-dataset",
    "master_doodle_dataframe.csv",
)

counts = df["word"].value_counts()
missing_animals = [animal for animal in animals if animal not in counts]

if missing_animals:
    print("Missing classes with no samples:", missing_animals)
else:
    print("All requested animals have samples available.")

df_animals = df[df["word"].isin(animals)].reset_index(drop=True)
print(df_animals["word"].value_counts())

# optional: cache the subset so subsequent runs skip KaggleHub
df_animals.to_csv("archive/animal_doodles.csv", index=False)
