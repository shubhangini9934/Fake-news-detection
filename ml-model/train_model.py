from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parent

dataset = [
    ("Government announces new public healthcare expansion for rural districts", 1),
    ("National weather center issues cyclone warning for coastal cities", 1),
    ("Researchers publish peer reviewed breakthrough in battery recycling", 1),
    ("Election commission releases verified turnout numbers for all states", 1),
    ("Central bank confirms updated interest rate guidance after review", 1),
    ("Health ministry shares official vaccination advisory before summer season", 1),
    ("Local authorities open emergency shelters after flood alert", 1),
    ("Space agency confirms successful satellite launch during morning window", 1),
    ("Court releases public ruling document in corruption investigation", 1),
    ("Energy department unveils approved solar subsidy plan for households", 1),
    ("Reuters reports inflation slows as food prices stabilise in March", 1),
    ("BBC reports heavy rain disrupts rail service across southern England", 1),
    ("The Hindu says parliament debate on education bill continues today", 1),
    ("Indian Express reports state board exam results to be released next week", 1),
    ("Associated Press confirms rescue teams reach quake hit villages", 1),
    ("WHO releases updated advisory on dengue prevention measures", 1),
    ("NASA scientists publish findings on asteroid surface samples", 1),
    ("Supreme court publishes full judgement in land dispute case", 1),
    ("Official police statement confirms road diversions after protest march", 1),
    ("Company files quarterly earnings report with stock exchange", 1),
    ("School board announces revised holiday schedule after monsoon warning", 1),
    ("IMD forecasts heatwave conditions across north India this week", 1),
    ("Election commission clarifies viral voting rumor is false in official notice", 1),
    ("Ministry of health confirms no nationwide lockdown has been announced", 1),
    ("University publishes peer reviewed study on air quality changes", 1),
    ("Celebrity reveals secret herb that cures every disease overnight", 0),
    ("Breaking miracle pill melts 20 kilos in a week doctors hate it", 0),
    ("You will not believe what this hidden government chip does to your brain", 0),
    ("Anonymous post says moonlight water reverses aging instantly", 0),
    ("Click now to claim emergency cash reward before midnight", 0),
    ("Secret banker leak proves all money will disappear tomorrow", 0),
    ("Viral chain message says smartphones explode after software update", 0),
    ("Magic spice mix guarantees cancer recovery without treatment", 0),
    ("This shocking cure was banned because hospitals fear losing profits", 0),
    ("Unverified blog claims invisible aircraft landed in village market", 0),
    ("Forward this message now or your bank account will be frozen tonight", 0),
    ("Doctors hate this one trick that regrows organs in three days", 0),
    ("Hidden video proves moon landing footage was filmed in local studio", 0),
    ("Secret government order says all schools will shut forever from Monday", 0),
    ("Whatsapp rumor claims salt water cures every viral infection instantly", 0),
    ("Unknown website says famous actor died but no official source confirms it", 0),
    ("Shocking post claims ATM machines will stop working after midnight", 0),
    ("Viral message says 5G towers cause sudden nosebleeds in children", 0),
    ("Anonymous account says drinking kerosene removes kidney stones", 0),
    ("Clickbait site claims election result leaked before voting started", 0),
    ("Miracle crypto formula guarantees double money in one hour", 0),
    ("No source report says vaccines contain tracking chips and magnets", 0),
    ("Conspiracy blog claims foreign birds are spying drones sent by secret agency", 0),
    ("Unverified video says bridge collapsed today but footage is from another country", 0),
    ("Chain message claims police will arrest people wearing black clothes tomorrow", 0)
]

df = pd.DataFrame(dataset, columns=["text", "label"])

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
model.fit(X, y)

with open(BASE_DIR / "model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open(BASE_DIR / "vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model trained and saved in ml-model/")
