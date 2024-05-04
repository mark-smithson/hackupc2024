def load_input_dataframe():
    import pandas as pd

    df = pd.read_csv("data/travelperk-subset.csv")

    df["Departure Date"] = pd.to_datetime(df["Departure Date"], format="%d/%m/%Y")
    df["Return Date"] = pd.to_datetime(df["Return Date"], format="%d/%m/%Y")
    df["Activities"] = df["Activities"].apply(lambda x: x.split(", "))

    return df
