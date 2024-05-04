import pandas as pd

def suggest_flight():
    # given list of flights and people travelling on the same date chose the best flight
    pass


def get_fake_events():
    events = {
        "Barcelona": [
            {"Name": "Concert", "Type": "music", "Date": "2024-02-01"},
            {"Name": "Football match", "Type": "sports", "Date": "2024-02-01"},
            {"Name": "Food festival", "Type": "food", "Date": "2024-02-01"},
        ],
        "Paris": [
            {"Name": "Concert", "Type": "music", "Date": "2024-02-01"},
            {"Name": "Football match", "Type": "sports", "Date": "2024-02-01"},
            {"Name": "Food festival", "Type": "food", "Date": "2024-02-01"},
        ]
    }
    return events


def get_fake_interests():
    interests = {
        "user1": ["Cooking Classes", "Concerts"],
        "user2": ["Football", "Concerts"],
        "user3": ["Cooking Classes", "Football"],
    }
    return interests


def get_interests_from_df(df, people):
    df = df[df["Traveller Name"].isin(people)]
    # use iterrows
    interests = {
        user: activities
        for user, activities in zip(df["Traveller Name"], df["Activities"])
    }
    return interests


def demo_suggest_event():

    print(suggest_event(get_fake_events()["Barcelona"], get_fake_interests()))


from sentence_transformers import SentenceTransformer, util


class Trip:
    def __init__(
        self, user, interests, depart_city, arrival_city, depart_date, return_date
    ):
        self.user = user
        self.interests = interests
        self.depart_flight = None
        self.depart_city = depart_city
        self.return_flight = None
        self.arrival_city = arrival_city
        self.depart_date = depart_date
        self.return_date = return_date
        self.events = []
        self.friends = []

    def add_trip_features(self, trips, people_in_cities, events):
        self.depart_flight = suggest_flight()
        self.return_flight = suggest_flight()

        # for each day in the trip
        for day in pd.date_range(self.depart_date, self.return_date):
            # get the people in the city
            people_in_city = people_in_cities[
                (people_in_cities["arrival_city"] == self.arrival_city)
                & (people_in_cities["date"] == day)
            ]["people"].values[0]

            # get the events in the city
            city_events = events[self.arrival_city]
            # todo filter on date and inputs as df
            interests = get_interests_from_df(trips, people_in_city)
            # suggest an event
            event = suggest_event(
                city_events, interests
            )

            self.friends.append(people_in_city)
            self.events.append(event)
        # choose an event based on the common interests of the people in the city

        pass

    def get_llm_trip_summary(self):
        pass

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def suggest_event(events, user_interests):
    """given a list of events choose the most appropiate one for that day based on interests of each user,
    Input: list of events, dictionary of users and interests
    """

    # Use a pipeline as a high-level helper

    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    events_embeddings = model.encode([event["Name"] for event in events])

    interests = str.join(",", {i for l in user_interests.values() for i in l})

    interests_embeddings = model.encode(interests)
    # compute similarity between events and interests
    similarities = util.pytorch_cos_sim(events_embeddings, interests_embeddings)

    argmax = similarities.argmax()

    # todo use chatbot after similarity
    # prompt = """Given the user interests
    # {interests}
    # Choose the most appropiate event for the day from the list of events
    # {events}"""
    return argmax, events[argmax], similarities
