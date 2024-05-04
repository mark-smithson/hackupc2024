import pandas as pd
from transformers import pipeline
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json


def get_all_flights(orig, dest):
    f = open('data/flightResults.json')

    flights = json.load(f)

    res = {}
    for flight in flights:
        if flights[flight]['data']['itineraries'][0]['legs'][0]['origin']['city'] == orig and \
                flights[flight]['data']['itineraries'][0]['legs'][0]['destination']['city'] == dest:
            res = []
            for it in flights[flight]['data']['itineraries']:
                flightPossible = {'price': it['price']['formatted'], 'go_stopCount': it['legs'][0]['stopCount'],
                                  'return_stopCount': it['legs'][1]['stopCount'],
                                  'go_departureTime': it['legs'][0]['departure'],
                                  'go_arrivalTime': it['legs'][0]['arrival'],
                                  'return_departureTime': it['legs'][1]['departure'],
                                  'return_arrivalTime': it['legs'][1]['arrival'],
                                  'go_duration': it['legs'][0]['durationInMinutes'],
                                  'return_duration': it['legs'][1]['durationInMinutes'],
                                  'go_flightNumber': it['legs'][0]['segments'][0]['flightNumber'],
                                  'return_flightNumber': it['legs'][1]['segments'][0]['flightNumber']}

                res.append(flightPossible)

    return res


def suggest_flight(orig, dest):
    # orig and dest are the names of the cities, for example suggest_flight('Barcelona', 'Madrid')
    flightsPossible = get_all_flights(orig, dest)
    bestFlight = {}
    minPrice = 9999
    for flight in flightsPossible:
        if int(flight['price'][1:]) < minPrice:
            bestFlight = flight
            minPrice = int(flight['price'][1:])

    return bestFlight


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

    def add_trip_features(self, trips, people_in_cities, events, embedding_model):
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
                city_events, interests, embedding_model
            )

            self.friends.append(people_in_city)
            self.events.append(event)
        # choose an event based on the common interests of the people in the city

        pass

    def get_llm_trip_summary(self, tokenizer, answers_model):
        question = """The following are events happening in {city}.
{events}
Make a vacation plan lasting {n_days} days to attend those events. Render it as a markdown and add emojis to make it more engaging."""
        event_names = [n[1]['Name'] for n in self.events]
        question = question.format(city=self.arrival_city, events=str.join("\n", event_names), n_days=len(self.events)
                                   )
        messages = [
            {"role": "user", "content": question}
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        encodeds = encodeds.to(answers_model.device)
        generated_ids = answers_model.generate(encodeds, max_new_tokens=400, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)[0].split("<|end_header_id|>\n\n")[-1]
        return decoded, question


def init_llm_models():
    # Use a pipeline as a high-level helper
    login(token=os.environ["HF_TOKEN"])
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    answers_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    return tokenizer, answers_model, embedding_model


def suggest_event(events, user_interests, embedding_model):
    """given a list of events choose the most appropiate one for that day based on interests of each user,
    Input: list of events, dictionary of users and interests
    """

    # Use a pipeline as a high-level helper

    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    events_embeddings = embedding_model.encode([event["Name"] for event in events])

    interests = str.join(",", {i for l in user_interests.values() for i in l})

    interests_embeddings = embedding_model.encode(interests)
    # compute similarity between events and interests
    similarities = util.pytorch_cos_sim(events_embeddings, interests_embeddings)

    argmax = similarities.argmax()

    # todo use chatbot after similarity
    # prompt = """Given the user interests
    # {interests}
    # Choose the most appropiate event for the day from the list of events
    # {events}"""
    return argmax, events[argmax], similarities
