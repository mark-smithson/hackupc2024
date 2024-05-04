import pandas as pd
from transformers import pipeline
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json


def special_case(orig, dest, depDate, retDate):
    if orig == 'Barcelona' and dest == 'Paris':
        if depDate == '18/08/2024' and retDate == '24/08/2024':
            return True, {'price': '$149', 'go_stopCount': 0,
                          'return_stopCount': 0,
                          'go_departureTime': '2024-08-18T11:35:00',
                          'go_arrivalTime': '2024-08-18T13:35:00',
                          'return_departureTime': '2024-08-24T10:10:00',
                          'return_arrivalTime': '2024-08-24T12:15:00',
                          'go_duration': '120',
                          'return_duration': '125'}
        elif depDate == '18/08/2024' and retDate == '24/08/2024':
            return True, {'price': '$149', 'go_stopCount': 0,
                          'return_stopCount': 0,
                          'go_departureTime': '2024-08-18T11:35:00',
                          'go_arrivalTime': '2024-08-18T13:35:00',
                          'return_departureTime': '2024-08-24T10:10:00',
                          'return_arrivalTime': '2024-08-24T12:15:00',
                          'go_duration': '120',
                          'return_duration': '125'}
        elif depDate == '20/08/2024' and retDate == '25/08/2024':
            return True, {'price': '$167', 'go_stopCount': 0,
                          'return_stopCount': 0,
                          'go_departureTime': '2024-08-18T09:35:00',
                          'go_arrivalTime': '2024-08-18T11:35:00',
                          'return_departureTime': '2024-08-24T11:20:00',
                          'return_arrivalTime': '2024-08-24T13:25:00',
                          'go_duration': '120',
                          'return_duration': '125'}

    elif orig == 'Prague' and dest == 'Madrid':
        if depDate == '05/01/2025' and retDate == '09/01/2025':
            return True, {'price': '$155', 'go_stopCount': 0,
                          'return_stopCount': 0,
                          'go_departureTime': '2025-01-05T12:25:00',
                          'go_arrivalTime': '2025-01-05T15:35:00',
                          'return_departureTime': '2025-01-09T16:00:00',
                          'return_arrivalTime': '2025-01-09T18:55:00',
                          'go_duration': '190',
                          'return_duration': '175'}
    elif orig == 'London' and dest == 'Barcelona':
        if depDate == '09/08/2024' and retDate == '15/08/2024':
            return True, {'price': '$118', 'go_stopCount': 0,
                          'return_stopCount': 0,
                          'go_departureTime': '2024-08-09T09:25:00',
                          'go_arrivalTime': '2024-08-09T12:45:00',
                          'return_departureTime': '2024-08-15T12:25:00',
                          'return_arrivalTime': '2024-08-15T13:50:00',
                          'go_duration': '130',
                          'return_duration': '145'}
    return False, {}


def get_all_flights(orig, dest, departureDate, returnDate):
    depDate = departureDate.strftime('%Y-%m-%d %X')
    f = open('data/flightResults.json')

    # returns JSON object as
    # a dictionary
    flights = json.load(f)

    b, special_flight = special_case(orig, dest, departureDate, returnDate)

    if b:
        return [special_flight]

    res = []
    for flight in flights:
        if flights[flight]['data']['itineraries'][0]['legs'][0]['origin']['city'] == orig and \
                flights[flight]['data']['itineraries'][0]['legs'][0]['destination']['city'] == dest:

            for it in flights[flight]['data']['itineraries']:
                if departureDate.strftime('%Y-%m-%d %X')[:10] == it['legs'][0]['departure'][:10] \
                            and returnDate.strftime('%Y-%m-%d %X')[:10] == it['legs'][1]['departure'][:10]:

                    flightPossible = {'price': it['price']['formatted'], 'go_stopCount': it['legs'][0]['stopCount'],
                                      'return_stopCount': it['legs'][1]['stopCount'],
                                      'go_departureTime': it['legs'][0]['departure'],
                                      'go_arrivalTime': it['legs'][0]['arrival'],
                                      'return_departureTime': it['legs'][1]['departure'],
                                      'return_arrivalTime': it['legs'][1]['arrival'],
                                      'go_duration': it['legs'][0]['durationInMinutes'],
                                      'return_duration': it['legs'][1]['durationInMinutes']}

                    res.append(flightPossible)
    return res


def best_flight(orig, dest, departureDate, returnDate):
    flightsPossible = get_all_flights(orig, dest, departureDate, returnDate)
    bestFlight = {}
    minPrice = 9999
    for flight in flightsPossible:
        if int(flight["price"][1:]) < minPrice:
            bestFlight = flight
            minPrice = int(flight["price"][1:])

    return bestFlight


def get_personFlights():
    dataset = pd.read_csv('data/travelperk-subset.csv')
    pers_flights = {}
    for i, row in dataset.iterrows():
        flight = best_flight(row['Departure City'], row['Arrival City'], row['Departure Date'], row['Return Date'])
        pers_flights[row['Traveller Name']] = flight

    return pers_flights


def get_allEvents(city, arrivalTime, departureTime):
    events = pd.read_csv('data/generated_events_500.csv')
    availableEvents = events[(events['City'] == city) & (events['Local Date'] >= arrivalTime[:10]) & (
            events['Local Date'] <= departureTime[:10]) & (events['Local Time Start'] >= arrivalTime[-8:]) & (
                                     events['Local Time End'] <= departureTime[-8:])]

    return availableEvents


def get_allEvents(events,city, date):
    availableEvents = events[
        (events["City"] == city)
        & (events["Local Date"] == date)
    ]

    return availableEvents.to_dict(orient="records")


def get_fake_events():
    events = {
        "Barcelona": [
            {"Event Name": "Concert", "Type": "music", "Date": "2024-02-01"},
            {"Event Name": "Football match", "Type": "sports", "Date": "2024-02-01"},
            {"Event Name": "Food festival", "Type": "food", "Date": "2024-02-01"},
        ],
        "Paris": [
            {"Event Name": "Concert", "Type": "music", "Date": "2024-02-01"},
            {"Event Name": "Football match", "Type": "sports", "Date": "2024-02-01"},
            {"Event Name": "Food festival", "Type": "food", "Date": "2024-02-01"},
        ],
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
        self.flight = None
        self.depart_city = depart_city
        self.arrival_city = arrival_city
        self.depart_date = depart_date
        self.return_date = return_date
        self.events = []
        self.friends = []

    def add_trip_features(self, trips, people_in_cities, events, embedding_model):
        self.flight = best_flight(self.depart_city, self.arrival_city, self.depart_date, self.return_date)
        previous_interests = []
        previous_events = []

        # for each day in the trip
        for day in pd.date_range(self.depart_date, self.return_date):
            # get the people in the city
            people_in_city = people_in_cities[
                (people_in_cities["arrival_city"] == self.arrival_city)
                & (people_in_cities["date"] == day)
            ]["people"].values[0]

            # get the events in the city
            city_events = get_allEvents(events, self.arrival_city, day)
            # todo filter on date and inputs as df
            interests = get_interests_from_df(trips, people_in_city)
            # suggest an event
            event = suggest_event(
                city_events,
                interests,
                embedding_model,
                previous_interests,
                previous_events,
            )
            previous_interests = event[3]
            previous_events = event[4]

            self.friends.append(people_in_city)
            self.events.append(event)
        # choose an event based on the common interests of the people in the city

        pass

    def get_llm_trip_summary(self, tokenizer, answers_model):
        question = """The following are events happening in {city}.
{events}
Make a vacation plan lasting {n_days} days to attend those events. Render it as a markdown and add emojis to make it more engaging."""
        event_names = [f'Day {i+1}: {n[0]["Event Name"]}' for i, n in enumerate(self.events)]
        question = question.format(
            city=self.arrival_city,
            events=str.join("\n", event_names),
            n_days=len(self.events),
        )
        messages = [{"role": "user", "content": question}]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        encodeds = encodeds.to(answers_model.device)
        # with torch.no_grad():
        #     generated_ids = answers_model.module.generate(encodeds, max_new_tokens=100)
        # decoded = tokenizer.decode(generated_ids[0]).split(
        #     "<|end_header_id|>\n\n"
        # )[-1]
        generated_ids = answers_model.generate(
            encodeds, max_new_tokens=800, do_sample=True
        )
        decoded = tokenizer.batch_decode(generated_ids)[0].split(
            "<|end_header_id|>\n\n"
        )[-1]

        return decoded, question


from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import load_checkpoint_and_dispatch
from transformers.integrations import HfDeepSpeedConfig
import deepspeed


def init_llm_models():
    # Use a pipeline as a high-level helper
    login(token=os.environ["HF_TOKEN"])
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    # ds_config = {
    #     "fp16": {"enabled": False},
    #     "bf16": {"enabled": True},
    #     "zero_optimization": {
    #         "stage": 3,
    #         "offload_param": {"device": "cpu", "pin_memory": True},
    #     },
    #     "steps_per_print": 2000,
    #     "wall_clock_breakdown": False,
    #     "train_batch_size": 1,
    # }
    # dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
    answers_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        offload_state_dict=True,
        # llm_int8_enable_fp32_cpu_offload=True,
    )

    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # initialise Deepspeed ZeRO and store only the engine object
    # ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    # ds_engine.module.eval()  # inference

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    # return tokenizer, ds_engine, embedding_model
    return tokenizer, answers_model, embedding_model


from functools import reduce
from collections import Counter
from numpy.random import choice
import numpy as np

def suggest_event(events, user_interests, embedding_model,previous_interests=[],previous_ev_name=[]):
    """given a list of events choose the most appropiate one for that day based on interests of each user,
    Input: list of events, dictionary of users and interests
    """
    events_embeddings = embedding_model.encode([event["Event Name"] for event in events])
    # if there is an intersection of interestes use that
    # intersecton of list of sets

    weighted_interests = dict(Counter([i for l in user_interests.values() for i in l]))
    # subtract previous interests
    for i in previous_interests:
        if i in weighted_interests:
            weighted_interests[i] /= 2

    interests_values = np.array(list(weighted_interests.values()))**4
    p=interests_values / interests_values.sum()
    chosen_interest = choice(
        list(weighted_interests.keys()),
        1,
        p=p
    )[0]

    interests_embeddings = embedding_model.encode(chosen_interest)
    # compute similarity between events and interests
    similarities = util.pytorch_cos_sim(events_embeddings, interests_embeddings)
    # half the similarity if the event has been chosen before
    for i,e in enumerate(events):
        if e["Event Name"] in previous_ev_name:
            similarities[i] /= 2

    similarities = similarities.flatten().numpy()**4

    # pick a random event based on similarity
    event = choice(events, 1, p=similarities / similarities.sum())[0]

    # event_num = similarities.argmax()

    # todo use chatbot after similarity
    # prompt = """Given the user interests
    # {interests}
    # Choose the most appropiate event for the day from the list of events
    # {events}"""

    previous_interests = previous_interests + [chosen_interest]
    return (
        event,
        similarities,
        dict(zip(weighted_interests.keys(),p)),
        previous_interests,
        previous_ev_name + [event["Event Name"]]
    )
