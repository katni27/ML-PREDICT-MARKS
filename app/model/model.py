import joblib
import sys
import requests
import json
import pandas as pd
import numpy as np
import __main__
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
from collections import defaultdict
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve(strict=True).parent

URL = "https://digtrace.susu.ru/integration/DigitalTrace/{}/{}"


def get_grades(journal_id):
    return requests.get(URL.format("GetGradesByJournalId", journal_id)).json()


def tree():
    return defaultdict(tree)


__main__.tree = tree


with open(f"{BASE_DIR}/model.pkl", "rb") as f:
    model = joblib.load(f)


def create_input_data(journal_id):
    result = pd.DataFrame()
    grades = pd.DataFrame(get_grades(journal_id))

    for id in grades["StudentId"].unique():
        result = pd.concat(
            [result, pd.DataFrame({"StudentId": id, "Rating": 0}, index=[0])],
            ignore_index=True,
        )

    electronic_event_numbers = sorted(grades["ElectronicEventNumber"].unique())
    max_electronic_event_number = int(electronic_event_numbers[0] or 0)

    for index, row in grades.iterrows():
        mark = row["Grade"] / row["GradeMax"] * row["Weight"]
        if mark != 0:
            max_electronic_event_number = max(
                int(row["ElectronicEventNumber"] or 0), max_electronic_event_number
            )
        result.loc[result["StudentId"] == row["StudentId"], "Rating"] += mark

    if max_electronic_event_number == 0:
        quantity_complete_tasks = len(grades["EventName"].unique())
    else:
        quantity_complete_tasks = (
            electronic_event_numbers.index(max_electronic_event_number) + 1
        )

    return {"InputData": result, "Quantity": quantity_complete_tasks}


def predict_pipeline(input):
    data = create_input_data(input.Id)
    predictor = model[input.SubjectName][input.CourseNumber][input.Term][
        input.DirectionName
    ][data["Quantity"]]

    if type(predictor) == defaultdict:
        raise HTTPException(status_code=404, detail="Model not found")

    marks = predictor.predict(data["InputData"]["Rating"].to_numpy().reshape(-1, 1))
    data["InputData"] = data["InputData"].drop("Rating", axis=1)
    data["InputData"] = data["InputData"].assign(Mark=marks)

    posts = json.loads('{"items":' + data["InputData"].to_json(orient="records") + "}")

    return posts["items"]
