FROM python:3.10.6-buster
WORKDIR medium

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

CMD uvicorn medium.api.fast:app --host 0.0.0.0 --port $PORT --reload
