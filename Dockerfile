FROM python:3.10.6-buster
WORKDIR boost_medium

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab
RUN python -m nltk.downloader wordnet

CMD uvicorn medium.api.fast:app --host 0.0.0.0 --port $PORT --reload
