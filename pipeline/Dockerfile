FROM python:3.9

RUN pip3 install poetry

RUN mkdir /app
WORKDIR /app

COPY pyproject.toml poetry.lock ./

# install libraries
RUN poetry config virtualenvs.in-project false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY . .

EXPOSE 8080
CMD /app/run_server.sh
