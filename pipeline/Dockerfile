FROM python:3.9

RUN pip3 install poetry

RUN mkdir /app
WORKDIR /app

COPY pyproject.toml poetry.lock ./

# install libraries
RUN poetry config virtualenvs.in-project false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY . .

EXPOSE 9015
CMD ["poetry", "run", "python3", "-m", "uvicorn", "server:app", "--reload", "--port", "9015", "--host", "0.0.0.0", "--loop", "uvloop"]
