

## Installation

Requires [poetry](https://python-poetry.org/docs/).

```bash
poetry install
```


## Run

You need to place a trained model as `./model/model.state`.
Contact me to get one.

```bash
poetry shell
uvicorn main:app --reload
```

checkout [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
