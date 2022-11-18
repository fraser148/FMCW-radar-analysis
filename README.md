# Project

Run virtual environment. From the source directory run:

```bash
env/Scripts/activate
```

Then you can install packages via `pip`

To update the requirements.txt:

```bash
pip freeze > requirements.txt
```

## Created venv via

```bash
py -m venv env
env/Scripts/activate
py -m pip install -r requirements.txt
```
