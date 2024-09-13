tree:
    git ls-files | tree --fromfile --dirsfirst

venv:
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt
    venv/bin/pip install -r requirements-dev.txt

pylint:
    pylint poema_top/ --max-line-length 120 || :

mypy:
    mypy poema_top/__main__.py --strict || :

pytest:
    pytest --setup-show

run:
    python3 -m poema_top
