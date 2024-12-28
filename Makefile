
doya:
	@PYTHONPATH=${PWD} python Experimental/Doya.py | tee -a log.Doya

install:
	@command -v python3.12 >/dev/null 2>&1 || { echo "Python 3.12 is not installed. Please install it." >&2; exit 1; }
	@test -d .venv || python3.12 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt

reset:
	@rm -rf .venv
