PIP				:= pip
PYTHON			:= python
SETUP_PY		:= setup.py
REQUIREMENTS	:= requirements.txt


.PHONY: all install dep clean


all: dep install


install: $(SETUP_PY) dep
	$(PYTHON) $(SETUP_PY) install


dep: $(REQUIREMENTS)
	$(PIP) install -r $^


clean:
	-rm -rf .eggs .tox build MANIFEST
