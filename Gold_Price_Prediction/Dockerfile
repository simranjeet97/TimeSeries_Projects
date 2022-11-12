FROM python:3.6
RUN mkdir /app
WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install -r requirements_proph.txt
RUN python -m pip install -r requirements.txt
CMD ["python", "api/app.py"]