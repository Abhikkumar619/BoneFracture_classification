FROM python:3.8
WORKDIR /service
COPY  . ./
RUN pip install -r requirement.txt
EXPOSE 5001
CMD [ "python3", "app.py" ]
