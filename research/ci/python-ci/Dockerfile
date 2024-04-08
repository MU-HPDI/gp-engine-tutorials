FROM python:3.8

RUN mkdir -p /app
COPY / /app

RUN pip install --upgrade pip
RUN pip install /app

RUN chmod +x /app/bin/*
ENV PATH=/app/bin:$PATH

RUN mkdir -p /data
RUN mv /app/airplane00.png /data

CMD /bin/bash