FROM python:3.7
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY serve /usr/local/bin
RUN chmod +x /usr/local/bin/serve
EXPOSE 8080