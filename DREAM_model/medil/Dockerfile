FROM python:3.7@sha256:d64c9b3d97e57a8df37ec1c4db9be638660efe0d8703f9679cc1254358a9136d

# Install dependencies
RUN pip install pandas \
  numpy \
  datetime \
  xgboost \
  sklearn

# Move to the app folder
RUN mkdir /app
WORKDIR /app

# Copy our python program for training and inference
COPY ./arielize_ehr.py .
COPY ./train.py .
COPY ./infer.py .

# Copy Bash scripts expected by the IT infrastructure of the EHR DREAM Challenge
COPY ./train.sh .
COPY ./infer.sh .

# Add executable permission to Bash scripts
RUN chmod +x train.sh infer.sh
