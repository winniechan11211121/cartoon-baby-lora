FROM replicate/cog:0.7.2

# Metadata
LABEL author="Your Name"

# Install required Python packages
RUN apt-get update && apt-get install -y libgl1

COPY requirements.txt /code/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /code/requirements.txt

COPY . /code
