FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

# like CD command in terminal. it will create directory if path is not existed
WORKDIR /root/projects

# Usual terminal commands for installing environment
RUN apt update && apt upgrade -y
RUN apt install python3 python3-pip -y
RUN apt install git -y
# I will use `pipenv` to dynamically controll my environment
# If you want to use `pip install`, just remove `pipenv` and continue with `pip install`
RUN pip install jupyter ipython pipenv

CMD tail -f /dev/null