FROM tianchi2
COPY . /workspace
WORKDIR /workspace/code
CMD ["sh", "run.sh"]
