ACTIVATE THE ENVIRONMENT
source venv/bin/activate

RUN THE APP
uvicorn main:app --host 0.0.0.0 --port 8001 --reload