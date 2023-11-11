# ReactAble Python ML Backend

This repo is a proxy to access python ML functions via our main nest js backend.


## Usage

- server starts at `http://localhost:5000`
- endpoint 1: `/sim`
    - given image checks similarity using ML
        - returns a similarity score double from 0 to 1 inclusive
    - post endpoint
    - encoding must be 
        - jpeg
        - png
        - jpg
    - body:
        ```
        {
            "image": <insert base64 string>;
        }
        ```
    - returns
        - on sucess:
            ```
            {
                "similarity_score": string;
            }
            ```
        - on error:
            ```
            {
                "message": <some error message>;
            }
            ```
- endpoint 2: `/detection`
    - given image checks if stairs, ramps, and guard rails are present
    - post endpoint
    - encoding must be 
        - jpeg
        - png
        - jpg
    - body:
        ```
        {
            "image": <insert base64 string>
        }
        ```
    - returns
        - on sucess:
            ```
            {
                "stairs": boolean;
                "ramps": boolean;
                "guard_rails": boolean;
            }
            ```
        - on error:
            ```
            {
                "message": <some error message>;
            }
            ```
## Setup

Download the weights for object detection
```
https://drive.google.com/file/d/1a-cHfvOgHT_W8empBxhk9NvY-7g71Q4p/view?usp=sharing
```

put it in root directory of     `main.py`

start the server using uvicorn
```
uvicorn main:app --workers 1 --host 0.0.0.0 --port 5000
```

### Docker start

- make sure you have gpu at index 0 available
- make sure weights are downloaded
- make sure docker installed
- start as such
```
docker compose up -d
```