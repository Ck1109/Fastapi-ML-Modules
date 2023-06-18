import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, status,Depends
import uvicorn
from pydantic import BaseModel
import time
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import RequestValidationError
import secrets
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import Stats
from collections import defaultdict


usernames = list()


class mode(BaseModel):
    texts: str

app = FastAPI()
security = HTTPBasic()
data=[]

'''@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )'''


class Item(BaseModel):
    title: str
    size: int


def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = b"admin"
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = b"admin"
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username




@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response



'''@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(content={"error": exc.detail}, status_code=exc.status_code)
'''



@app.post("/user/admin")
def getinformation(username: str = Depends(get_current_username)):
    return {
        "data": cust.to_json()
    }


@app.post('/user/standard_deviation')
def standard_deviation(data1:list[int],username: str = Depends(get_current_username)):
    try :
        data=[]
        data.append(data1)
        output=Stats.standard_deviation(data1)
    except Exception as e:
        output=e

    return {
        'standard_deviation':output
    }

@app.post('/user/variance')
def variance(data1:list[int],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        output=Stats.variance(data)
    except Exception as e:
        output=e
    return {
        'variance':output
    }


@app.post('/user/mean')
def mean(data1:list[int],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        output=Stats.mean(data)
    except Exception as e:
        output=e
    return {
        'mean':output
    }


@app.post("/user/median")
def median(data1:list[int],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "Median":Stats.median(data)

    }



@app.post('/user/floor')
def floor(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {
        'floor':Stats.floor(data1)
    }



@app.post('/user/ceil')
def ceil(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {
        'ceil':Stats.ceil(data1)
    }


@app.post('/user/fix')
def fix(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {
        'fix':Stats.fix(data1)
    }


@app.post("/user/nanprod")
def nanprod(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "nanprod":Stats.nanprod(data1)

    }


@app.post("/user/nansum")
def nansum(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "nansum":Stats.nansum(data1)

    }



@app.post("/user/nansum")
def nansum(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "nansum":Stats.nansum(data1)

    }


@app.post("/user/nancumsum")
def nancumsum(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "nancumsum":Stats.nancumsum(data1)

    }


@app.post("/user/exp")
def exp(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "exp":Stats.exp(data1)

    }



@app.post("/user/expm1")
def expm1(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "expm1":Stats.expm1(data1)

    }


@app.post("/user/exp2")
def exp2(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "exp2":Stats.exp2(data1)

    }


@app.post("/user/log")
def log(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "log":Stats.log(data1)

    }


@app.post("/user/log10")
def log10(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "log10":Stats.log10(data1)

    }




@app.post("/user/log2")
def log2(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "log2":Stats.log2(data1)

    }



@app.post("/user/LinearRegression")
def LinearRegression(x: list[float], y: list[float], username: str = Depends(get_current_username)):
    #      y=[]
    #     y.append(y)

    #    x=[]
    #   x.append(x)

    return {
        "Y_intercept": Stats.Y_intercept(np.array(x, dtype=object).reshape(-1, 1), np.array(y, dtype=object)),
        "Slope": (Stats.Slope(np.array(x).reshape(-1, 1), np.array(y)))[0]


    }


'''

@app.post("/user/interquartile_range")
def interquartile_range(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "interquartile_range":Stats.interquartile_range(data1)

    }


@app.post("/user/Q1")
def Q1(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "Q1":Stats.Q1(data1)

    }


@app.post("/user/Q3")
def Q3(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "Q3":Stats.Q3(data1)

    }

@app.post("/user/OutlierLimits")
def OutlierLimits(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
    data.append(data1)
    return {

        "OutlierLimits":Stats.OutlierLimits(data1)

    }


@app.post("/user/find_outliers_IQR")
def find_outliers_IQR(data1: list[float], username: str = Depends(get_current_username)):
    data = []
    data.append(data1)
    return {

        "find_outliers_IQR": Stats.find_outliers_IQR(data1)

}

@app.post("/user/Box")
def box(data1: list[float], username: str = Depends(get_current_username)):
    data = []
    data.append(data1)
    return {

        "Q1": Stats.box(data1)[0],
        "Q3":Stats.box(data1)[1],
        "outliers":Stats.box(data1)[2],
        "IQR": Stats.box(data1)[3],
        "lower_bound": Stats.box(data1)[4],
        "upper_bound": Stats.box(data1)[5]

}





@app.post("/user/LabelEncoding")
def LabelEncoding(data1:list[str],username: str = Depends(get_current_username)):
    data=[]
   # data.append(data1)
    output=Stats.LabelEncoding(data1)

    for i in range(0,len(output)):
        data.append(output[i].item())

    return data




@app.post("/user/Normalize")
def Normalize(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
   # data.append(data1)
    output=Stats.Normalize(data1)
    for i in range(0,len(output)):
        data.append(output[i].item())

    return {

        "data":data

    }

@app.post("/user/Normalize1")
def Normalize1(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
   # data.append(data1)
    output=Stats.Normalize1(data1)
    for i in range(0,len(output)):
        data.append(output[i].item())

    return {

        "data":data

    }


@app.post("/user/Replace_Outliers")
def Replace_Outliers(data1:list[float],username: str = Depends(get_current_username)):
    data=[]
   # data.append(data1)
    output=Stats.Replace_Outliers(data1)
    for i in range(0,len(output)):
        data.append(output[i].item())

    return {

        "data":data

    }

print(Replace_Outliers([27.1, 27.2, 26.9, 30, 22, 28, 25, 23, 27, 25,101,0]))

@app.post("/user/Kmeans_cluster")
def Kmeans_cluster(X:list[float],Y:list[float],username: str = Depends(get_current_username)):
    try:
        data=[]
       # data.append(data1)
        output=Stats.Kmeans_cluster(np.array(X).reshape(-1,1),Y)

        cluster=list(output['Class'].values.flatten())
        X1=list(output["X1"].values.flatten())
        Y1=list(output["Y"].values.flatten())
        X=[]
        Y=[]
        Class=[]
        for i in range(0,len(X1)):
            X.append(X1[i].item())
            Y.append(Y1[i].item())
            Class.append(cluster[i].item())



        return {

            "Class":Class ,
            "X":X ,
            "Y":Y



        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

'''

if __name__ == "__main__":
    uvicorn.run(app,host='127.0.0.1',port=8080)
