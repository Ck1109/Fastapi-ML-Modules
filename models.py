import pandas as pd
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, Request, HTTPException, status,Depends, File, UploadFile ,Response
import uvicorn
import numpy as np
from pydantic import BaseModel
import time
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import RequestValidationError
import secrets
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import Stats
from collections import defaultdict
import sweetviz as sw
import pandas_profiling
import tempfile


usernames = list()

class mode(BaseModel):
    texts: str

app = FastAPI()
security = HTTPBasic()
data=[]


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


'''
@app.post("/user/forecasting")
def post_forecasting(dependent_col,Date_col,data1:dict,username: str = Depends(get_current_username)):
    data1=pd.DataFrame(data1)
    data1=Stats.forecasting(data1,dependent_col,Date_col).to_dict('list')
    print(Stats.forecasting(data1,dependent_col,Date_col).to_dict('list'))

    return {

        "data":data1

    }
'''
@app.post("/user/Model_building")
async def post_model_building(dependent_col,Date_col,file: UploadFile = File(...),username: str = Depends(get_current_username)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Load the JSON file into a pandas DataFrame
            data1 = pd.read_csv(file_path)
         #   data1=pd.DataFrame(data1)
            Stats.model_forecasting(data1,dependent_col,Date_col)
          #  print(Stats.model_forecasting(data1,dependent_col,Date_col).to_dict('list'))

            return 'Done...!!!'
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/forecasting1")
async def post_forecasting1(dependent_col,Date_col,file: UploadFile = File(...),username: str = Depends(get_current_username)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Load the JSON file into a pandas DataFrame
            data1 = pd.read_csv(file_path)




           # data1=pd.DataFrame(data1)
            data1=Stats.forecasting1(data1,dependent_col,Date_col).to_dict('list')
            print(Stats.forecasting1(data1,dependent_col,Date_col).to_dict('list'))

            return {

                "data":data1

             }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/Seasonality")
async def post_Seasonality(dependent_col,Date_col,file: UploadFile = File(...),username: str = Depends(get_current_username)):
    #data1=pd.DataFrame(data1)
   # data1=Stats.forecasting1(data1,dependent_col,Date_col).to_dict('list')
   # print(Stats.forecasting1(data1,dependent_col,Date_col).to_dict('list'))
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Load the JSON file into a pandas DataFrame
            data1 = pd.read_json(file_path)

            output1=Stats.seasonality(data1,Date_col,dependent_col).to_dict('list')

            print(Stats.seasonality(data1,Date_col,dependent_col).to_dict('list'))

            return {'data':output1}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




'''@app.post("/user/eda")
def post_eda(data1:dict,username: str = Depends(get_current_username)):
    data=pd.DataFrame(data1)
    my_report = sw.analyze(data)
    name='EdaReport.html'
    my_report.show_html(name)
    file_path ="./API_Python/"

    return FileResponse(file_path+name, media_type='text/html', filename=name)
'''

'''@app.post("/uploadfile/")
async def create_upload_file(data1:dict,username: str = Depends(get_current_username)):
    data=pd.DataFrame(data1)
 #   report = sw.analyze(data)
  #  report.show_html(filepath='report.html')
   # return FileResponse('report.html', media_type='text/html')


    report = sw.analyze(data)
    report.show_html(filepath='report.html')
    return FileResponse('report.html', media_type='text/html')'''

@app.post("/generate_report")
async def generate_report(file: UploadFile = File(...),username: str = Depends(get_current_username)):
    # Save the uploaded file to a temporary directory
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Load the JSON file into a pandas DataFrame
            data = pd.read_csv(file_path)

            # Generate the pandas profiling report
            report = pandas_profiling.ProfileReport(data)

            # Save the report as an HTML file to a temporary directory
            html_file_path = f"{temp_dir}/report.html"
            report.to_file(html_file_path)

            # Return the HTML file as a downloadable file
            with open(html_file_path, "rb") as buffer:
                contents = buffer.read()

            response = Response(content=contents)
            response.headers["Content-Disposition"] = f"attachment; filename=report.html"
            return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

'''
@app.post("/user/Normalize1")
async def post_Normalize1(col:list[str],file: UploadFile = File(...),username: str = Depends(get_current_username)):
    if file.content_type != "application/json":
        raise HTTPException(status_code=400, detail="Only JSON files are supported")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Load the JSON file into a pandas DataFrame
            data1 = pd.read_json(file_path)
            col=col[0].split(',')
            data=[]
            # data.append(data1)

            #data1.reset_index(inplace=True)
            #data1.drop(columns='index', inplace=True)
            #output=Stats.Normalize1(data1,list(col))
            #output=pd.DataFrame(output)
            #output=pd.concat([data1,output])
     #       for i in range(0,len(output)):
      #          data.append(output[i].item())
           # output=output.to_dict('list')

            return {

                "data":col

            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

'''

@app.post("/user/Normalize2")
async def post_Normalize2(col:list[str],file: UploadFile = File(...),username: str = Depends(get_current_username)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            col=list(col[0].split(','))
            # Load the JSON file into a pandas DataFrame
            data1 = pd.read_csv(file_path)
           # ind=[]
            #for i in range(len(data1)):
             #   ind.append(i)
            #data1=pd.DataFrame(data1)
            data1.select_dtypes(include='object').fillna('', inplace=True)
            data1.select_dtypes(include='number').fillna(0, inplace=True)
            data1=data1.fillna('')




            data=[]
            # data.append(data1)

         #   data1.reset_index(inplace=True)
         #   data1.drop(columns='index', inplace=True)
          #  data1=data1.fillna('')
            #col=list(data1.columns)

            #col1=list(data1[col].columns)

            output=Stats.Normalize1(data1.head(),col)

           # output=Stats.Normalize1(data1,col1)




            return jsonable_encoder(output)#jsonable_encoder(output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user/Kmeans_cluster")
async def post_Kmeans_cluster(X:str,Y:str,file: UploadFile = File(...),username: str = Depends(get_current_username)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Load the JSON file into a pandas DataFrame
            data1 = pd.read_csv(file_path)
            X=list(data1[X].head(5000).values)
            Y=list(data1[Y].head(5000).values)
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



@app.post("/user/find_outliers_IQR")
async def post_find_outliers_IQR(col: str,file: UploadFile = File(...), username: str = Depends(get_current_username)):
    data = []
    data.append(data1)
    if file.content_type != "text/csv":
               raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Load the JSON file into a pandas DataFrame
            data1 = pd.read_csv(file_path)
            data1=list(data1[col].values)
        return {
            "find_outliers_IQR": Stats.find_outliers_IQR(data1)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/user/Replace_Outliers")
async def post_Replace_Outliers(col:str,file: UploadFile = File(...),username: str = Depends(get_current_username)):
#    data=[]

    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Load the JSON file into a pandas DataFrame
            data1 = pd.read_csv(file_path)
            data1=list(data1[col].head(5000).values)
           # data.append(data1)
            output=np.array(Stats.Replace_Outliers(data1))
#            for i in range(0,len(output)):
#                data.append(output[0])

            return {

                "data":list(output.flatten())

            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#print(post_Replace_Outliers([27.1, 27.2, 26.9, 30, 22, 28, 25, 23, 27, 25, 101, 0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




'''@app.post("/user/Box")
def post_box(data1: list[float], username: str = Depends(get_current_username)):
    data = []
    data.append(data1)
    return {

        "Q1": Stats.box(data1)[0],
        "Q3":Stats.box(data1)[1],
        "outliers":Stats.box(data1)[2],
        "IQR": Stats.box(data1)[3],
        "lower_bound": Stats.box(data1)[4],
        "upper_bound": Stats.box(data1)[5]

}'''





@app.post("/user/LinearRegression")
async def post_LinearRegression(x: str, y: str ,file: UploadFile = File(...), username: str = Depends(get_current_username)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Load the JSON file into a pandas DataFrame
            data1 = pd.read_csv(file_path)
            x=data1[x].values.reshape(-1,1)
            y=data1[y].values
    #      y=[]
    #     y.append(y)

    #    x=[]
    #   x.append(x)


            return {
        "Y_intercept": Stats.Y_intercept(np.array(x, dtype=object).reshape(-1, 1), np.array(y, dtype=object)),
        "Slope": (Stats.Slope(np.array(x).reshape(-1, 1), np.array(y)))[0]
    }
#            return {
#           "Y_intercept": Stats.Y_intercept(x,y).flatten(),
#            "Slope": Stats.Slope(x,y).flatten()
#        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/LabelEncoding")
async def post_LabelEncoding(cols: list[str],file: UploadFile = File(...), username: str = Depends(get_current_username)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            cols=cols[0].split(',')
            # Load the JSON file into a pandas DataFrame
            data1 = pd.read_csv(file_path)
            data1=data1.fillna(0)
    #        data = []
            #data.append(data1)
            output = Stats.LabelEncoding(data1,cols)

    #        for i in range(0, len(output)):

     #           data.append(output[i].item())

            return output.head(1000).to_dict('list')
 #           data1.head(1000).to_dict()


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))












@app.post('/user/standard_deviation')
def standard_deviation(data1:list[int],username: str = Depends(get_current_username)):
    try :
        data=[]
        data.append(data1)
        output=Stats.standard_deviation(data1)

        output=e

        return {
        'standard_deviation':output
    }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/user/variance')
def variance(data1:list[int],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        output=Stats.variance(data)

        output=e
        return {
        'variance':output
    }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/user/mean')
def mean(data1:list[int],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        output=Stats.mean(data)
  #  except Exception as e:
        output=e
        return {
        'mean':output
    }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/user/median")
def median(data1:list[int],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        return {

            "Median":Stats.median(data)

        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post('/user/floor')
def floor(data1:list[float],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        return {
            'floor':Stats.floor(data1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post('/user/ceil')
def ceil(data1:list[float],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        return {
            'ceil':Stats.ceil(data1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/user/fix')
def fix(data1:list[float],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        return {
            'fix':Stats.fix(data1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user/nanprod")
def nanprod(data1:list[float],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        return {

            "nanprod":Stats.nanprod(data1)

        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/nansum")
def nansum(data1:list[float],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        return {

            "nansum":Stats.nansum(data1)

        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/nansum")
def nansum(data1:list[float],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        return {

            "nansum":Stats.nansum(data1)

        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/nancumsum")
def nancumsum(data1:list[float],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        return {

            "nancumsum":Stats.nancumsum(data1)

        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/exp")
def exp(data1:list[float],username: str = Depends(get_current_username)):
    try:

        data=[]
        data.append(data1)
        return {

            "exp":Stats.exp(data1)

        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user/expm1")
def expm1(data1:list[float],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        return {

            "expm1":Stats.expm1(data1)

        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/exp2")
def exp2(data1:list[float],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        return {

            "exp2":Stats.exp2(data1)

        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/log")
def log(data1:list[float],username: str = Depends(get_current_username)):

    try:
        data=[]
        data.append(data1)
        return {

            "log":Stats.log(data1)

        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/log10")
def log10(data1:list[float],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        return {

            "log10":Stats.log10(data1)

        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/user/log2")
def log2(data1:list[float],username: str = Depends(get_current_username)):
    try:
        data=[]
        data.append(data1)
        return {

            "log2":Stats.log2(data1)

        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user/LinearRegression")
def LinearRegression(x: list[float], y: list[float], username: str = Depends(get_current_username)):
    #      y=[]
    #     y.append(y)

    #    x=[]
    #   x.append(x)
    try:
        return {
            "Y_intercept": Stats.Y_intercept(np.array(x, dtype=object).reshape(-1, 1), np.array(y, dtype=object)),
            "Slope": (Stats.Slope(np.array(x).reshape(-1, 1), np.array(y)))[0]


        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))