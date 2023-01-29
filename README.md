# Serverless Machine Learning inference on AWS Lambda with Scikit-Learn and flask front end

This project classifies a given abstract in different subfields of Machine Learning (using labels from arxiv). 
Under the hood there is a Scikit-Learn NLP classification model deployed to AWS Lambda using the Serverless framework.
The abstract is submited unsing a web form that is powered by flask and wsgi. 

by: Andreas Merentitis

![relative path 6](/deploy.png?raw=true "deploy.png")
![relative path 1](/infer.png?raw=true "infer.png")


### Prerequisites

The project requires a pre-trained keras model on a certain subset of arxiv. 
For downloading the required data please check the repository arxiv_collector
or use the scripts provided in the data folder. For example from the root 
directory:

```
python data/collect.py -c stat 2015ml.h5 -start 2015-01-01 -end 2015-12-31
```

The file train_local_data.py can be used to train the model using these data.


#### Setup serverless

```  
curl -fsSL https://deb.nodesource.com/setup_14.x | sudo -E bash -

sudo apt-get install -y nodejs

sudo npm install -g serverless

sudo serverless plugin install -n serverless-python-requirements

sudo serverless plugin install -n serverless-wsgi

pip install -r requirements.txt

```
#### Setup AWS credentials

Make sure you have the AWS access key and secret keys setup locally, following this video [here](https://www.youtube.com/watch?v=KngM5bfpttA)

### Download the code locally

```  
serverless create --template-url https://github.com/AndreasMerentitis/SkLambda-arxiv-keras --path SkLambda-arxiv
```

### Update S3 bucket to unique name
In serverless.yml:
```  
  environment:
    BUCKET: <your_unique_bucket_name> 
```

### Check the file syntax for any files changed 
```
pyflakes infer.py

```
We can ignore the warning about not using 'unzip_requirements' as its needed to set the requirements for lamda 

### Train the model from scratch

```
source activate py37

python local_train_new_model.py 

```


### Deploy to the cloud  


```
cd SkLambda-arxiv

npm install

serverless deploy --stage dev

aws s3 cp finalized_vectorizer.sav s3://serverless-ml-1/ --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers

aws s3 cp finalized_model.sav s3://serverless-ml-1/ --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers

curl -X GET https://syrqz8iwfd.execute-api.eu-west-1.amazonaws.com/dev

curl -X GET https://syrqz8iwfd.execute-api.eu-west-1.amazonaws.com/dev/{proxy+}

```

### Clean up (remove deployment) 


```
aws s3 rm s3://serverless-ml-1 --recursive

serverless remove --stage dev 
```

# Using data and extending the basic idea from these sources:
* https://github.com/mikepm35/TfLambdaDemo
* https://medium.com/@mike.p.moritz/running-tensorflow-on-aws-lambda-using-serverless-5acf20e00033
* https://github.com/wingkitlee0/arxiv_explore
* https://github.com/wingkitlee0/arxiv_collector
* https://github.com/alexdebrie/serverless-flask
* https://www.fernandomc.com/posts/developing-flask-based-serverless-framework-apis/
* https://pypi.org/project/serverless-wsgi/
* https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
* https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html#sklearn.calibration.calibration_curve









