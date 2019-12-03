# Restaurant Setup success prediction using Yelp Dataset

In order to run this project, you need to have the following prerequisites:
* pyenv to install 3.5.7 (Google AI only supports python 3.5.7)

`brew install pyenv`

`pyenv install Python 3.5`

Our code uses `mysqlalchemy` to save checkpoints into MySql DB (Amazon RDS)
So you need to have mysqlclient installed in your machine

`brew install mysql`

Our initial pre-processing also requires spark to be running on your machine.
Instructions to install spark can be found in pre-processing folder of this repository

Make sure you install all the dependencies from `requirements.txt` file.
