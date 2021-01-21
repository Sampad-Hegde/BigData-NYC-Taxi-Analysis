# NYC Taxi Analysis using BigData Approch

This Project is developed using `PYTHON3` and Hadoop frame work  (i.e. HDFS, Hadoop, Spark) for using this project Hadoop and Spark should be installed on your Linux Machine.

After installing Hadoop and Spark

***Start the HDFS, YARN, Spark***
using this commands 
```bash 
start-all.sh
```

verify the byfollowing command
```bash
jps
```
Here, is the 2014 Yellow Taxi Dataset [link](https://data.cityofnewyork.us/Transportation/2014-Yellow-Taxi-Trip-Data/gn7m-em8n)

Here, is the 2014 Yellow Taxi Dataset [link](https://data.cityofnewyork.us/Transportation/2014-Green-Taxi-Trip-Data/2np7-5jsg/data)

The following command is for loading Datasets to hdfs storage `Dataset` is a sample folder name

```bash
hdfs dfs -mkdir Dataset
hdfs dfs -put *csv /Dataset
```

create a python Virtual Environment if required

Here `nyc` is  sample name given to virtual environment

```bash
python3 -m venv nyc
source nyc/bin/activate
```

```bash 
pip install -r requirements.txt
```

Run the main python file `Taxi_Analysis.py`
 
####***It will take Lot of time minimum  45 minutes So wait for few minutes***

Output Will be generated in jpg files.