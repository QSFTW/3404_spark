import csv
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
from pyspark.sql.window import Window
from pyspark.sql.functions import *

file_path = 's3://qsftw-bucket/assignment_data_files/'
large_file_path = 's3://usyddata3404/'
user_specified_year = '1995'
user_defined_country = "'United States'"
size = 'tiny'

# local file path configured for testing locally
aircraft_path = file_path+'ontimeperformance_aircrafts.csv'
airline_path = file_path+'ontimeperformance_airlines.csv'
airport_path = file_path+'ontimeperformance_airports.csv'

# Alter flight_path to assess performace later for Qestion 2
flight_path = file_path + 'ontimeperformance_flights_' + size+'.csv'

# Start Spark Session and load data to the form of Spark Dataframe
spark = SparkSession \
    .builder \
    .getOrCreate()

aircraft_df_raw = spark.read.csv(aircraft_path, header=True) \
    .select("tailnum", "manufacturer", "model") \
    .na.drop() \

airline_df = spark.read.csv(airline_path, header=True).cache()

flight_schema = StructType([
    StructField("flight_id", StringType(), True),
    StructField("carrier_code", StringType(), True),
    StructField("flight_number", StringType(), True),
    StructField("flight_date", DateType(), True),
    StructField("origin", StringType(), True),
    StructField("destination", StringType(), True),
    StructField("tail_number", StringType(), True),
    StructField("scheduled_depature_time", StringType(), True),
    StructField("scheduled_arrival_time", StringType(), True),
    StructField("actual_departure_time", StringType(), True),
    StructField("actual_arrival_time", StringType(), True),
    StructField("distance", IntegerType(), True)
])

flight_df = spark.read.csv(flight_path, header=True, schema=flight_schema) \
    .select("carrier_code", "flight_date", "tail_number",
                            "scheduled_depature_time", "actual_departure_time") \
    .cache()

# Further preprocess the aircraft_df to extract the first 3 digits from 'model' attribute


def get_digits(model):
    '''
    Get the first 3 digit from the model string
    If it has not digit, return the first 3 character
    '''
    count = 0
    model_num = ''
    for char in list(model):
        if char.isdigit():
            model_num += char
            count += 1
        if count == 3:
            break
    return model_num if model_num != '' else model[:3]


get_digits_udf = udf(lambda x: get_digits(x))

aircraft_df = aircraft_df_raw.withColumn(
    "model_num", get_digits_udf("model")).drop("model").cache()

# Task 1: Top-3 Cessna Models
# Filter out CESSNA aircraft from cessna_aircraft_df
cessna_aircraft_df = aircraft_df.where("manufacturer == 'CESSNA'")

# Inner Join flight flight_cessna_df and cessna_aircraft_df
# Broadcast join is used for performace as cessna_aircraft_df is a very small set (12 rows in the proviced dataset)
flight_cessna_df = flight_df \
    .join(broadcast(cessna_aircraft_df),
          cessna_aircraft_df.tailnum == flight_df.tail_number) \
    .select("model_num")

top3 = flight_cessna_df \
    .groupBy("model_num") \
    .count() \
    .orderBy(desc("count")) \
    .limit(3) \
    .rdd.map(lambda row: "Cassna " + row['model_num'] + "\t" + str(row[1])) \
    .saveAsTextFile(file_path+'task1_' + size)

# Task 2: Average Departure Delay
# Transform columns in flight_df to get year, and scheduled_depature_time, actual_departure_time in minutes
## User Defined Function to get sum of minutes from start of the day to departure time
def get_minutes(dep_str, act_str):
    h,m,s = dep_str.split(":")
    h2,m2,s2 = act_str.split(":")
    return int(h2)*60 + int(m2) - int(h)*60 - int(m)


get_minutes_udf = udf(lambda x,y:get_minutes(x,y), IntegerType())

# c
flight_year_df = flight_df.select("carrier_code", "flight_date", "scheduled_depature_time", "actual_departure_time") \
                            .withColumn("year",year("flight_date")) \
                            .where("year == " + user_specified_year) \
                            .where("actual_departure_time is not null") \
                            .withColumn("lateness", get_minutes_udf("scheduled_depature_time", "actual_departure_time")) \
                            .drop("flight_date", "scheduled_depature_time", "actual_departure_time")

delayed_flight_df = flight_year_df.where("lateness > 0")

# Filter out US airlines
airline_us_df = airline_df.where("country == 'United States'").drop("country")
# broadcast join flight and US airlines
airline_flight_df = delayed_flight_df.join(broadcast(airline_us_df), "carrier_code").cache()


## Get summary statistics
lateness_summary = airline_flight_df \
                    .groupBy("name") \
                    .agg(count("lateness").alias("count"),
                         avg("lateness").alias("average"),
                         min("lateness").alias("minimum"),
                         max("lateness").alias("maximum")) \
                    .coalesce(1) \
                    .orderBy("name") \
                    .rdd.map(lambda row : row['name']+'\t'+str(row['count'])+ \
                                         '\t'+str(row['average'])+'\t'+ \
                                         str(row['minimum'])+'\t'+str(row['maximum'])) \
                    .saveAsTextFile(file_path+"task2_"+size)
# Task 3: Most Popular Aircraft Types
airline_selected_df = airline_df.where("country ==" + user_defined_country).drop("country")
# join flight data with selected airlines and then join with aircrafts
flight_airline_aircraft_df = flight_df \
                                .join(broadcast(airline_selected_df), "carrier_code") \
                                .join(broadcast(aircraft_df), 
                                      flight_df.tail_number == aircraft_df.tailnum) \
                                .select("name","tailnum", "manufacturer", "model_num")

rank_window = Window() \
                .partitionBy("name") \
                .orderBy(desc("count"))
# count number of flights per partition of aircraft model_num
# rank the model_num based on counts of flights within an airline
# filter out model_num that has rank <= 5
# order by name and rank
# concatenate "manufacturer" and "model_num" columns for convenience later
popular_aircraft_result = flight_airline_aircraft_df \
                            .groupBy("name", "manufacturer", "model_num") \
                            .count() \
                            .withColumn("rank", rank().over(rank_window)) \
                            .where("rank <=5") \
                            .orderBy("name", "rank") \
                            .select("name",concat("manufacturer", lit(' '),"model_num").alias("aircraft_type")) \
                            .collect()

# Deal with the situation where some airline has less than 5 aircraft model 
# while some may have ties in the top5

airline_list = []
aircraft_dict = {}
for row in popular_aircraft_result:
    if row["name"] not in airline_list:
        airline_list.append(row["name"])
    aircraft_dict.setdefault(row["name"],[]).append(row["aircraft_type"])

result = []
for row in airline_list:
    line = row+'\t'
    list_of_aircrafts = aircraft_dict[row]
    if(len(list_of_aircrafts)>5):
        line += str(list_of_aircrafts[:5]) + "\n"
    else:
        line += str(list_of_aircrafts) + "\n"
    result.append(line)

spark.sparkContext.parallelize(result).coalesce(1).saveAsTextFile(file_path+"task3_" + size)