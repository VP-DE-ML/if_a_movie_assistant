from pyspark import SparkContext
from pyspark.sql import SparkSession
import pandas as pd

from input import env


class SparkDB:

    def __init__(self):
        self.app_configuration = "spark.some.config.option"
        self.app_name = "Movie Assistant"
        self.table_name = "MovieTable"
        self.df_movie_output = ""
        self.session_status = "Closed"
        env.setup_environment()

    def create_session(self):
        if self.session_status == "Closed":
            spark = SparkSession.builder.appName(self.app_name + "_session").config(self.app_configuration, "").getOrCreate()
            self.session_status = "Created"
            print("Spark Session: " + self.session_status)
        df = spark.read.option("delimiter", ",").option("header", "true").csv('../input/datasets_2745_4700_movies.csv')
        df.createOrReplaceTempView(self.table_name)
        # spark.sql("select * from MovieTable").show(10)
        # df.printSchema()
        # print(df.count())
        return spark

    def stop_spark_session(self, spark):
        self.session_status = "Closed"
        spark.stop()
        print("Spark Session: " + self.session_status)

    def create_context(self):
        sc = SparkContext(master="local", appName=self.app_name + "_sc")
        return sc

    def recommend_movie_by(self, spark, query_filter):
        query = "select name, star from " + self.table_name + " where " + query_filter + " order by rating desc, score desc limit 5"
        print(query)
        df_movie_output = spark.sql(query)
        spark_pandas_output = df_movie_output.toPandas()
        pandas_pandas_output = pd.DataFrame(spark_pandas_output).values.tolist()
        counter = 1
        output_string = "Recommending first 5 movies: \n"
        for row in pandas_pandas_output:
            output_string = output_string + "\n " + str(counter) + " " + row[0] + " with " + row[1]
            counter += 1
        return output_string

    def retrieve_movie_information(self, spark, query_filter):
        dfMovieOutput = spark.sql("select * from " + self.table_name + " where " + query_filter)
        return dfMovieOutput
