// Databricks notebook source
// STARTER CODE - DO NOT EDIT THIS CELL
import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import spark.implicits._
import org.apache.spark.sql.expressions.Window

// COMMAND ----------

// STARTER CODE - DO NOT EDIT THIS CELL
val customSchema = StructType(Array(StructField("lpep_pickup_datetime", StringType, true), StructField("lpep_dropoff_datetime", StringType, true), StructField("PULocationID", IntegerType, true), StructField("DOLocationID", IntegerType, true), StructField("passenger_count", IntegerType, true), StructField("trip_distance", FloatType, true), StructField("fare_amount", FloatType, true), StructField("payment_type", IntegerType, true)))

// COMMAND ----------

// STARTER CODE - YOU CAN LOAD ANY FILE WITH A SIMILAR SYNTAX.
val df = spark.read
   .format("com.databricks.spark.csv")
   .option("header", "true") // Use first line of all files as header
   .option("nullValue", "null")
   .schema(customSchema)
   .load("/FileStore/tables/nyc_tripdata.csv") // the csv file which you want to work with
   .withColumn("pickup_datetime", from_unixtime(unix_timestamp(col("lpep_pickup_datetime"), "MM/dd/yyyy HH:mm")))
   .withColumn("dropoff_datetime", from_unixtime(unix_timestamp(col("lpep_dropoff_datetime"), "MM/dd/yyyy HH:mm")))
   .drop($"lpep_pickup_datetime")
   .drop($"lpep_dropoff_datetime")

// COMMAND ----------

// STARTER CODE - DO NOT EDIT THIS CELL
val lookupSchema = StructType(Array(StructField("LocationID", IntegerType, true), StructField("Borough", StringType, true), StructField("Zone", StringType, true), StructField("service_zone", StringType, true)))

// COMMAND ----------

// LOAD THE "taxi_zone_lookup.csv" FILE SIMILARLY AS ABOVE. CAST ANY COLUMN TO APPROPRIATE DATA TYPE IF NECESSARY.
val taxi_zone_lookup = spark.read
   .format("com.databricks.spark.csv")
   .option("header", "true") // Use first line of all files as header
   .option("nullValue", "null")
   .schema(lookupSchema)
   .load("/FileStore/tables/taxi_zone_lookup.csv") // the csv file which you want to work withs
// ENTER THE CODE BELOW

// COMMAND ----------

// STARTER CODE - DO NOT EDIT THIS CELL
// Some commands that you can use to see your dataframes and results of the operations. You can comment the df.show(5) and uncomment display(df) to see the data differently. You will find these two functions useful in reporting your results.
display(df)
//df.show(5) // view the first 5 rows of the dataframe

// COMMAND ----------

// STARTER CODE - DO NOT EDIT THIS CELL
// Filter the data to only keep the rows where "PULocationID" and the "DOLocationID" are different and the "trip_distance" is strictly greater than 2.0 (>2.0).

// VERY VERY IMPORTANT: ALL THE SUBSEQUENT OPERATIONS MUST BE PERFORMED ON THIS FILTERED DATA

val df_filter = df.filter($"PULocationID" =!= $"DOLocationID" && $"trip_distance" > 2.0)
df_filter.show(5)

// COMMAND ----------

// PART 1a: The top-5 most popular drop locations - "DOLocationID", sorted in descending order - if there is a tie, then one with lower "DOLocationID" gets listed first
// Output Schema: DOLocationID int, number_of_dropoffs int 
var doSorted = df_filter.groupBy($"DOLocationID").agg(count("DOLocationID").alias("number_of_dropoffs"))
doSorted.sort($"number_of_dropoffs".desc).show(5)
//val doSorted = df_filter.groupBy("DOLocationID").count().alias("number_of_dropoffs")
//doSorted.sort($"count".desc).show(5)
// Hint: Checkout the groupBy(), orderBy() and count() functions.

// ENTER THE CODE BELOW


// COMMAND ----------

// PART 1b: The top-5 most popular pickup locations - "PULocationID", sorted in descending order - if there is a tie, then one with lower "PULocationID" gets listed first 
// Output Schema: PULocationID int, number_of_pickups int

// Hint: Code is very similar to part 1a above.
var puSorted = df_filter.groupBy($"PULocationID").agg(count("PULocationID").alias("number_of_pickups"))
puSorted.sort($"number_of_pickups".desc).show(5)
// ENTER THE CODE BELOW


// COMMAND ----------

// PART 2: List the top-3 locations with the maximum overall activity, i.e. sum of all pickups and all dropoffs at that LocationID. In case of a tie, the lower LocationID gets listed first.
// Output Schema: LocationID int, number_activities int
val joinDF = doSorted.join(puSorted,doSorted("DOLocationID") === puSorted("PULocationID"),"inner")
val activitiesDF = joinDF.withColumn("number_activities", col("number_of_pickups")+col("number_of_dropoffs"))
val finalDFColsDropped = activitiesDF.drop("number_of_pickups","number_of_dropoffs","DOLocationID")
val finalDF = finalDFColsDropped.withColumnRenamed("PULocationID","LocationID")
val activityFinalDf = finalDF.select(col("LocationID"),col("number_activities").cast("integer"))
activityFinalDf.sort($"number_activities".desc).show(3)
// Hint: In order to get the result, you may need to perform a join operation between the two dataframes that you created in earlier parts (to come up with the sum of the number of pickups and dropoffs on each location). 

// ENTER THE CODE BELOW


// COMMAND ----------

// PART 3: List all the boroughs in the order of having the highest to lowest number of activities (i.e. sum of all pickups and all dropoffs at that LocationID), along with the total number of activity counts for each borough in NYC during that entire period of time.
// Output Schema: Borough string, total_number_activities int
val boroughDF = finalDF.join(taxi_zone_lookup,finalDF("LocationID") === taxi_zone_lookup("LocationID"),"inner")
val boroughGroupByDF = boroughDF.groupBy("Borough").agg(sum("number_activities"))
val boroughRenamedDf = boroughGroupByDF.withColumnRenamed("sum(number_activities)","total_number_activities")
val boroughFinalDf = boroughRenamedDf.select(col("Borough"),col("total_number_activities").cast("integer"))
boroughFinalDf.sort($"total_number_activities".desc).show()

// Hint: You can use the dataframe obtained from the previous part, and will need to do the join with the 'taxi_zone_lookup' dataframe. Also, checkout the "agg" function applied to a grouped dataframe.

// ENTER THE CODE BELOW


// COMMAND ----------

// PART 4: List the top 2 days of week with the largest number of (daily) average pickups, along with the values of average number of pickups on each of the two days. The day of week should be a string with its full name, for example, "Monday" - not a number 1 or "Mon" instead.
// Output Schema: day_of_week string, avg_count float
val dailyActivityDF = df_filter.withColumn("day_of_week", date_format(col("pickup_datetime"), "EEEE"))
val dailyActivityDF2 = dailyActivityDF.withColumn("pickup_date", to_date(col("pickup_datetime")))
//val partition_result = df.withColumn("trip_rate", sum("trip_rate_rows").over(Window.partitionBy(*partition_cols)))
val dailyActivityGroupByDF = dailyActivityDF2.groupBy("day_of_week","pickup_date").agg(count("pickup_date"))
val dailyActivityGroupByFinalDF = dailyActivityGroupByDF.withColumnRenamed("count(pickup_date)","count_daily_activities")
val dayGroupBy = dailyActivityGroupByFinalDF.groupBy("day_of_week").agg(mean("count_daily_activities"))
val dayGroupByFinalDF = dayGroupBy.withColumnRenamed("avg(count_daily_activities)","avg_count")
val dayGroupByCastedDF = dayGroupByFinalDF.select(col("day_of_week"), col("avg_count").cast("float")) 
dayGroupByCastedDF.sort($"avg_count".desc).show(2)

// Hint: You may need to group by the "date" (without time stamp - time in the day) first. Checkout "to_date" function.

// ENTER THE CODE BELOW


// COMMAND ----------

// PART 5: For each particular hour of a day (0 to 23, 0 being midnight) - in their order from 0 to 23, find the zone in Brooklyn borough with the LARGEST number of pickups. 
// Output Schema: hour_of_day int, zone string, max_count int
val hourlyPickups = df_filter.join(taxi_zone_lookup, df_filter("PULocationID") === taxi_zone_lookup("LocationID"),"inner")
val hourlyPickups2 = hourlyPickups.withColumn("hour_of_day",hour(col("pickup_datetime")))
val hourlyPickupsBrooklyn = hourlyPickups2.where(hourlyPickups2("Borough") === "Brooklyn")
val hourlyPickupsBrooklynGroupBy =  hourlyPickupsBrooklyn.groupBy("hour_of_day","Zone").agg(count("hour_of_day"))
val hourlyPickupsZone = hourlyPickupsBrooklynGroupBy.where((hourlyPickupsBrooklynGroupBy("Zone") === "Fort Greene"))
val hourlyPickupsBrooklynFinal = hourlyPickupsZone.withColumnRenamed("count(hour_of_day)","max_count")
hourlyPickupsBrooklynFinal.sort($"hour_of_day".asc).show(40)

// Hint: You may need to use "Window" over hour of day, along with "group by" to find the MAXIMUM count of pickups

// ENTER THE CODE BELOW


// COMMAND ----------

// PART 6 - Find which 3 different days of the January, in Manhattan, saw the largest percentage increment in pickups compared to previous day, in the order from largest increment % to smallest increment %. 
// Print the day of month along with the percent CHANGE (can be negative), rounded to 2 decimal places, in number of pickups compared to previous day.
// Output Schema: day int, percent_change float
val hourlyPickups3 = hourlyPickups.withColumn("month",month(col("pickup_datetime")))
val hourlyPickups4 = hourlyPickups3.withColumn("pickup_date", to_date(col("pickup_datetime")))
val hourlyPickupsManhattan = hourlyPickups4.where((hourlyPickups4("Borough") === "Manhattan") && (hourlyPickups4("month")===1) && (year(col("pickup_datetime"))===2019))
val dailyPickupsManhattan = hourlyPickupsManhattan.groupBy("pickup_date").agg(count("pickup_date")).sort($"pickup_date".asc)
val dailyPickupsManhattanFinal = dailyPickupsManhattan.withColumnRenamed("count(pickup_date)","pickup_counts")
val window = Window.partitionBy().orderBy("pickup_date")
val dailyPickupsManhattanLag = dailyPickupsManhattanFinal.withColumn("prev_value", lag("pickup_counts", 1, 0).over(window))
val dailyPickupsManhattanLagFinal = dailyPickupsManhattanLag.withColumn("percent_change", round(((col("pickup_counts")-col("prev_value"))/col("prev_value")*100),2))
val percentChangeManhattanLagFinal = dailyPickupsManhattanLagFinal.select(date_format(col("pickup_date"),"d").alias("day").cast("integer"),col("percent_change").cast("float"))
percentChangeManhattanLagFinal.sort($"percent_change".desc).show(3)

//df.withColumn('Value',abs(df.Value))

// Hint: You might need to use lag function, over a window ordered by day of month.

// ENTER THE CODE BELOW
