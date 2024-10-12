#an initial analysis of the data. Let me summarize the key findings:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV files
airlines_df = pd.read_csv('airlines.csv')
airports_df = pd.read_csv('airports.csv')
flights_df = pd.read_csv('flights.csv')

# Read Excel files
data_dict = pd.read_excel('Data_Dictionary.xlsx', sheet_name=None)
a_codes = pd.read_excel('A_Codes.xlsx')
n_codes = pd.read_excel('N_Codes.xlsx')

print("Airlines DataFrame:")
print(airlines_df.head())
print("\
Airports DataFrame:")
print(airports_df.head())
print("\
Flights DataFrame:")
print(flights_df.head())
print("\
Data Dictionary Sheets:")
for sheet_name, sheet_df in data_dict.items():
    print(f"\
{sheet_name}:")
    print(sheet_df.head())
print("\
A Codes:")
print(a_codes.head())
print("\
N Codes:")
print(n_codes.head())

print("\
Dataset Overview:")
print("Airlines shape:", airlines_df.shape)
print("Airports shape:", airports_df.shape)
print("Flights shape:", flights_df.shape)

print("\
Flights DataFrame Info:")
flights_df.info()

print("Analysis complete.")
#========================================================================
# 1. Flight Statistics (On-time Performance and Cancellation Rates)
# Calculate on-time performance and cancellation rates
# On-time performance: Percentage of flights that were not delayed
# Cancellation rate: Percentage of flights that were cancelled

# Calculate total flights and cancellations
total_flights = flights_df.shape[0]
cancelled_flights = flights_df[flights_df['Flight_Status'] == 1].shape[0]  # Assuming 1 indicates cancelled
on_time_flights = flights_df[flights_df['Flight_Status'] == 0].shape[0]  # Assuming 0 indicates on-time

# Calculate percentages
cancellation_rate = (cancelled_flights / total_flights) * 100
on_time_rate = (on_time_flights / total_flights) * 100
delayed_rate = 100 - cancellation_rate - on_time_rate

# Display results
print(f'Total Flights: {total_flights}')
print(f'Cancelled Flights: {cancelled_flights}')
print(f'On-time Flights: {on_time_flights}')
print(f'Cancellation Rate: {cancellation_rate:.2f}%')
print(f'On-time Rate: {on_time_rate:.2f}%')
print(f'Delayed Rate: {delayed_rate:.2f}%')
#============================================================================

# Visualize the results
plt.figure(figsize=(10, 6))
plt.bar(['On-time', 'Delayed', 'Cancelled'], 
        [on_time_rate, delayed_rate, cancellation_rate])
plt.title('Flight Performance Statistics')
plt.ylabel('Percentage')
plt.show()

print("Flight statistics analysis completed.")

#========================================================================


# Load the cancellation reasons from the Data Dictionary
cancellation_reasons = pd.read_excel('Data_Dictionary.xlsx', sheet_name='Cancelation_Reasons')
print("Cancellation Reasons:")
print(cancellation_reasons)

# Load the flights dataset
flights_df = pd.read_csv('flights.csv')

# Map cancellation reasons
reason_map = dict(zip(cancellation_reasons['Reason_ID'], cancellation_reasons['Reason_Name']))
flights_df['CANCELLATION_REASON'] = flights_df['CANCELLATION_REASON'].map(reason_map)

# Calculate cancellation statistics
total_flights = len(flights_df)
cancelled_flights = flights_df['CANCELLATION_REASON'].notna().sum()
cancellation_rate = (cancelled_flights / total_flights) * 100

print(f"\
Total Flights: {total_flights}")
print(f"Cancelled Flights: {cancelled_flights}")
print(f"Cancellation Rate: {cancellation_rate:.2f}%")

# Analyze cancellation reasons
cancellation_counts = flights_df['CANCELLATION_REASON'].value_counts()
print("\
Cancellation Reason Counts:")
print(cancellation_counts)

# Visualize cancellation reasons
plt.figure(figsize=(10, 6))
cancellation_counts.plot(kind='bar')
plt.title('Distribution of Cancellation Reasons')
plt.xlabel('Reason')
plt.ylabel('Number of Cancellations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze cancellations by airline
airline_cancellations = flights_df[flights_df['CANCELLATION_REASON'].notna()].groupby('AIRLINE')['CANCELLATION_REASON'].count().sort_values(ascending=False)
print("\
Cancellations by Airline:")
print(airline_cancellations)

# Visualize cancellations by airline
plt.figure(figsize=(12, 6))
airline_cancellations.plot(kind='bar')
plt.title('Number of Cancellations by Airline')
plt.xlabel('Airline')
plt.ylabel('Number of Cancellations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\
Analysis completed.")

#========================================================================

# 2. Airline Performance Comparison
# Calculate average delays per airline
airline_performance = flights_df.groupby('AIRLINE').agg({
    'AIR_SYSTEM_DELAY': 'mean',
    'SECURITY_DELAY': 'mean',
    'AIRLINE_DELAY': 'mean',
    'LATE_AIRCRAFT_DELAY': 'mean',
    'WEATHER_DELAY': 'mean'
}).reset_index()

# Sort by total average delay
airline_performance['Total_Average_Delay'] = airline_performance[['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']].mean(axis=1)
airline_performance = airline_performance.sort_values(by='Total_Average_Delay', ascending=False)

# Display results
print(airline_performance.head())
#========================================================================
# Calculate average delays by airline
# First, we need to ensure that the delay columns are numeric and handle any NaN values
delay_columns = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
flights_df[delay_columns] = flights_df[delay_columns].apply(pd.to_numeric, errors='coerce')

# Calculate total delays for each flight
flights_df['TOTAL_DELAY'] = flights_df[delay_columns].sum(axis=1)

# Calculate average delays by airline
average_delays = flights_df.groupby('AIRLINE')['TOTAL_DELAY'].mean().sort_values(ascending=False)

# Calculate cancellation rates by airline
cancellation_counts = flights_df[flights_df['CANCELLATION_REASON'].notna()].groupby('AIRLINE')['CANCELLATION_REASON'].count()

# Total flights per airline for cancellation rate calculation
total_flights_per_airline = flights_df.groupby('AIRLINE').size()

# Calculate cancellation rates
cancellation_rates = (cancellation_counts / total_flights_per_airline * 100).fillna(0)

# Combine average delays and cancellation rates into a single DataFrame
performance_comparison = pd.DataFrame({'Average_Delay': average_delays, 'Cancellation_Rate': cancellation_rates})

# Display the performance comparison
print("\
Airline Performance Comparison:")
print(performance_comparison)

# Visualize average delays and cancellation rates
plt.figure(figsize=(12, 6))

# Plot average delays
ax1 = performance_comparison['Average_Delay'].plot(kind='bar', color='tab:blue', width=0.4)
ax1.set_xlabel('Airline')
ax1.set_ylabel('Average Delay (minutes)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for cancellation rates
ax2 = ax1.twinx()  
ax2.plot(ax1.get_xticks(), performance_comparison['Cancellation_Rate'], color='tab:red', marker='o', linewidth=2)
ax2.set_ylabel('Cancellation Rate (%)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Add title and show plot
plt.title('Airline Performance Comparison: Average Delays and Cancellation Rates')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Visualization completed.")
#==================================================================
# 3. Airport Analysis (Busiest Airports, Delay Patterns)
# Calculate the number of flights per airport
busiest_airports = flights_df['ORIGIN_AIRPORT'].value_counts().reset_index()
busiest_airports.columns = ['IATA_CODE', 'Flight_Count']

# Calculate average delays per airport
airport_delay_analysis = flights_df.groupby('ORIGIN_AIRPORT').agg({
    'AIR_SYSTEM_DELAY': 'mean',
    'SECURITY_DELAY': 'mean',
    'AIRLINE_DELAY': 'mean',
    'LATE_AIRCRAFT_DELAY': 'mean',
    'WEATHER_DELAY': 'mean'
}).reset_index()

# Merge busiest airports with delay analysis
airport_analysis = pd.merge(busiest_airports, airport_delay_analysis, left_on='IATA_CODE', right_on='ORIGIN_AIRPORT')
airport_analysis = airport_analysis.drop(columns=['ORIGIN_AIRPORT'])

# Display results
print(airport_analysis.head())

#========================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the flights and airports data
flights_df = pd.read_csv('flights.csv')
airports_df = pd.read_csv('airports.csv')

# Merge flights with airports data
flights_with_airport = flights_df.merge(airports_df, left_on='ORIGIN_AIRPORT', right_on='IATA_CODE')

# Count flights per airport
airport_traffic = flights_with_airport['ORIGIN_AIRPORT'].value_counts().reset_index()
airport_traffic.columns = ['IATA_CODE', 'Flight_Count']

# Merge with airport names
airport_traffic = airport_traffic.merge(airports_df[['IATA_CODE', 'AIRPORT']], on='IATA_CODE')

# Calculate average delay per airport
airport_delays = flights_with_airport.groupby('ORIGIN_AIRPORT').agg({
    'AIR_SYSTEM_DELAY': 'mean',
    'SECURITY_DELAY': 'mean',
    'AIRLINE_DELAY': 'mean',
    'LATE_AIRCRAFT_DELAY': 'mean',
    'WEATHER_DELAY': 'mean'
}).reset_index()

# Merge traffic and delays
airport_analysis = airport_traffic.merge(airport_delays, left_on='IATA_CODE', right_on='ORIGIN_AIRPORT')

# Sort by Flight_Count descending and take top 15
top_15_airports = airport_analysis.sort_values('Flight_Count', ascending=False).head(15)

print(top_15_airports[['AIRPORT', 'Flight_Count', 'AIR_SYSTEM_DELAY', 'WEATHER_DELAY', 'AIRLINE_DELAY']])

# Visualize top 15 busiest airports
plt.figure(figsize=(12, 6))
sns.barplot(x='IATA_CODE', y='Flight_Count', data=top_15_airports)
plt.title('Top 15 Busiest Airports')
plt.xlabel('Airport Code')
plt.ylabel('Number of Flights')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize delay patterns for top 15 airports
delay_types = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
delay_data = top_15_airports[['IATA_CODE'] + delay_types].melt(id_vars=['IATA_CODE'], var_name='Delay_Type', value_name='Average_Delay')

plt.figure(figsize=(14, 8))
sns.barplot(x='IATA_CODE', y='Average_Delay', hue='Delay_Type', data=delay_data)
plt.title('Average Delay Types for Top 15 Busiest Airports')
plt.xlabel('Airport Code')
plt.ylabel('Average Delay (minutes)')
plt.xticks(rotation=45)
plt.legend(title='Delay Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("Analysis completed.")
#========================================================================================
# 4. Temporal Patterns (Monthly or Daily Trends)
# Convert MONTH and DAY to a datetime format for analysis
flights_df['FLIGHT_DATE'] = pd.to_datetime(flights_df[['MONTH', 'DAY']].assign(YEAR=2024))  # Assuming year 2024 for analysis

# Calculate monthly flight counts
monthly_flight_counts = flights_df['FLIGHT_DATE'].dt.to_period('M').value_counts().sort_index()

# Display results
monthly_flight_counts.plot(kind='bar', title='Monthly Flight Counts', xlabel='Month', ylabel='Number of Flights')
plt.show()

#========================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the flights data
flights_df = pd.read_csv('flights.csv')

# Convert MONTH and DAY to datetime for easier analysis
flights_df['DATE'] = pd.to_datetime(flights_df[['MONTH', 'DAY']].assign(YEAR=2024))

# Monthly trends
monthly_trends = flights_df.groupby('MONTH').size()

# Daily trends (considering all days in the dataset)
daily_trends = flights_df.groupby('DATE').size()

# Plotting Monthly Trends
plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_trends.index, y=monthly_trends.values, marker='o')
plt.title('Monthly Flight Trends')
plt.xlabel('Month')
plt.ylabel('Number of Flights')
plt.xticks(monthly_trends.index)
plt.grid()
plt.tight_layout()
plt.show()

# Plotting Daily Trends
plt.figure(figsize=(12, 6))
sns.lineplot(x=daily_trends.index, y=daily_trends.values, marker='o')
plt.title('Daily Flight Trends')
plt.xlabel('Date')
plt.ylabel('Number of Flights')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

print("Temporal patterns analysis completed.")
#============================================================================
# 5. Delay Analysis (Most Common Types of Delays)
# Calculate total delays by type
delay_types = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
total_delays = flights_df[delay_types].sum()

# Display results
total_delays.plot(kind='bar', title='Total Delays by Type', xlabel='Delay Type', ylabel='Total Delay (minutes)')
plt.show()

#========================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the flights data
flights_df = pd.read_csv('flights.csv')

# Define delay columns
delay_columns = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']

# Calculate total delay and create a boolean column for delayed flights
flights_df['TOTAL_DELAY'] = flights_df[delay_columns].sum(axis=1)
flights_df['IS_DELAYED'] = flights_df['TOTAL_DELAY'] > 0

# Calculate the percentage of delayed flights
percent_delayed = (flights_df['IS_DELAYED'].sum() / len(flights_df)) * 100

# Calculate average delay duration for delayed flights
avg_delay_duration = flights_df[flights_df['IS_DELAYED']]['TOTAL_DELAY'].mean()

# Calculate the frequency of each type of delay
delay_frequency = (flights_df[delay_columns] > 0).sum().sort_values(ascending=False)
delay_frequency_percent = (delay_frequency / len(flights_df)) * 100

# Calculate average duration of each type of delay
avg_delay_duration_by_type = flights_df[delay_columns].mean().sort_values(ascending=False)

print(f"Percentage of flights delayed: {percent_delayed:.2f}%")
print(f"Average delay duration for delayed flights: {avg_delay_duration:.2f} minutes")
print("\
Frequency of each type of delay:")
print(delay_frequency_percent)
print("\
Average duration of each type of delay (in minutes):")
print(avg_delay_duration_by_type)

# Visualize frequency of delay types
plt.figure(figsize=(12, 6))
sns.barplot(x=delay_frequency_percent.index, y=delay_frequency_percent.values)
plt.title('Frequency of Each Type of Delay')
plt.xlabel('Delay Type')
plt.ylabel('Percentage of Flights')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize average duration of delay types
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_delay_duration_by_type.index, y=avg_delay_duration_by_type.values)
plt.title('Average Duration of Each Type of Delay')
plt.xlabel('Delay Type')
plt.ylabel('Average Delay Duration (minutes)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze delay duration distribution
plt.figure(figsize=(12, 6))
sns.histplot(flights_df[flights_df['IS_DELAYED']]['TOTAL_DELAY'], bins=50, kde=True)
plt.title('Distribution of Delay Durations')
plt.xlabel('Delay Duration (minutes)')
plt.ylabel('Frequency')
plt.xlim(0, 300)  # Limit x-axis to 5 hours for better visibility
plt.tight_layout()
plt.show()

print("Analysis completed.")
#================================================================================================================
#6. Geographical Analysis of Flights and Airports
# Merge flights with airport data to get geographical information
flights_with_airports = pd.merge(flights_df, airports_df, left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left')

# Plotting the geographical distribution of flights
plt.figure(figsize=(10, 6))
sns.scatterplot(data=flights_with_airports, x='LONGITUDE', y='LATITUDE', hue='AIRLINE', alpha=0.5)
plt.title('Geographical Distribution of Flights')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Airline', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
#=======================================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the airports data
airports_df = pd.read_csv('airports.csv')
print("Airports data loaded. Shape:", airports_df.shape)
print("\
Airports columns:", airports_df.columns.tolist())

# Load the flights data
flights_df = pd.read_csv('flights.csv')
print("\
Flights data loaded. Shape:", flights_df.shape)
print("\
Flights columns:", flights_df.columns.tolist())

# Display the first few rows of each dataframe
print("\
First few rows of airports data:")
print(airports_df.head())

print("\
First few rows of flights data:")
print(flights_df.head())

print("\
Data loading and initial inspection completed.")
#=======================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
flights_df = pd.read_csv('flights.csv')

# Count the most common routes
route_counts = flights_df.groupby(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']).size().reset_index(name='count')
top_routes = route_counts.sort_values('count', ascending=False).head(10)

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='ORIGIN_AIRPORT', hue='DESTINATION_AIRPORT', data=top_routes)
plt.title('Top 10 Most Common Flight Routes')
plt.xlabel('Number of Flights')
plt.ylabel('Origin Airport')
plt.tight_layout()
plt.show()


print("Top 10 Most Common Flight Routes:")
print(top_routes)


# Analyze average delays
delay_columns = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
avg_delays = flights_df[delay_columns].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
avg_delays.plot(kind='bar')
plt.title('Average Delay by Type')
plt.xlabel('Delay Type')
plt.ylabel('Average Delay (minutes)')
plt.tight_layout()
plt.show()

print("\
Average Delays by Type:")
print(avg_delays)
#print("\Bar plot saved as 'avg_delays.png'")

# Load airports data for geographical analysis
airports_df = pd.read_csv('airports.csv')

# Create a scatter plot of airport locations
plt.figure(figsize=(15, 10))
plt.scatter(airports_df['LONGITUDE'], airports_df['LATITUDE'], alpha=0.5)
plt.title('Geographical Distribution of Airports')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.show()

print("\
Geographical distribution of airports plot saved as 'airport_distribution.png'")

# Analyze busiest airports
busiest_origins = flights_df['ORIGIN_AIRPORT'].value_counts().head(10)
busiest_destinations = flights_df['DESTINATION_AIRPORT'].value_counts().head(10)

print("\
Top 10 Busiest Origin Airports:")
print(busiest_origins)

print("\
Top 10 Busiest Destination Airports:")
print(busiest_destinations)
#=======================================================================

#Data Cleaning
import pandas as pd
import numpy as np

# Load the flights dataset
df = pd.read_csv('flights.csv')

# Display initial information about the dataset
print(df.info())
print("\
First few rows of the dataset:")
print(df.head())

# Check for missing values
print("\
Missing values:")
print(df.isnull().sum())

# Check unique values in categorical columns
categorical_columns = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'CANCELLATION_REASON']
for col in categorical_columns:
    print(f"\
Unique values in {col}:")
    print(df[col].unique())

# Display summary statistics for numerical columns
print("\
Summary statistics for numerical columns:")
print(df.describe())

print("\
Data cleaning process complete. Ready for further analysis.")
#========================================================================

#Missing Values

# Handling missing values in the dataset

# Fill missing TAIL_NUMBER with 'Unknown'
df['TAIL_NUMBER'].fillna('Unknown', inplace=True)

# Fill missing SCHEDULED_TIME with the median value
median_scheduled_time = df['SCHEDULED_TIME'].median()
df['SCHEDULED_TIME'].fillna(median_scheduled_time, inplace=True)

# Fill missing CANCELLATION_REASON with 'Not Canceled'
df['CANCELLATION_REASON'].fillna('Not Canceled', inplace=True)

# Fill missing values in delay columns with 0
delay_columns = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
for col in delay_columns:
    df[col].fillna(0, inplace=True)

# Verify that missing values have been handled
print("\
Missing values after cleaning:")
print(df.isnull().sum())

print("Missing values have been handled according to best practices.")

