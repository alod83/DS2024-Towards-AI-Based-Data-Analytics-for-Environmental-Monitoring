import pandas as pd

for year in range(2013,2024):
    print(year)
    df = pd.read_csv(f"source/{year}.csv", sep=';')


    df = df[['cod_staz', 'yyyymmdd_hhii', 'T']]
    df = df[df['cod_staz'] == 'AL007']
    df = df[df['T'] != -9999.9]
    df = df[df['T'] != "-9.999.900"]

    # Define a custom date parsing function
    def parse_date(date_str):
        try:
            # Split the date and time parts
            date_part, time_part = date_str.split('_')
            hour = time_part[:2]
            minute = time_part[2:]
            
            # Correct invalid minute values if necessary
            if int(minute) >= 60:
                minute = '59'
            
            # Reconstruct the corrected time string
            corrected_time_str = f"{hour}{minute}"
            corrected_date_str = f"{date_part}_{corrected_time_str}"
            
            # Parse the corrected date string
            return pd.to_datetime(corrected_date_str, format='%Y%m%d_%H%M')
        except Exception as e:
            # Handle any parsing exceptions
            print(f"Error parsing date: {date_str} - {e}")
            return pd.NaT  # Return a NaT (Not a Time) if parsing fails


    # Apply the custom date parsing function to the DataFrame column
    df['date'] = df['yyyymmdd_hhii'].apply(parse_date)

    df[df['date'] != pd.NaT]
    df = df[['date', 'T']]

    df.to_csv(f"data/{year}_AL007.csv", index=False)


