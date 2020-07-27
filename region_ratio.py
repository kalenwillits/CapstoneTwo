
def calculate_all_regions(regions)
    pacific_df = df[df['Origin'].isin(pacific)] # This get all columns in df that include those regions

    def calculate_region_ratios(regions):
        ratios = []
        for region in regions:
            region_ratio = calculate_origin_ratios(region)
            ratios.append(region_ratio)
        return pd.concat(ratios)


    pacific_region_delays = calculate_region_ratios(pacific)
    pacific_region_delays = pacific_region_delays.drop('Origin')

    def calculate_region_means(region, df=df):
        flight_dates = region['FlightDate'].unique()
        ratio_means = {'FlightDate': [], 'DelayRatio': []}
        for flight_date in flight_dates:
            ratio_means['FlightDate'].append(flight_date)
            ratio_means['DelayRatio'].append(np.mean(region[region['FlightDate'] == flight_date]['DelayRatio']))
        return pd.DataFrame(ratio_means)


    pacific_DelayRatio = calculate_region_means(pacific_region_delays)


    plt.figure(figsize=(10,10))
    plt.plot(p_region['FlightDate'], p_region['DelayRatio'])
    plt.xlabel('Flight Date')
    plt.xticks(rotation=90)
    plt.ylabel('Delay Ratio')
    plt.title('Pacific Delay Ratios by Date')
    plt.savefig('figures/DelayRatios_Pacific.png')
