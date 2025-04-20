import pandas as pd
from common import TARGET


def process_data():
    df = pd.read_csv(
        'neural_network/simulated_dataset.csv').drop(columns=['entry log', 'Team', 'Driver'])
    print(df.columns)
    # First, encode all non-number fields
    motorsport_types = df['Motorsport_Type'].unique()
    motorsport_type_dict = {motorsport: i for i,
                            motorsport in enumerate(motorsport_types)}
    df['Motorsport_Type'] = df['Motorsport_Type'].map(motorsport_type_dict)

    tracks = df['Track'].unique()
    track_dict = {track: i for i, track in enumerate(tracks)}
    df['Track'] = df['Track'].map(track_dict)

    tire_compounds = df['Tire_Compound'].unique()
    tire_compound_dict = {tire_compound: i for i,
                          tire_compound in enumerate(tire_compounds)}
    df['Tire_Compound'] = df['Tire_Compound'].map(tire_compound_dict)

    driving_styles = df['Driving_Style'].unique()
    driving_style_dict = {driving_style: i for i,
                          driving_style in enumerate(driving_styles)}
    df['Driving_Style'] = df['Driving_Style'].map(driving_style_dict)

    events = df['Event'].unique()
    event_dict = {event: i for i, event in enumerate(events)}
    df['Event'] = df['Event'].map(event_dict)

    # Find out which features(columns) are the most influential to our target
    correlations = df.corr()[TARGET].drop(TARGET).abs()
    most_influential = correlations.sort_values(ascending=False)
    print(most_influential)

    # Drop the columns that do not contribute to the variation of the target
    threshold = 0.1
    columns_to_drop = most_influential[most_influential < threshold].index
    df_reduced = df.drop(columns=columns_to_drop)

    print(f"Dropped columns: {list(columns_to_drop)}")

    df_reduced.to_csv('neural_network/data.csv', index=False)

    with open('neural_network/keys.txt', 'w') as file:
        file.write(str(motorsport_type_dict))
        file.write('\n')
        file.write(str(track_dict))
        file.write('\n')
        file.write(str(tire_compound_dict))
        file.write('\n')
        file.write(str(driving_style_dict))
        file.write('\n')
        file.write(str(event_dict))
    
    return df_reduced

process_data()