import pandas as pd

# Load the data
df = pd.read_csv('data/data_after_unnormalized.csv', parse_dates=['date'])

# List of variables to summarize
columns = [
    'circumplex.arousal_avg', 'circumplex.valence_avg', 'mood_avg',
    'circumplex.arousal_std', 'circumplex.valence_std', 'mood_std',
    'activity', 'appCat.builtin', 'appCat.communication',
    'appCat.entertainment', 'appCat.finance', 'appCat.game',
    'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
    'appCat.unknown', 'appCat.utilities', 'appCat.weather',
    'call', 'screen', 'sms'
]

# Summary statistics
summary = df[columns].agg(['count', 'min', 'max', 'mean', 'median']).T.round(2)

# Add optional descriptions
descriptions = {
    'circumplex.arousal_avg': 'Avg self-reported arousal (-2 to 2)',
    'circumplex.valence_avg': 'Avg self-reported valence (-2 to 2)',
    'mood_avg': 'Avg self-reported mood (1 to 10)',
    'circumplex.arousal_std': 'Std of arousal scores',
    'circumplex.valence_std': 'Std of valence scores',
    'mood_std': 'Std of mood scores',
    'activity': 'Activity score (0 to 1)',
    'appCat.builtin': 'Builtin app usage (sec)',
    'appCat.communication': 'Communication app usage (sec)',
    'appCat.entertainment': 'Entertainment app usage (sec)',
    'appCat.finance': 'Finance app usage (sec)',
    'appCat.game': 'Game app usage (sec)',
    'appCat.office': 'Office app usage (sec)',
    'appCat.other': 'Other app usage (sec)',
    'appCat.social': 'Social app usage (sec)',
    'appCat.travel': 'Travel app usage (sec)',
    'appCat.unknown': 'Unknown app usage (sec)',
    'appCat.utilities': 'Utilities app usage (sec)',
    'appCat.weather': 'Weather app usage (sec)',
    'call': 'Number of calls',
    'screen': 'Screen time (sec)',
    'sms': 'Number of SMS'
}
summary['description'] = summary.index.map(descriptions)
summary = summary.round(2)


# View or export
print(summary)

