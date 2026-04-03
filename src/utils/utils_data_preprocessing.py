import pandas as pd


# --- DATA PATH ---
DATA_PATH_CELEBRITY = "../datasets/Celebrity/df_celebrity.csv"
DATA_PATH_CIDII = "../datasets/CIDII/df_cidii.csv"
DATA_PATH_FAKES = "../datasets/FaKES/df_fakes.csv"
DATA_PATH_FAKEVSATIRE = "../datasets/FakeVsSatire/df_fakevssatire.csv"
DATA_PATH_HORNE = "../datasets/Horne/df_horne.csv"
DATA_PATH_INFODEMIC = "../datasets/Infodemic/df_infodemic.csv"
DATA_PATH_ISOT = "../datasets/ISOT/df_isot.csv"
DATA_PATH_KAGGLE_CLEMENT = "../datasets/Kaggle_clement/df_kaggle_clement.csv"
DATA_PATH_KAGGLE_MEG = "../datasets/Kaggle_meg/df_kaggle_meg.csv"
DATA_PATH_LIAR_PLUS = "../datasets/LIAR_PLUS/df_liarplus.csv"
DATA_PATH_POLITIFACT = "../datasets/Politifact/df_politifact.csv"
DATA_PATH_UNIPI_NDF = "../datasets/Unipi_NDF/df_ndf.csv"

# --- TOPICS ---
TOPIC_CELEBRITY = "gossip"
TOPIC_CIDII = "islam"
TOPIC_FAKES = "syria"
TOPIC_FAKEVSATIRE = "politics"
TOPIC_HORNE = "politics"
TOPIC_INFODEMIC = "covid"
TOPIC_ISOT = "politics"
TOPIC_KAGGLE_CLEMENT = "politics"
TOPIC_KAGGLE_MEG = "general"
TOPIC_LIAR_PLUS = "politics"
TOPIC_POLITIFACT = "politics"
TOPIC_UNIPI_NDF = "notredame"



# ------------------------------
# Data loading and preprocessing
# ------------------------------
def data_loading():
    """
    Preprocess and load all datasets into a dictionary of DataFrames.

    Returns:
        dict: A dictionary where keys are dataset names and values are preprocessed DataFrames.
    """

    # --- CELEBRITY ---
    dfCelebrity = pd.read_csv(DATA_PATH_CELEBRITY, sep="\t", encoding="utf-8")
    # texts and labels
    dfCelebrity = dfCelebrity[['texts', 'labels']] # keep only relevant columns
    dfCelebrity = dfCelebrity.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    dfCelebrity = dfCelebrity.drop_duplicates(subset=['texts', 'labels']) # drop duplicates
    # date
    dfCelebrity['date'] = pd.NaT # add empty date column to match other datasets
    # topic
    dfCelebrity['topic'] = TOPIC_CELEBRITY

    # --- CIDII ---
    dfCidii = pd.read_csv(DATA_PATH_CIDII, sep="\t", encoding="utf-8")
    # texts and labels
    dfCidii = dfCidii[['texts', 'labels']] # keep only relevant columns
    dfCidii = dfCidii.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    dfCidii = dfCidii.drop_duplicates(subset=['texts', 'labels']) # drop duplicates
    # date
    dfCidii['date'] = pd.NaT # add empty date column to match other datasets
    # topic
    dfCidii['topic'] = TOPIC_CIDII

    # --- FAKES ---
    dfFakes = pd.read_csv(DATA_PATH_FAKES, sep="\t", encoding="utf-8")
    # texts and labels
    dfFakes = dfFakes[['texts', 'labels']] # keep only relevant columns
    dfFakes = dfFakes.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    dfFakes = dfFakes.drop_duplicates(subset=['texts', 'labels']) # drop duplicates
    # date
    dfFakesDates = pd.read_csv("../datasets/FaKES/date_df_fakes.csv", encoding="ISO-8859-1") # load dataset with dates
    dfFakesDates = dfFakesDates.rename(columns={'article_content': 'texts'}) # rename column to match
    dfFakes = pd.merge(dfFakes, dfFakesDates[['texts', 'date']], on="texts", how="left") # merge to add dates
    dfFakes['date'] = pd.to_datetime(dfFakes['date'], errors='coerce') # convert date column to datetime, coerce errors to NaT
    # topic
    dfFakes['topic'] = TOPIC_FAKES

    # --- FAKEVSATIRE ---
    dfFakeVsSatire = pd.read_csv(DATA_PATH_FAKEVSATIRE, sep="\t", encoding="utf-8")
    # texts and labels
    dfFakeVsSatire = dfFakeVsSatire[['texts', 'labels']] # keep only relevant columns
    dfFakeVsSatire = dfFakeVsSatire.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    dfFakeVsSatire = dfFakeVsSatire.drop_duplicates(subset=['texts', 'labels']) # drop duplicates
    # date
    dfFakeVsSatire['date'] = pd.to_datetime("2016-01-01") # add static date column to match other datasets
    # topic
    dfFakeVsSatire['topic'] = TOPIC_FAKEVSATIRE
    
    # --- HORNE ---
    dfHorne = pd.read_csv(DATA_PATH_HORNE, sep="\t", encoding="utf-8")
    # texts and labels
    dfHorne = dfHorne[['texts', 'labels']] # keep only relevant columns
    dfHorne = dfHorne.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    dfHorne = dfHorne.drop_duplicates(subset=['texts', 'labels']) # drop duplicates
    # date
    dfHorne['date'] = pd.to_datetime("2016-01-01") # add static date column to match other datasets
    # topic
    dfHorne['topic'] = TOPIC_HORNE

    # --- INFODEMIC ---
    dfInfodemic = pd.read_csv(DATA_PATH_INFODEMIC, sep="\t", encoding="utf-8")
    # texts and labels
    dfInfodemic = dfInfodemic[['texts', 'labels']] # keep only relevant columns
    dfInfodemic = dfInfodemic.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    dfInfodemic = dfInfodemic.drop_duplicates(subset=['texts', 'labels']) # drop duplicates
    # date
    dfInfodemic['date'] = pd.to_datetime("2020-01-01") # add static date column to match other datasets
    # topic
    dfInfodemic['topic'] = TOPIC_INFODEMIC

    # --- ISOT ---
    dfIsot = pd.read_csv(DATA_PATH_ISOT, sep="\t", encoding="utf-8")
    # texts and labels
    dfIsot = dfIsot[['texts', 'labels']] # keep only relevant columns
    dfIsot = dfIsot.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    dfIsot = dfIsot.drop_duplicates(subset=['texts', 'labels']) # drop duplicates
    # date
    dfIsot['date'] = pd.to_datetime("2016-01-01") # add static date column to match other datasets
    # topic
    dfIsot['topic'] = TOPIC_ISOT

    # --- KAGGLE_CLEMENT ---
    dfKaggleClement = pd.read_csv(DATA_PATH_KAGGLE_CLEMENT, encoding="utf-8")
    # texts and labels
    dfKaggleClement['texts'] = dfKaggleClement['title'].astype(str) + " " + dfKaggleClement['text'].astype(str) # merge title and text
    dfKaggleClement = dfKaggleClement[['texts', 'labels', 'date']] # keep only relevant columns
    dfKaggleClement = dfKaggleClement.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    dfKaggleClement = dfKaggleClement.drop_duplicates(subset=['texts', 'labels'])
    # date
    dfKaggleClement['date'] = pd.to_datetime(dfKaggleClement['date'], errors='coerce') # convert date column to datetime, coerce errors to NaT
    # topic
    dfKaggleClement['topic'] = TOPIC_KAGGLE_CLEMENT

    # --- KAGGLE_MEG ---
    dfKaggleMeg = pd.read_csv(DATA_PATH_KAGGLE_MEG, encoding="utf-8")
    # texts and labels
    dfKaggleMeg['texts'] = dfKaggleMeg['title'].astype(str) + " " + dfKaggleMeg['text'].astype(str) # merge title and text
    dfKaggleMeg['labels'] = dfKaggleMeg['spam_score'].apply(lambda x: 1 if x > 0.5 else 0) # create binary labels based on spam_score
    dfKaggleMeg = dfKaggleMeg[['texts', 'labels', 'published']] # keep only relevant columns
    dfKaggleMeg = dfKaggleMeg.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    dfKaggleMeg = dfKaggleMeg.drop_duplicates(subset=['texts', 'labels'])
    # date
    dfKaggleMeg = dfKaggleMeg.rename(columns={'published': 'date'}) # rename published to date
    dfKaggleMeg['date'] = pd.to_datetime("2016-01-01") # add static date column to match other datasets
    # topic
    dfKaggleMeg['topic'] = TOPIC_KAGGLE_MEG

    # --- LIAR_PLUS ---
    dfLiarPlus = pd.read_csv(DATA_PATH_LIAR_PLUS, sep="\t", encoding="utf-8")
    # texts and labels
    dfLiarPlus = dfLiarPlus[['texts', 'labels']] # keep only relevant columns
    dfLiarPlus = dfLiarPlus.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    dfLiarPlus = dfLiarPlus.drop_duplicates(subset=['texts', 'labels']) # drop duplicates
    # date
    dfLiarPlus['date'] = pd.NaT # add empty date column to match other datasets
    # topic
    dfLiarPlus['topic'] = TOPIC_LIAR_PLUS

    # --- POLITIFACT ---
    dfPolitifact = pd.read_csv(DATA_PATH_POLITIFACT, sep="\t", encoding="utf-8")
    # texts and labels
    dfPolitifact = dfPolitifact[['texts', 'labels']] # keep only relevant columns
    dfPolitifact = dfPolitifact.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    dfPolitifact = dfPolitifact.drop_duplicates(subset=['texts', 'labels']) # drop duplicates
    # date
    dfPolitifact['date'] = pd.NaT # add empty date column to match other datasets
    # topic
    dfPolitifact['topic'] = TOPIC_POLITIFACT

    # --- UNIPI_NDF ---
    dfNDF = pd.read_csv(DATA_PATH_UNIPI_NDF, sep="\t", encoding="utf-8")
    # texts and labels
    dfNDF = dfNDF[['texts', 'labels']] # keep only relevant columns
    dfNDF = dfNDF.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    dfNDF = dfNDF.drop_duplicates(subset=['texts', 'labels']) # drop duplicates
    # date
    dfNDF['date'] = pd.to_datetime("2019-01-01") # add static date column to match other datasets
    # topic
    dfNDF['topic'] = TOPIC_UNIPI_NDF

    return {
        "Celebrity": dfCelebrity,
        "CIDII": dfCidii,
        "FaKES": dfFakes,
        "FakeVsSatire": dfFakeVsSatire,
        "Horne": dfHorne,
        "Infodemic": dfInfodemic,
        "ISOT": dfIsot,
        "Kaggle_clement": dfKaggleClement,
        "Kaggle_meg": dfKaggleMeg,
        "LIAR_PLUS": dfLiarPlus,
        "Politifact": dfPolitifact,
        "Unipi_NDF": dfNDF
    }



# ----------------------------
# Data by topic/date functions
# ----------------------------
def data_by_topic(inverted=False):
    """
    Organize datasets by topic

    Returns:
        dict: A dictionary where keys are topics and values are DataFrames containing all samples for that
    """

    df_dict = data_loading()

    combined_df = pd.concat(df_dict.values(), ignore_index=True).reset_index(drop=True) # combine all datasets

    combined_df = combined_df.dropna(subset=['topic']) # drop rows where topic is NaN

    grouped = combined_df.groupby('topic', sort=False) # group by topic without sorting

    df_dict_by_topic = {topic: group for topic, group in grouped} # create dict from groupby

    # order topics by frequency
    topic_order = combined_df['topic'].value_counts().index
    df_dict_by_topic = {t: df_dict_by_topic[t] for t in topic_order}

    if inverted:
        df_dict_by_topic = dict(reversed(list(df_dict_by_topic.items())))

    return df_dict_by_topic


def data_by_date():
    """
    Organize datasets by date

    Returns:
        dict: A dictionary where keys are years and values are DataFrames containing all samples for that
    """

    df_dict = data_loading()

    combined_df = pd.concat(df_dict.values(), ignore_index=True) # combine all datasets

    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce') # assure date format
    combined_df = combined_df.dropna(subset=['date']) # drop rows where date is NaT
    combined_df = combined_df.sort_values(by='date').reset_index(drop=True) # sort by date

    combined_df['year'] = combined_df['date'].dt.year # extract year for grouping

    # Considering only the year, create a dict of dataframes per year
    grouped = combined_df.groupby('year', sort=False)
    df_dict_by_date = {year: group for year, group in grouped}

    # merge data from 2011 to 2015 into "2011-2015", and "2011-2015" should come first in order
    years_to_merge = [2011, 2012, 2013, 2014, 2015]
    df_dict_by_date['2011-2015'] = pd.concat([df_dict_by_date[year] for year in years_to_merge], ignore_index=True)
    for year in years_to_merge:
        del df_dict_by_date[year]

    # order years
    df_dict_by_date = dict(sorted(df_dict_by_date.items(), key=lambda x: (int(x[0].split('-')[0]) if isinstance(x[0], str) and '-' in x[0] else x[0])))

    return df_dict_by_date