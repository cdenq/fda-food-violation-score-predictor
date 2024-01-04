#----------------------------------------------------
# Imports
#----------------------------------------------------
from modules import helper
from modules import imports

#----------------------------------------------------
# Helper Functions
#----------------------------------------------------
def extract_cuisine_data() -> tuple:
    """
    Loads and creates the cuisine data for input fields.

    Returns -> tuple
        Returns a tuple containing the unique cuisine labels and the mapper
    """
    filepath = f"{imports.APP_PATH_TO_RAW_DATA}/cuisines.csv"
    cuis_labels_df = imports.pd.read_csv(filepath)
    cuis_labels_unique = cuis_labels_df["x"].unique()
    mapper = {}
    for i in range(len(cuis_labels_unique)):
        mapper[cuis_labels_unique[i]] = [cuis_labels_df["asian"][i], cuis_labels_df["ethnic"][i]]
    return (cuis_labels_unique, mapper)    

def sidebar_input_with_icon(header: str, tooltip: str, input_type: str, **kwargs) -> imports.st.sidebar:
    """
    Generates an input panel with the "hover-for-more-information" icon.

    header -> str
        Given input field label

    tooltip -> str
        Given input field description

    input_type -> str
        Given input field input type

    Returns -> Streamlit.sidebar
        Returns the appropriate Streamlit sidebar panel based on the input type
    """
    # Check for correct mode
    options = ["number_input", "selectbox", "text_input", "date_input", "multiselect"]
    if input_type not in options:
        raise ValueError(f"{input_type} invalid; must be within {options}.")
    
    # Create a combined header with icon
    imports.st.sidebar.markdown(f"<div style='display: flex; align-items: center; margin-bottom: -60px;'>"
                                f"<span style='margin-right: 5px; font-size: 16px;'>{header}</span>"
                                f"<span style='font-size:18px; color: grey;' title='{tooltip}'>&#9432;</span>"
                                "</div>", unsafe_allow_html=True)
    
    # Return the appropriate Streamlit input based on input_type
    if input_type == "number_input":
        return imports.st.sidebar.number_input("", **kwargs)
    elif input_type == "selectbox":
        return imports.st.sidebar.selectbox("", **kwargs)
    elif input_type == "text_input":
        return imports.st.sidebar.text_input("", **kwargs)
    elif input_type == "date_input":
        return imports.st.sidebar.date_input("", **kwargs)
    elif input_type == "multiselect":
        return imports.st.sidebar.multiselect("", **kwargs)

def user_input_features() -> imports.pd.DataFrame:
    """
    Displays and logs the user inputs in a collapsible side bar
    
    Returns -> Pandas.DataFrame
        Returns the .csv data as a dataframe
    """
    imports.st.sidebar.subheader("Raw User Input Features")

    # Fetch and process date data
    inspection_start = sidebar_input_with_icon("Inspection Start Date", 
                                               "The start date of the inspection.",
                                               "date_input",
                                               key="inspection_start") 
    inspection_end = sidebar_input_with_icon("Inspection End Date", 
                                             "The end date of the inspection.",
                                             "date_input",
                                             key="inspection_end")
    season_Spring = 0
    season_Summer = 0
    season_Winter = 0
    season = helper.get_season(inspection_start)
    if season == "Spring":
        season_Spring = 1
    elif season == "Summer":
        season_Summer = 1
    elif season == "Winter":
        season_Winter = 1
    inspection_dur = helper.extract_duration_days(inspection_start, inspection_end)
    if inspection_dur < 0:
        inspection_dur = "Start date is after end date; please correct before predicting."

    # Fetch and process score data
    inspection_prev = sidebar_input_with_icon("Prev Inspection Score", 
                                              "The immediate previous inspection penalty score.",
                                              "number_input",
                                              value=50.0, 
                                              key="inspection_prev")
    inspection_avg = sidebar_input_with_icon("Avg Prev Inspection Scores", 
                                             "The average of all previous inspection penalty scores.",
                                             "number_input",
                                             value=50.0, 
                                             key="inspection_avg")
    if inspection_prev < inspection_avg: # the lower the score the better, thus giving positive 1 for lower
        review_trend = 1
    elif inspection_prev > inspection_avg:
        review_trend = -1
    else:
        review_trend = 0
    
    # Fetch and process cuisine data
    cuisine_tuple = extract_cuisine_data()
    cuis_labels_unique = cuisine_tuple[0]
    cuis_mapper = cuisine_tuple[1]
    cuisine_labels = sidebar_input_with_icon("Cuisine Labels",
                                             "The various cuisine labels the restaurant has.",
                                             "multiselect",
                                             default=cuis_labels_unique[0],
                                             options=cuis_labels_unique)
    num_cuis = len(cuisine_labels)
    asian = False
    ethnic = False
    for label in cuisine_labels:
        asian = asian or cuis_mapper[label][0]
        ethnic = ethnic or cuis_mapper[label][1]

    # Fetch and process text data
    review_text = sidebar_input_with_icon("Review Text",
                                          "The text of the inspection review.",
                                          "text_input", 
                                          value="I loved my experience here!",
                                          key="review_text")
    sentiment_score = round(helper.extract_sentiment_score(review_text), 4)

    # Fetch and process review count data
    review_count = sidebar_input_with_icon("Total Reviews",
                                           "The total number of reviews for this restaurant, including the non-positive reviews.",
                                           "number_input",
                                           value=30,
                                           key="review_count")
    nonpos_review_count = sidebar_input_with_icon("Total Non-Positive Reviews",
                                                  "The total number of non-positive reviews for this restaurant.",
                                                  "number_input",
                                                  value=10,
                                                  key="nonpos_review_count")
    non_pos_review_perc = round(nonpos_review_count / review_count, 4)

    # Creating df from user inputs
    data = {"inspection_average_prev_penalty_scores": inspection_avg,
            "asian": asian,
            "ethnic": ethnic,
            "number_of_cuisines": num_cuis,
            "inspection_duration (days)": inspection_dur,
            "season_Spring": season_Spring,
            "season_Summer": season_Summer,
            "season_Winter": season_Winter,
            "sentiment_score": sentiment_score,
            "perc_non_positive_review_count": non_pos_review_perc,
            "review_trend": review_trend}
    features = imports.pd.DataFrame(data, index=[0])

    return features

#----------------------------------------------------
# Main
#----------------------------------------------------
def main():
    """
    Main function to run the Streamlit app.
    """
    # Load model & preprocessor
    filepath = f"{imports.APP_PATH_TO_SAVED_MODELS}/logistic_regression_final.pkl"
    model = imports.joblib.load(filepath)
    filepath = f"{imports.APP_PATH_TO_SAVED_MODELS}/preprocessor_logistic_regression_minmax.pkl"
    preprocessor = imports.joblib.load(filepath)

    # Set up the title and description
    imports.st.title("FDA Noncompliance Predictor")
    imports.st.write("This app uses a logistic regression model to predict whether an establishment is at risk of noncompliance.")

    # Generate side-bar with user inputs
    input_df = user_input_features()

    # Display user input features in a vertical table format
    imports.st.subheader("Interpreted Input Features")
    imports.st.write("Note: season_Fall as a data field is intentionally not included.")
    imports.st.write("---")
    for key, value in input_df.iloc[0].items():
        imports.st.text(f"{key}: {value}")
    imports.st.write("---")

    # Prediction
    if imports.st.button("Predict"):
        scaled_df = preprocessor.transform(input_df)
        prediction = model.predict(scaled_df)
        probability = model.predict_proba(scaled_df)

        # Display Prediction and info icon
        pred_header = "Prediction"
        pred_tooltip = "This model was optimized for reducing false positives. In context, this means that when the model predicts noncompliance, it is right 98% of the time, and when it predicts compliance, it is right 61% of the time. This model can be easily retrained for a more balanced performance. Given the regulatory context, reducing false positives was the chosen starting metric."
        imports.st.markdown(f""" 
                            <div style='display: flex; align-items: center;'>
                            <span style='font-size: 20px;'>{pred_header}</span>
                            <span style='font-size: 18px; color: grey; margin-left: 10px;' title='{pred_tooltip}'>&#9432;</span>
                            </div>
                            """, unsafe_allow_html=True)

        # imports.st.subheader("Prediction")
        if prediction[0] == 0:
            label_class = "Complaint - no serious violation"
        else:
            label_class = "Noncomplaint - serious violation"
        imports.st.write(f"{label_class}, {round(imports.np.max(probability) * 100, 2)}% sure")
        imports.st.write("---")

    # Bottom
    imports.st.subheader("Project Info")
    imports.st.write("This project was created by [Christopher Denq](https://github.com/cdenq), and its repo is located [here](https://github.com/cdenq/fda-food-violation-score-predictor).")
    return None

#----------------------------------------------------
# Entry
#----------------------------------------------------
if __name__ == "__main__":
    main()
