import streamlit as st
from openai import OpenAI
import pandas as pd
import logging
import json
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=st.secrets["OPEN_AI_KEY"])


# Function to read JSON data from a file
def read_file(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data


# Function to extract the numerical value from a percentage string
def extract_number(value):

    percentage_pattern = r'(\d+(\.\d+)?)%'
    match = re.match(percentage_pattern, value.strip())
    if match:
        return float(match.group(1))
    else:
        return None


# Cached function to query OpenAI and retrieve structured information about a disease
@st.cache_data(show_spinner=False)
def get_disease_info(name):
    # Load template that defines the expected format of the response
    disease_template = read_file('./disease_response_template.json')

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system",
             "content": f"Please provide information on the following aspects for {name}: 1. Key Statistics, 2. Recovery Options, 3. Recommended Medications. Format the response in JSON with keys for 'name', 'statistics', 'total_cases' (this always has to be a number), 'recovery_rate' (this always has to be a percentage), 'mortality_rate' (this always has to be a percentage) 'recovery_options', (explain each recovery option in detail), and 'medication', (give some side effect examples and dosages).Also this is a json template that you MUST respect {disease_template}. Finally the response should be in json format and not in markdown"}
        ]
    )
    content = response.choices[0].message.content
    return content


# Function to display detailed information about a disease
def display_disease_info(disease):
    try:
        info = json.loads(disease)

        # Extract recovery and mortality rates for display
        recovery_rate = extract_number(info['statistics']["recovery_rate"])
        mortality_rate = extract_number(info['statistics']["mortality_rate"])

        # Define which tabs to display based on available information
        tabs = []
        if recovery_rate is not None and mortality_rate is not None:
            tabs.append("Statistics")
        tabs.append("Recovery")
        tabs.append("Medication")

        # Dynamically create the tabs in the Streamlit UI
        tab_objects = st.tabs(tabs)

        # Display content based on selected tab
        for idx, tab_name in enumerate(tabs):
            with tab_objects[idx]:  # Match tab with content
                if tab_name == "Statistics":
                    display_statistics(recovery=recovery_rate, mortality=mortality_rate, name=info['name'])
                elif tab_name == "Recovery":
                    display_recovery_options(info['recovery_options'])
                elif tab_name == "Medication":
                    display_medication(info['medication'])

    except json.JSONDecodeError:
        st.error("Failed to decode the response into JSON. Please check the format of the OpenAI response.")


# Function to display medication details
def display_medication(medication):

    st.subheader(f"{medication['name']}")

    # Display side effects in a bullet point format
    st.write("#### Side Effects")
    st.markdown("\n".join([f"- {effect}" for effect in medication['side_effects']]))

    # Display dosage information
    st.write("#### Dosage")
    st.write(medication['dosage'])

    st.write("---")  # Add a divider


# Function to display recovery options
def display_recovery_options(recovery_options):
    # Loop over recovery options and display each one with its description
    for option, description in recovery_options.items():
        st.subheader(option)
        st.markdown(f"* {description}")


# Function to display statistics using a bar chart
def display_statistics(recovery, mortality, name):
    # Create a DataFrame for the chart data
    chart_data = pd.DataFrame(
        {
            "Recovery Rate": [recovery],
            "Mortality Rate": [mortality],
        },
        index=["Rate"]
    )

    st.write(f"## Statistics for {name}")
    st.bar_chart(chart_data) # Display the bar chart


# Dialog to display JSON data in a Streamlit modal
@st.dialog("OpenAI response", width="large")
def display_json(data):
    st.json(data)   # Display JSON data in a formatted way


# Function to show OpenAI response when a button is clicked
def show_openai_response(openai_response):
    # Button to show JSON data
    if st.button('Show OpenAI response'):
        display_json(openai_response)


# Main function to run the Streamlit app
def main():
    st.title("Disease Information Dashboard")

    # Input field for the user to enter the disease name
    disease_name = st.text_input("Enter the name of the disease:")
    disease_info = None

    if disease_name:
        # Show a spinner while fetching the disease data
        with st.spinner(f"Fetching disease information for '{disease_name}' from OpenAI..."):
            disease_info = get_disease_info(disease_name)

    if disease_info:
        # Display the raw OpenAI response
        show_openai_response(disease_info)
        st.divider()

        # Display detailed disease information
        display_disease_info(disease_info)
    elif disease_name and not disease_info:
        st.error("No valid disease information found. Please try again.")


if __name__ == "__main__":
    main()
