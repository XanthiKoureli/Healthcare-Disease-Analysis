import streamlit as st
from openai import OpenAI
import pandas as pd
import logging
import json
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=st.secrets["OPEN_AI_KEY"])


def read_file(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data


def extract_number(value):
    """
    Extracts the number from a percentage string.
    """
    percentage_pattern = r'(\d+(\.\d+)?)%'
    match = re.match(percentage_pattern, value.strip())
    if match:
        return float(match.group(1))
    else:
        return None


@st.cache_data(show_spinner=False)
def get_disease_info(name):
    """
    Function to query OpenAI and return structured information about a disease.
    """

    disease_template = read_file('disease_response_template.json')

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system",
             "content": f"Please provide information on the following aspects for {name}: 1. Key Statistics, 2. Recovery Options, 3. Recommended Medications. Format the response in JSON with keys for 'name', 'statistics', 'total_cases' (this always has to be a number), 'recovery_rate' (this always has to be a percentage), 'mortality_rate' (this always has to be a percentage) 'recovery_options', (explain each recovery option in detail), and 'medication', (give some side effect examples and dosages).Also this is a json template that you MUST respect {disease_template}. Finally the response should be in json format and not in markdown"}
        ]
    )
    content = response.choices[0].message.content
    return content


def display_disease_info(disease):
    try:
        info = json.loads(disease)
        # Statistics  Diagram
        recovery_rate = extract_number(info['statistics']["recovery_rate"])
        mortality_rate = extract_number(info['statistics']["mortality_rate"])

        # Define the tabs you want to show based on conditions
        tabs = []
        if recovery_rate is not None and mortality_rate is not None:
            tabs.append("Statistics")
        tabs.append("Recovery")
        tabs.append("Medication")

        # Create the tabs dynamically
        tab_objects = st.tabs(tabs)

        # Loop over the tabs to display content
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


def display_medication(medication):
    # st.write("## Medication")
    st.subheader(f"{medication['name']}")

    # Display side effects in a bullet point format using markdown
    st.write("#### Side Effects")
    st.markdown("\n".join([f"- {effect}" for effect in medication['side_effects']]))

    # Display dosage information
    st.write("#### Dosage")
    st.write(medication['dosage'])

    st.write("---")  # Add a divider


def display_recovery_options(recovery_options):
    # st.write("## Recovery Options")
    for option, description in recovery_options.items():
        st.subheader(option)
        st.markdown(f"* {description}")


def display_statistics(recovery, mortality, name):
    chart_data = pd.DataFrame(
        {
            "Recovery Rate": [recovery],
            "Mortality Rate": [mortality],
        },
        index=["Rate"]  # This is a single index. You might adjust it based on your data structure.
    )

    st.write(f"## Statistics for {name}")
    st.bar_chart(chart_data)


@st.dialog("OpenAI response", width="large")
def display_json(data):
    st.json(data)


def show_openai_response(openai_response):
    # Button to show JSON data
    if st.button('Show OpenAI response'):
        display_json(openai_response)



def main():
    st.title("Disease Information Dashboard")

    # Text input to get the disease name from the user
    disease_name = st.text_input("Enter the name of the disease:")
    disease_info = None

    if disease_name:
        # Display a spinner while fetching the data
        with st.spinner(f"Fetching disease information for '{disease_name}' from OpenAI..."):
            disease_info = get_disease_info(disease_name)

    if disease_info:
        # Display the OpenAI response
        show_openai_response(disease_info)
        st.divider()
        # Display more detailed disease info
        display_disease_info(disease_info)
    elif disease_name and not disease_info:
        st.error("No valid disease information found. Please try again.")


if __name__ == "__main__":
    main()
