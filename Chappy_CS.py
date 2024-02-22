# Import required libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from openai import AzureOpenAI
import os
import re
import configparser
import io
import base64


# Initialize the ConfigParser
global config
config = configparser.ConfigParser()

# Read the config file
config.read('.gitignore/config.ini')

global AZURE_OPENAI_KEY
global AZURE_OPENAI_ENDPOINT

AZURE_OPENAI_KEY = config['credentials']['AZURE_OPENAI_KEY']
AZURE_OPENAI_ENDPOINT = config['credentials']['AZURE_OPENAI_ENDPOINT']

client = AzureOpenAI(
  api_key = AZURE_OPENAI_KEY,  
  api_version = "2023-12-01-preview",
  azure_endpoint = AZURE_OPENAI_ENDPOINT
)


def save_figure(figure: go.Figure, file):
    return figure.write_image(file, format='pdf')


def llm(prompt,system,tag,model="gpt-35-turbo",temperature=0,max_new_tokens=512):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        user=tag,
        temperature=temperature,
        max_tokens=max_new_tokens)
    
    return response.choices[0].message.content


def call_vision_api(image_base64, model="vision", max_tokens=512):


    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant whose job is to summarize images in detail."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the following image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=max_tokens
    )

    return response.choices[0].message.content

def conversation(messages,model,tag,temperature=0,max_new_tokens=512, image=None):    

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        user=tag,
        temperature=temperature,
        max_tokens=max_new_tokens)
    return response.choices[0].message.content


def format_conversation(messages):
    formatted_messages = []
    for message in messages:
        if message['role'] == 'system':
            continue  # Skip system messages

        # Determine the alignment and text class based on the role
        if message['role'] == 'user':
            alignment_class = "user-message"
            text_class = "user-text"
        else:
            alignment_class = "Chappy-message"
            text_class = "Chappy-text"

        formatted_messages.append({
            'alignment_class': alignment_class,
            'text_class': text_class,
            'content': message['content']
        })
    return formatted_messages

def display_chat(formatted_messages):
    st.markdown("""
        <style>
            .message-box {
                display: flex;
                flex-direction: column;
                align-items: flex-start; /* Default alignment for Chappy */
                margin-bottom: 20px;
                width: 100%; /* Adjust the width */
            }
            .message-box.user {
                align-items: flex-end; /* Alignment for user */
            }
            .message-text {
                border-radius: 20px;
                color: white;
                display: inline-block;
                word-wrap: break-word;
                max-width: 80%; /* Adjust the max width */
                padding: 10px 15px;
            }
            .message-text a { /* Targeting links within message-text */
                color: #f934b9; /* Light pink color for links */
                text-decoration: none; /* Optional: removes underline from links */
            }
            .message-text a:hover { /* Optional: Change color on hover */
                color: #fb7dd2; /* A darker pink for hover effect */
            }
            .user-text {
                background-color: #8913db;
            }
            .Chappy-text {
                background-color: #0084ff;
            }
            .message-label {
                font-size: 0.8em;
                color: #666;
                margin-bottom: 2px;
            }
        </style>
    """, unsafe_allow_html=True)

    for message in formatted_messages:
        label = "You" if message['alignment_class'] == 'user-message' else "Chappy"

        #if message['content'].endswith("```"):
        #    message['content'] = message['content'] + "\n"
        
        # Isolate the first and last line
        first_line = message['content'].split('\n', 1)[0]
        last_line = message['content'].split('\n')[-1]
        # Regex pattern to match common Markdown syntax in the first line
        # Adjust the pattern as needed to match specific Markdown elements you're interested in
        markdown_pattern = r'(\#{1,6}\s|>|`{3,}|[*_]{1,2}|!\[.*?\]\([^)]*?\)|\[.*?\]\([^)]*?\))'

        if re.search(markdown_pattern, first_line):
            message['content'] = "\n\n" + message['content']

        if re.search(markdown_pattern, last_line):
            message['content'] =  message['content'] + "\n"

        st.markdown(f"<div class='message-box {message['alignment_class']}'><div class='message-label'>{label}</div><div class='message-text {message['text_class']}'>{message['content']}</div></div>", unsafe_allow_html=True)


def display_editable_conversation(conversation):
    """Display the conversation for editing, including the system message as the first element."""
    edited_conversation = []
    for idx, message in enumerate(conversation):
        user_label = "System" if message['role'] == 'system' else ("You" if message['role'] == 'user' else "AI")
        # Use a read-only field for the system message to prevent editing
        if message['role'] == 'system':
            edited_message = st.text_area(f"{user_label} (Message {idx+1}):", value=message['content'], disabled=True, key=f"message_{idx}")
        else:
            edited_message = st.text_area(f"{user_label} (Message {idx+1}):", value=message['content'], key=f"message_{idx}")
        edited_conversation.append({"role": message['role'], "content": edited_message})
    return edited_conversation

def process_edited_conversation(edited_conversation):
    """Process the edited conversation."""
    new_conversation = []
    for message in edited_conversation:
        if message['content'].strip():  # Ensure that empty messages are not included
            new_conversation.append(message)
    return new_conversation

def export_conversation(conversation):
    """Converts the conversation history to a text format, including system prompts."""
    conversation_text = ""
    for message in conversation:
        if message['role'] == 'system':
            prefix = "*_* System: "
        elif message['role'] == 'user':
            prefix = "*_* You: "
        else:  # Assuming the only other role is 'assistant'
            prefix = "*_* Chappy: "
        conversation_text += prefix + message['content'] + "\n\n"
    return conversation_text


def update_system_prompt():
    new_system_prompt = st.session_state.new_system_prompt
    if new_system_prompt:
        # Check if the first message is a system prompt and update it
        if 'conversation' in st.session_state and st.session_state['conversation']:
            if st.session_state['conversation'][0]['role'] == 'system':
                st.session_state['conversation'][0]['content'] = new_system_prompt
            else:
                st.session_state['conversation'].insert(0, {"role": "system", "content": new_system_prompt})
        else:
            st.session_state['conversation'] = [{"role": "system", "content": new_system_prompt}]

def get_image_base64(image):
    base64_image = base64.b64encode(image.read()).decode('utf-8')
    return base64_image



# Setting Page Title, Page Icon and Layout Size
st.set_page_config(
    page_title='Chappy',
    page_icon="logoF.png",
    layout='wide',
    initial_sidebar_state="expanded",
    menu_items={
    'Get Help': 'https://github.com/tatonetti-lab/chappy',
    'Report a bug': "https://github.com/tatonetti-lab/chappy/issues",
    'About': "# This is a header. This is about *chappy* !"
}
    
)

st.markdown(
    r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True
)

def main():
    """Main function to run the Streamlit application."""

    # Initialize user_tag in session_state if not present
    if 'user_tag' not in st.session_state:
        st.session_state['user_tag'] = ""

    # Display tag input field only if the tag is not set
    if not st.session_state['user_tag']:
        st.title("Welcome to Chappy")
        st.subheader("A product of the Tatonetti Lab")
        st.text("Enter your user tag to start:")

        initial_user_tag = st.text_input("Enter your user tag:", key="initial_user_tag",type="password")
        if initial_user_tag:
            # Check if the user tag is in the config file under the 'users' section
            if initial_user_tag in config['users'].values():
                st.session_state['user_tag'] = initial_user_tag
                st.experimental_rerun()
            else:
                st.error("User tag not recognized. Please enter a valid user tag.")

    # Check if user_tag is set and proceed with the rest of the application
    if st.session_state['user_tag']:
        # Sidebar Configuration
        st.sidebar.title("Settings")
        # Option to change user tag in the sidebar
        user_tag_input = st.sidebar.text_input("Change your user tag:", value="", key="user_tag_input")
        if st.sidebar.button("Update Tag", key="update_tag_button"):
            if user_tag_input:
                st.session_state['user_tag'] = user_tag_input
                st.experimental_rerun()
            else:
                st.sidebar.error("Please enter a valid user tag.")
        model=st.sidebar.radio('Select from the following models:', ['GPT-35-Turbo', 'GPT-4'])
        st.sidebar.title(f"Using: {model}")
        st.sidebar.divider()
        st.sidebar.header("Model Parameters")
        # Get user input for model parameters and provide validation
        new_tokens = st.sidebar.number_input("New Tokens (number of new tokens to be generated)", value=516)
        temperature = st.sidebar.number_input("Temperature (randomness of response)", value=0.10)
        tag=st.session_state['user_tag']

        try:
            new_tokens = int(new_tokens)
        except ValueError:
            st.sidebar.warning("Please enter a valid integer for new tokens.")
        try:
            temperature = float(temperature)
            if temperature < 0 or temperature > 1:
                raise ValueError
        except ValueError:
            st.sidebar.warning("Please enter a value between 0 and 1.")
            



        # Create a two-column layout
        col1, col2 = st.columns([3, 1]) # this will just call methods directly in the returned objects

        # Inside the first column, add the answer text
        with col1:
            # Main Application Content
            st.title('Chappy')
            st.subheader('Built safe for PHI')
            functionality=st.radio('Select from the following:', ['Chat','Quick Analysis Script Writer', 'Graphic Generation'])

        # Inside the second column, add the image
        with col2:
            st.image("logo.png", use_column_width=True)

        if (functionality == 'Quick Analysis Script Writer'):
            top_k = st.sidebar.number_input("Top k (number of observations to base data summary on)", value=5)
            display_data=st.sidebar.toggle("Display csv?", value=True)
        
            st.subheader('Please upload your CSV file and enter your request below:')
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                # Display details of the uploaded file
                file_details = {
                    "FileName": uploaded_file.name,
                    "FileType": uploaded_file.type,
                    "FileSize": uploaded_file.size
                }
                st.write(file_details)

                # Save the uploaded file to a temporary location and read its content
                file_path = os.path.join("./data/tmp", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                df = pd.read_csv(file_path)
                if display_data:
                    st.dataframe(df)

                # Get or set session states for CSV description and suggested queries
                if "csv_description" not in st.session_state:
                    st.session_state.csv_description = ""
                st.subheader("Enter a brief description of the CSV, including any relevant columns.")

                # Auto-fill button to generate a dataset summary
                if st.button("Auto-Fill"):
                    column_data_types = df.dtypes
                    columns_info = '\n'.join([f"Column Name: {col}\nData Type: {dtype}" for col, dtype in column_data_types.items()])
                    st.session_state.csv_description =  llm(
                        prompt = ("The dataset head is: " + df.head(top_k).to_string() + "\n\n" + columns_info),
                        system = f"""         
                            You are a bot whose purpose is to summarize the data format based upon the column names, data types, and dataset head. The head is only for example for you to understand the structure, do not use these values in the response. Return nothing except the formatting. Write your answer for the dataset descriptions in the exact format:     
                                Column Name: [Column1 Name]
                                Data Type: [Type, e.g., Integer, String, Date]
                                Description: [Brief summary of what this column contains, any special notes about its content.]

                                Column Name: [Column2 Name]
                                Data Type: [Type, e.g., Integer, String, Date]
                                Description: [Brief summary of what this column contains, any special notes about its content.]
                                [... Repeat for all columns ...] """,
                        model=model,
                        temperature=temperature,
                        max_new_tokens = new_tokens,
                        tag=tag
                    )
                    st.session_state.csv_filename = uploaded_file.name

                # Display the dataset summary and allow user to modify
                csv_description = st.text_area(label='Be sure to look over and adjust as needed.',value=st.session_state.csv_description)

                # Get or set session state for suggested queries
                if "suggested_queries" not in st.session_state:
                    st.session_state.suggested_queries = ""
                
                # Auto-fill button to suggest analyses
                if st.button("Suggest Analyses"):
                    st.session_state.suggested_queries = llm(
                        prompt = ("The dataset summary is: " + csv_description),
                        system = 'Provide 3 insightful analysis suggestions based on the dataset summary.',
                        model=model,
                        temperature=temperature,
                        max_new_tokens = new_tokens,
                        tag=tag
                    )
                st.write(st.session_state.suggested_queries)

                # Take user input for their analysis/query
                user_input = st.text_area("Your Request")

                # Get or set session state for the generated code
                if "generated_code" not in st.session_state:
                    st.session_state.generated_code = ""

                # Generate Python code based on the user's query
                if st.button('Generate Code'):
                    with st.spinner('Writing Script...'): 
                        response = llm(
                            prompt = user_input,
                            system = f'''Write a python script to address the user instructions using the following dataset: ##{csv_description}##. Load the data from: ##{st.session_state.csv_filename} ##.''',
                            max_new_tokens = new_tokens,
                            temperature = temperature,
                            tag=st.session_state['user_tag']
                        )
                        st.session_state.generated_code = response
                        st.write(st.session_state.generated_code)

        if (functionality == 'Graphic Generation'):
            # Add top k to sidebar
            top_k = st.sidebar.number_input("Top k (number of observations to base data summary on)", value=5)
            display_data=st.sidebar.toggle("Display csv?", value=True)

            st.subheader('Please upload your CSV file and enter your request below:')
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                # Display details of the uploaded file
                file_details = {
                    "FileName": uploaded_file.name,
                    "FileType": uploaded_file.type,
                    "FileSize": uploaded_file.size
                }
                st.write(file_details)

                # Save the uploaded file to a temporary location and read its content
                file_path = os.path.join("./data/tmp", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                df = pd.read_csv(file_path)
                if display_data:
                    st.dataframe(df)

                # Get or set session states for CSV description and suggested queries
                if "csv_description" not in st.session_state:
                    st.session_state.csv_description = ""
                st.subheader("Enter a brief description of the CSV, including any relevant columns.")

                # Auto-fill button to generate a dataset summary
                if st.button("Auto-Fill"):
                    column_data_types = df.dtypes
                    columns_info = '\n'.join([f"Column Name: {col}\nData Type: {dtype}" for col, dtype in column_data_types.items()])
                    st.session_state.csv_description =  llm(
                        prompt = ("The dataset head is: " + df.head(top_k).to_string() + "\n\n" + columns_info),
                        system = f"""         
                            You are a bot whose purpose is to summarize the data format based upon the column names, data types, and dataset head. The head is only for example for you to understand the structure, do not use these values in the response. Return nothing except the formatting. Write your answer for the dataset descriptions in the exact format:     
                                Column Name: [Column1 Name]
                                Data Type: [Type, e.g., Integer, String, Date]
                                Description: [Brief summary of what this column contains, any special notes about its content.]

                                Column Name: [Column2 Name]
                                Data Type: [Type, e.g., Integer, String, Date]
                                Description: [Brief summary of what this column contains, any special notes about its content.]
                                [... Repeat for all columns ...] """,
                        model=model,
                        temperature=temperature,
                        max_new_tokens = new_tokens,
                        tag=tag
                    )
                    st.session_state.csv_filename = uploaded_file.name

                # Display the dataset summary and allow user to modify
                csv_description = st.text_area(label='Be sure to look over and adjust as needed.',value=st.session_state.csv_description)

                # Get or set session state for suggested queries
                if "suggested_plots" not in st.session_state:
                    st.session_state.suggested_plots = ""
                    st.session_state.suggested_plots_code = ""

                # Auto-fill button to suggest analyses
                if st.button("Suggest Plots"):
                    st.session_state.suggested_plots = llm(
                        prompt = ("The dataset summary is: " + csv_description),
                        system = 'Provide 2 insightful plot suggestions based on the dataset summary. Treat all objects in the description as strings. Avoid using dates in your suggestions.', 
                        model=model,
                        temperature=temperature,
                        max_new_tokens = new_tokens,
                        tag=tag
                    )
                    st.session_state.suggested_plots_code = llm(
                        prompt = f"Description: {st.session_state.suggested_plots}",
                        system = f"Based on the description given, return strictly python code to generate plotly graphics named plot1 and plot2. Very importantly, do not return any text besides the script and do not show the plots. Always use markdown formatting. Use the following dataset: ##{csv_description}## and load the data from: ##{file_path}##. Most imoprtantly, ensure that the code can run without error,", 
                        model=model,
                        temperature=temperature,
                        max_new_tokens = new_tokens,
                        tag=tag
                    )
                st.write(st.session_state.suggested_plots)
                st.write(st.session_state.suggested_plots_code)
                # Conditional execution to generate plots
                if 'suggested_plots_code' in st.session_state and st.session_state.suggested_plots_code:
                    if st.button("Generate Suggested Plots"):
                        cleaned_code = re.findall(r"```python(.*?)```", st.session_state.suggested_plots_code, re.DOTALL)
                        cleaned_code = '\n'.join(cleaned_code).strip()
                        # st.session_state.suggested_plots_code.replace("```python", "").replace("```", "")
                        # Create a dictionary to hold variables defined in the exec() scope
                        local_vars = {}
                        exec(cleaned_code, globals(), local_vars)

                        # Check if plot1 and plot2 are defined in the local_vars dictionary
                        if 'plot1' in local_vars and 'plot2' in local_vars:
                            st.session_state.suggested_plot1 = local_vars['plot1']
                            st.session_state.suggested_plot2 = local_vars['plot2']

                # Conditional display of plots
                if 'suggested_plot1' in st.session_state and st.session_state.suggested_plot1:
                    st.plotly_chart(st.session_state.suggested_plot1)
                else:
                    st.session_state.suggested_plot1 = None

                if 'suggested_plot2' in st.session_state and st.session_state.suggested_plot2:
                    st.plotly_chart(st.session_state.suggested_plot2)
                else:
                    st.session_state.suggested_plot2 = None

                if "user_plots" not in st.session_state:
                    st.session_state.user_plots = ""
                    st.session_state.user_plots_code = ""       

                st.subheader("Enter your custom plot request:")
                st.session_state.user_plot = st.text_area("Your Request")


                # Generate Python code based on the user's query
                if st.button('Generate Code') and st.session_state.user_plot:
                    # Take user input for their analysis/query
                    with st.spinner('Writing Script...'): 
                        st.session_state.user_plots_code = llm(
                            prompt = f"Description: {st.session_state.user_plot}",
                            system = f"Based on the description given, return strictly python code to generate a plotly graphic named fig with: fig = go.Figure(). Very importantly, do not return any text besides the script. Do not use fig.show() as we are printing the graph later. Always use markdown formatting. Use the following dataset: ##{csv_description}## and load the data from: ##{file_path}##. Most imoprtantly, ensure that the code can run without error,", 
                            model=model,
                            temperature=temperature,
                            max_new_tokens = new_tokens,
                            tag=tag
                        )

                # Toggle for editing the code
                if st.session_state.user_plot:
                        if st.toggle("Edit Code",False):
                            st.session_state.user_plots_code = st.text_area("Edit the Code Here", value=st.session_state.user_plots_code)

                st.write(st.session_state.user_plots_code)
                if 'user_plots_code' in st.session_state and st.session_state.user_plots_code:
                    if st.button("Generate User Plot"):
                        cleaned_code = re.findall(r"```python(.*?)```", st.session_state.user_plots_code, re.DOTALL)
                        cleaned_code = '\n'.join(cleaned_code).strip()
                        local_vars = {}
                        exec(cleaned_code, globals(), local_vars)

                        # Check if user_plot is defined in the local_vars dictionary
                        if 'fig' in local_vars:
                            st.session_state.user_generated_plot = local_vars['fig']

                # Conditional display of user plot
                if 'user_generated_plot' in st.session_state and st.session_state.user_generated_plot:
                    st.plotly_chart(st.session_state.user_generated_plot)
                    # Create an in-memory buffer
                    buffer = io.BytesIO()
                    st.session_state.user_generated_plot.write_image(file=buffer, format="pdf")
                    # Download the pdf from the buffer
                    st.download_button(
                        label="Download PDF",
                        data=buffer,
                        file_name="figure.pdf",
                        mime="application/pdf",
                    )

                    #st.plotly_chart(fig)

                    
                    # Display the plot
                    #st.plotly_chart(st.session_state.user_generated_plot)
                else:
                    st.session_state.user_generated_plot = None

        if (functionality == 'Chat'):

            #Default system prompt:
            default_system="""Your name is Chappy and you are a chatbot working for the Tatonetti Lab (also called the TLab). You are a helpful, friendly assistant who provides concise and accurate answers. Use markdown format where applicable. This is a brief description of our lab:

We are making drugs safer through the analysis of data. Everyday millions of us or our loved ones take medications to manage our health. We trust in these prescriptions to improve our lives and give us hope for a healthier future. Often, however, these drugs have harmful side effects or dangerous interactions. Adverse drug reactions are experienced by millions of patients each year and cost the healthcare industry billions of dollars. In the Tatonetti Lab, we use advanced data science methods, including artificial intelligence and machine learning, to investigate these medicines. Using emerging resources, such as electronic health records (EHR) and genomics databases, we are working to identify for whom these drugs will be safe and effective and for whom they will not.

**Our past projects:**

- **OnSIDES, side effects extracted from FDA Structured Product Labels**  
[OnSIDES](https://www.tatonettilab.org/onsides-side-effects-extracted-from-fda-structured-product-labels/) is the newest member of the NSIDES family. The initial release (v01) of the OnSIDES database of adverse reactions and boxed warnings extracted from the FDA structured product labels. All labels available to download from DailyMed as of April 2022 were processed in this analysis. In total 2.7 million adverse reactions were extracted from 42,000 labels for just under 2,000 drug ingredients or combination of ingredients.  
We created OnSIDES using the ClinicalBERT language model and 200 manually curated labels available from Denmer-Fushman et al.. The model achieves an F1 score of 0.86, AUROC of 0.88, and AUPR of 0.91 at extracting effects from the ADVERSE REACTIONS section of the label and an F1 score of 0.66, AUROC of 0.71, and AUPR of 0.60 at extracting effects from the BOXED WARNINGS section.  
Read more at [http://nsides.io/](http://nsides.io/).  
Cite this resource as  
Makar AB, McMartin KE, Palese M, Tephly TR. Formate assay in body fluids: application in methanol poisoning. Biochem Med. 1975;13(2):117-126. doi:10.1016/0006-2944(75)90147-7

- **Sex-specific side effects (AwareDX)**  
[AwareDX](https://www.tatonettilab.org/sex-specific-side-effects-awaredx/) Adverse drug effects posing sex-specific risks. Risks in this database were predicted by AwareDX to publicly available data. Over 20,000 sex risks spanning over 800 drugs and 300 side effects.  
Download the data at [AwareDX Data](http://tatonettilab.org/data/AwareDX_Data.xlsx) or browse the repository at [GitHub](https://github.com/tatonetti-lab/sex_risks).  
Cite this resource as  
Chandak P, Tatonetti NP. Using Machine Learning to Identify Adverse Drug Effects Posing Increased Risk to Women. Patterns (N Y). 2020;1(7):100108. doi:10.1016/j.patter.2020.100108

- **OffSIDES and TwoSIDES**  
[OffSIDES and TwoSIDES](https://www.tatonettilab.org/offsides/) Drug side effects and drug-drug interactions were mined from publicly available data. OffSIDES is a database of drug side-effects that were found, but are not listed on the official FDA label. TwoSIDES is the only comprehensive database drug-drug-effect relationships. Over 3,300 drugs and 63,000 combinations connected to millions of potential adverse reactions.  
Read more and access the data at nsides.io.  
Cite this resource as  
Tatonetti NP, Ye PP, Daneshjou R, Altman RB. Data-driven prediction of drug effects and interactions. Sci Transl Med. 2012;4(125):125ra31. doi:10.1126/scitranslmed.3003377

- **Family Relationship and Disease Data**  
[Family Relationship and Disease Data](https://www.tatonettilab.org/family-relationship-and-disease-data/) De-identified family data on over 3,000 conditions at two sites. Data are from approximately 1.5 million patients across the two sites and all identifying information has been removed. Further, ages have been replaced with a random poisson distribution with lambda set to the actual age of the patient. Data are compatible with the observation heritability estimation software.  
If you would like the 500 significant traits as reported in Polubriaginof, et al. in Cell, go to this page at [RIFTEHR](http://riftehr.tatonettilab.org/).  
All code to generate the relationships from hospital data is publicly available in our RIFTEHR github at [GitHub](https://github.com/tatonetti-lab/riftehr).  
Cite this resource as  
Polubriaginof FCG, Vanguri R, Quinnies K, et al. Disease Heritability Inferred from Familial Relationships Reported in Medical Records. Cell. 2018;173(7):1692-1704.e11. doi:10.1016/j.cell.2018.04.032

- **DATE**  
[DATE](https://www.tatonettilab.org/date/) Downstream effects of targeted proteins is essential to drug design. We introduce a data-driven method named DATE, which integrates drug-target relationships with gene expression, protein-protein interaction, and pathway annotation data to connect Drugs to target pAthways by the Tissue Expression. Links drugs to tissue-specific target pathways.  
467,396 connections for 1,034 drugs and 954 pathways in 259 tissues/cell lines available at [DATE Resource](http://tatonettilab.org/resources/DATE/date_resource.zip).  
Cite this resource as  
Hao Y, Quinnies K, Realubit R, Karan C, Tatonetti NP. Tissue-Specific Analysis of Pharmacological Pathways. CPT Pharmacometrics Syst Pharmacol. 2018;7(7):453-463. doi:10.1002/psp4.12305

- **GOTE**  
[GOTE](https://www.tatonettilab.org/gote/) G protein-coupled receptors (GPCRs) are central to how cells respond to their environment and a major class of pharmacological targets. We developed a data-driven method named GOTE, that connects Gpcrs to dOwnstream cellular pathways by the Tissue Expression. Links G-protein coupled receptors to tissue-specific molecular pathways.  
93,012 connections for 213 GPCRs and 654 pathways in 196 tissues/cell types available. Code available here at [GOTE Source Code](http://tatonettilab.org/resources/GOTE/source_code/).  
Cite this resource as  
Hao Y, Tatonetti NP. Predicting G protein-coupled receptor downstream signaling by tissue expression. Bioinformatics. 2016;32(22):3435-3443. doi:10.1093/bioinformatics/btw510

- **MADSS**  
#algorithm#side effects Network analysis framework that identifies adverse event (AE) neighborhoods within the human interactome (protein-protein interaction network). Drugs targeting proteins within this neighborhood are predicted to be involved in mediating the AE. Links drugs to seed sets of proteins and phenotypes, like drug side-effects and diseases.  
A description of the algorithm is available here at [MADSS](http://madss.tatonettilab.org/). Code in Python available on GitHub at [GitHub](http://www.github.com/tal-baum/MADSS).  
Cite this resource as  
Lorberbaum T, Nasir M, Keiser MJ, Vilar S, Hripcsak G, Tatonetti NP. Systems pharmacology augments drug safety surveillance. Clin Pharmacol Ther. 2015;97(2):151-158. doi:10.1002/cpt.2

- **VenomKB**  
[VenomKB](https://www.tatonettilab.org/venomkb/) The world’s first comprehensive knowledge base for therapeutic uses of venoms. As of its original release, contains 39,000 mined from MEDLINE describing potentially therapeutic effects of venoms on the human body. Links venom compounds to physiological effects.  
39K venom/effect associations in three databases available for download. Code available on GitHub at [GitHub](http://www.github.com/jdromano2/venomkb).  
Cite this resource as  
Romano JD, Tatonetti NP. VenomKB, a new knowledge base for facilitating the validation of putative venom therapies. Sci Data. 2015;2:150065. Published 2015 Nov 24. doi:10.1038/sdata.2015.65

- **SINaTRA**  
[SINaTRA](https://www.tatonettilab.org/sinatra/) Interspecies, network-based predictions of synthetic lethality and the first genome-wide scale prediction of synthetic lethality in humans. Scores were validated against three independent databases of synthetic lethal pairs in humans, mouse, and yeast. The original release contains ~109 million gene pairs with their associated synthetic lethality scores.  
Human synthetic lethal gene pairs available in 3 parts: [part 1](https://figshare.com/articles/Human_Synthetic_Lethal_Predictions_1_3_/1501103), [part 2](https://figshare.com/articles/Human_Synthetic_Lethal_Predictions_2_3_/1501105), and [part 3](https://figshare.com/articles/Human_Synthetic_Lethal_Predictions_3_3_/1501115). And mouse too at [Mouse Synthetic Lethal Predictions](https://figshare.com/articles/Mouse_Synthetic_Lethal_Predictions/1519150).  
Cite this resource as  
Jacunski A, Dixon SJ, Tatonetti NP. Connectivity Homology Enables Inter-Species Network Models of Synthetic Lethality. PLoS Comput Biol. 2015;11(10):e1004506. Published 2015 Oct 9. doi:10.1371/journal.pcbi.1004506
            """


            # Initialize conversation in session_state if not present
            if 'conversation' not in st.session_state or not st.session_state['conversation']:
                st.session_state['conversation'] = []
                st.session_state['conversation'].insert(0, {"role": "system", "content": default_system})

            # Format and display the conversation
            formatted_messages = format_conversation(st.session_state['conversation'])
            display_chat(formatted_messages)

            if 'reset_input' in st.session_state and st.session_state.reset_input:
                st.session_state.user_input = ""
                st.session_state.reset_input = False

            st.write("")

            with st.form(key='message_form'):
                # Use session state to hold the value of the input box
                user_input = st.text_area("Type your message here...", key="user_input", height=100, value=st.session_state.get('user_input', ''))
                send_pressed = st.form_submit_button("Send")

            if send_pressed and user_input:
                # Update conversation history with user input
                st.session_state['conversation'].append({"role": "user", "content": user_input})

                # Get Chappy response
                Chappy_response = conversation(st.session_state['conversation'], model, tag, temperature, new_tokens)

                # Update conversation history with Chappy response
                st.session_state['conversation'].append({"role": "assistant", "content": Chappy_response})

                # Clear the input box by setting its value in the session state to an empty string
                st.session_state.reset_input = True

                # Rerun the app to update the conversation display
                st.experimental_rerun()

            # Add 'Edit Conversation' button in sidebar if in chat mode
            st.sidebar.divider()
            if st.sidebar.button("Edit Conversation", key="edit_conversation_button"):
                st.session_state['edit_mode'] = True

            # Display editable conversation if in edit mode
            if st.session_state.get('edit_mode', False):
                edited_conversation = display_editable_conversation(st.session_state['conversation'])
                if st.button("Save Changes"):
                    st.session_state['conversation'] = process_edited_conversation(edited_conversation)
                    st.session_state['edit_mode'] = False
                    st.experimental_rerun()
            
            # Add a button to clear the conversation
            if st.sidebar.button("Clear Conversation", key="clear_conversation_button"):
                st.session_state['conversation'] = []  # Reset the conversation
                st.experimental_rerun()  # Rerun the app to update the conversation display

            new_system_prompt = st.sidebar.text_area("System Prompt:", value=st.session_state['conversation'][0]['content'], key="new_system_prompt", on_change=update_system_prompt)
            
            conversation_text = export_conversation(st.session_state['conversation'])
            # Filename for the download which includes the current date and time for uniqueness
            filename = f"conversation_{pd.Timestamp('now').strftime('%Y-%m-%d_%H-%M-%S')}.txt"
            st.sidebar.download_button(label="Download Conversation",
                                    data=conversation_text,
                                    file_name=filename,
                                    mime='text/plain')
            

            st.sidebar.title("Upload an image:")
            uploaded_image = st.sidebar.file_uploader("👀", type=['jpg', 'jpeg', 'png'])
            if uploaded_image is not None:
                st.session_state['uploaded_image'] = uploaded_image
                if st.sidebar.button('Process Uploaded Image'):
                    # Convert the uploaded image to base64
                    base64_image = get_image_base64(uploaded_image)

                    # Call the vision API with the base64 image
                    Chappy_response = call_vision_api(base64_image, model="vision", max_tokens=512)

                    # Update conversation history with Chappy's response
                    st.session_state['conversation'].append({"role": "assistant", "content": Chappy_response})

                    st.session_state['uploaded_image'] = None

                    # Rerun the app to update the conversation display, but now it won't reprocess the image
                    st.experimental_rerun()

            st.sidebar.divider()
            st.sidebar.title("Upload past conversation:")
            uploaded_file = st.sidebar.file_uploader("🧠", type=['txt'], key="file_uploader")

            if uploaded_file is not None:
                # Use a session state variable to hold the file temporarily
                st.session_state['uploaded_file'] = uploaded_file
                
                # Provide a button to confirm the processing of the uploaded file
                if st.sidebar.button('Process Uploaded Conversation'):
                    # Ensure there's a file to process
                    if 'uploaded_file' in st.session_state:
                        uploaded_file = st.session_state['uploaded_file']
                        
                        # Read and process the file
                        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                        string_data = stringio.read()
                        
                        # Parse the uploaded conversation
                        uploaded_conversation = []
                        for line in string_data.split("*_* "):
                            if line.startswith("You: "):
                                message_content = line.replace("You: ", "", 1)
                                role = 'user'
                            elif line.startswith("Chappy: "):
                                message_content = line.replace("Chappy: ", "", 1)
                                role = 'assistant'  # Adjust based on your application's roles
                            elif line.startswith("System: "):
                                message_content = line.replace("System: ", "", 1)
                                role = 'system'  # Adjust based on your application's roles
                            else:
                                continue  # Skip lines that don't match the expected format
                            
                            uploaded_conversation.append({"role": role, "content": message_content})
                        
                        # Update the current conversation
                        st.session_state['conversation'] = uploaded_conversation
                        st.session_state['uploaded_file'] = None  # Clear the uploaded file after processing
                        
                        # Inform the user of success and refresh the display
                        st.sidebar.success("Uploaded conversation processed successfully.")
                        st.experimental_rerun()
                    else:
                        st.sidebar.error("No file uploaded.")

            
if __name__ == "__main__":
    main()
