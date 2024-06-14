import streamlit as st
import openai
from generate_recommendations import (
    load_and_preprocess_data,
    perform_clustering,
    generate_suggestions,
    generate_shopping_list,
    initialize_vectorstore,
    ask_question,
    generate_food_recipes,
    generate_personalized_promotion,
    get_customer_data
)
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set the OpenAI API key
openai_api_key = os.getenv('openai_API')
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")
openai.api_key = openai_api_key

st.set_page_config(page_title="Retail AI Solutions", page_icon="🛍️", layout="wide")

st.sidebar.title("Retail AI Solutions")
page = st.sidebar.selectbox("Select a Solution", [
    "Personalized product suggestions",
    "Customized shopping lists",
    "Personalized promotions",
    "Chat-bots and virtual assistants",
    "Food recipes suggestions"
])

st.sidebar.markdown("---")
st.sidebar.write("Created by [Team Innovision](https://TeamInnovision.com)")

st.title("Retail AI Solutions Dashboard")
st.write("Select an option from the sidebar to get started.")

st.markdown("### Upload Data Files")
trx_file = st.file_uploader("Upload Transactions File", type=["csv"])
item_file = st.file_uploader("Upload Item Info File", type=["csv"])

if trx_file and item_file:
    st.success("Files uploaded successfully!")

    data = load_and_preprocess_data(trx_file, item_file)
    cluster_data = perform_clustering(data)

    st.write("Data preprocessing and clustering completed.")

    if page == "Personalized product suggestions":
        st.markdown("### Enter Customer Code")
        customer_code = st.text_input("Customer Code:")
        if st.button("Search") and customer_code:
            purchase_history = data[data['customer_code'] == customer_code]['item_name'].tolist()
            if purchase_history:
                suggestions = generate_suggestions(purchase_history)
                st.subheader("Personalized Product Suggestions")
                st.write(suggestions)
            else:
                st.error("No purchase history found for this customer code.")

    elif page == "Customized shopping lists":
        st.markdown("### Enter Customer Code")
        customer_code = st.text_input("Customer Code:")
        if st.button("Search") and customer_code:
            customer_data = data[data['customer_code'] == customer_code]
            if not customer_data.empty:
                purchase_history = customer_data['item_name'].tolist()
                customer_profile = customer_data.iloc[0].to_dict()
                shopping_list = generate_shopping_list(purchase_history, customer_profile)
                st.subheader("Customized Shopping List")
                st.write(shopping_list)
            else:
                st.error("No purchase history found for this customer code.")

    elif page == "Personalized promotions":
        st.markdown("### Enter Customer Code")
        customer_code = st.text_input("Customer Code:")
        if st.button("Generate Promotion") and customer_code:
            customer_data = get_customer_data(cluster_data, customer_code)
            if customer_data is not None:
                promotion = generate_personalized_promotion(customer_data)
                st.subheader("Personalized Promotion")
                st.write(promotion)
            else:
                st.error(f"No data found for customer code: {customer_code}")

    elif page == "Chat-bots and virtual assistants":
        st.markdown("### Ask a Question")
        question = st.text_input("Ask the Question")
        if st.button("Ask") and question:
            vectorstore = initialize_vectorstore(data)
            answer = ask_question(vectorstore, question)
            st.subheader("Answer")
            st.write(answer)

    elif page == "Food recipes suggestions":
        st.markdown("### Enter Ingredients")
        ingredients = st.text_area("Ingredients (comma separated):")
        if st.button("Suggest Recipes") and ingredients:
            recipe_suggestions = generate_food_recipes(ingredients, openai_api_key)
            st.subheader("Recipe Suggestions")
            st.write(recipe_suggestions)
else:
    st.info("Please upload the required files to proceed.")
