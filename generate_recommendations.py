import pandas as pd
import numpy as np
from langchain_community.chat_models import ChatOpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import openai
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub
import faiss
from langchain_experimental.agents import create_pandas_dataframe_agent
import datetime as dt
import plotly.express as px

# Load environment variables
load_dotenv()

# Set the OpenAI API key
openai_api_key = os.getenv('openai_API')
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")
openai.api_key = openai_api_key


def load_and_preprocess_data(trx_file, item_file):
    trx = pd.read_csv(trx_file)
    item = pd.read_csv(item_file)
    trx['transaction_time'] = pd.to_datetime(trx['transaction_time'])
    trx = trx.drop_duplicates()

    threshold = 3
    for col in ['sales_quantity']:
        mean = trx[col].mean()
        std_dev = trx[col].std()
        trx = trx[(trx[col] >= mean - threshold * std_dev) & (trx[col] <= mean + threshold * std_dev)]

    label_encoder = LabelEncoder()
    trx['item_code_encoded'] = label_encoder.fit_transform(trx['item code'])
    data = pd.merge(trx, item, on='item code', how='left')
    data = data.sort_values(by=['customer_code', 'transaction_time'])

    return data


def perform_clustering(data, num_clusters=8):
    current_date = data['transaction_time'].max()

    customer_metrics = data.groupby('customer_code').agg({
        'transaction_time': [
            lambda x: (current_date - x.max()).days,
            lambda x: (current_date - x.min()).days,
            'count'
        ],
        'sales_quantity': 'sum'
    })

    customer_metrics.columns = ['recency', 'tenure', 'frequency', 'total_sales']
    customer_metrics.reset_index(inplace=True)

    unique_items = data.groupby('customer_code')['item code'].nunique().reset_index()
    unique_items.columns = ['customer_code', 'unique_items_purchased']

    most_frequent_item = data.groupby(['customer_code', 'item code']).agg({
        'sales_quantity': 'sum'
    }).reset_index().sort_values(by=['customer_code', 'sales_quantity'], ascending=[True, False])

    most_frequent_item = most_frequent_item.groupby('customer_code').first().reset_index()
    most_frequent_item.columns = ['customer_code', 'most_frequent_item', 'most_frequent_item_quantity']

    avg_quantity_per_item = data.groupby('customer_code').agg({
        'sales_quantity': 'mean'
    }).reset_index()
    avg_quantity_per_item.columns = ['customer_code', 'average_quantity_per_item']

    customer_metrics = customer_metrics.merge(unique_items, on='customer_code', how='left')
    customer_metrics = customer_metrics.merge(most_frequent_item, on='customer_code', how='left')
    customer_metrics = customer_metrics.merge(avg_quantity_per_item, on='customer_code', how='left')

    pivot_count = data.pivot_table(index='customer_code', columns='item code', values='transaction_time',
                                   aggfunc=lambda x: len(x), fill_value=0, margins=False)
    pivot_count.columns = [f'{col}_count' for col in pivot_count.columns]

    pivot_sum = data.pivot_table(index='customer_code', columns='item code', values='sales_quantity',
                                 aggfunc='sum', fill_value=0, margins=False)
    pivot_sum.columns = [f'{col}_sales' for col in pivot_sum.columns]

    merged_pivot = pd.concat([pivot_count, pivot_sum], axis=1)
    merged_pivot = merged_pivot.reindex(sorted(merged_pivot.columns), axis=1)
    merged_pivot = merged_pivot.reset_index()

    scaler = StandardScaler()
    numerical_cols = [col for col in merged_pivot.columns if 'count' in col or 'sales' in col]
    X = merged_pivot[numerical_cols]
    X = scaler.fit_transform(X)

    pca = PCA(0.9)
    X_pca = pca.fit_transform(X)

    km_model = KMeans()
    visualizer = KElbowVisualizer(km_model, k=(2, 15))
    visualizer.fit(X_pca)
    optimal_k = visualizer.elbow_value_

    km_model = KMeans(n_clusters=optimal_k)
    km_model.fit(X_pca)
    merged_pivot['cluster'] = km_model.labels_

    cluster_results = merged_pivot[['customer_code', 'cluster']]

    customer_data = customer_metrics.merge(merged_pivot[['customer_code', 'cluster']], on='customer_code')
    cluster_data = data.merge(cluster_results, on='customer_code')
    return cluster_data,customer_metrics


def create_cluster_summary(cluster_data):
    cluster_summary = cluster_data.groupby('cluster').agg({
        'customer_code': 'nunique',
        'sales_quantity': 'sum',
        'item_category': lambda x: x.value_counts().head(3).to_dict()
    }).reset_index()

    cluster_summary['top_item_categories'] = cluster_summary['item_category'].apply(
        lambda x: ', '.join([f"{k}: {v}" for k, v in x.items()]))
    cluster_summary.drop(columns=['item_category'], inplace=True)
    return cluster_summary

# Function to prepare data for OpenAI
def prepare_data_for_openai(cluster_summary):
    data_list = []
    for index, row in cluster_summary.iterrows():
        # Customize the prompt based on your aggregation
        prompt = f"Cluster {row['cluster']} consists of {row['customer_code']} customers. They have purchased a total of {row['sales_quantity']} items. The top three item categories they purchased are: {row['top_item_categories']}. Provide insights about this customer segment."

        # Create message structure for chatbot context
        message = {
            "role": "user",
            "content": prompt
        }
        data_list.append(message)
    return data_list

# Function to get the assistant's response using the OpenAI API.
def get_assistant_response(messages):
    r = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        temperature=0.5,  # Controls randomness. Lower values make responses more deterministic.
        max_tokens=128,  # Maximum length of the response.
        top_p=1  # Controls diversity. Setting to 1.0 takes the most likely next words.
    )
    response = r.choices[0].message['content']
    return response

# Function to generate insights for each cluster
def generate_insights(cluster_summary):
    messages = [
        {
            "role": "system",
            "content": "You are Marv, a retail specialized chatbot assign name for the cluster and give insights "
        }
    ]

    data_for_openai = prepare_data_for_openai(cluster_summary)
    insights = {}
    for message in data_for_openai:
        try:
            response = get_assistant_response(messages + [message])
            insights[message['content']] = response
        except Exception as e:
            print(f"Error generating response for prompt: {message['content']}, Error: {str(e)}")
    return insights


def generate_suggestions(purchase_history):
    prompt = f"Given the following purchase history: {purchase_history}, suggest new products the customer might like."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
    )
    return response.choices[0].message['content'].strip()


def generate_shopping_list(purchase_history, customer_profile):
    prompt = (
        f"Based on the following purchase history: {purchase_history}, "
        f"and the customer's profile: {customer_profile}, generate a customized shopping list for the next week."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()


def initialize_vectorstore(data):
    data['combined_text'] = data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_texts(data['combined_text'].tolist(), embeddings)
    return vectorstore


def ask_question(vectorstore, question):
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vectorstore.as_retriever())
    return qa.run(question)
def generate_personalized_promotion(customer_row):
    customer_code = customer_row['customer_code']
    top_items = customer_row['top_items']
    top_categories = customer_row['top_categories']

    promotion_message = f"Dear {customer_code},\n\nWe have a special offer just for you!\n\n"

    promotion_message += "Top Picks for You:\n"
    for item in top_items:
        promotion_message += f"- {item['item_name'].title()}: Buy 1 Get 1 Free!\n"

    promotion_message += "\nDon't miss these deals in your favorite categories:\n"
    for category in top_categories:
        promotion_message += f"- {category['item_category'].title()}: Up to 20% off!\n"

    return promotion_message

def get_customer_data(df, customer_code):
    customer_row = df[df['customer_code'].str.lower() == customer_code.lower()]
    if len(customer_row) == 0:
        return None
    return customer_row.iloc[0]

def generate_food_recipes(ingredients, openai_api_key):
    # Initialize API wrapper for Wikipedia
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    # Load and split documents from a web source
    loader = WebBaseLoader("https://tasty.co/article/melissaharrison/easy-dinner-recipes")
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=openai_api_key))
    retriever = vectordb.as_retriever()
    retriever_tool = create_retriever_tool(retriever, "recipes","Search delicious recipes for input ingredients. For questions about simple meals, use this tool!")

    # Load and split documents from a PDF source
    loader = PyPDFLoader("cookbook.pdf")
    pages = loader.load_and_split()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(pages)
    vector2db = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=openai_api_key))
    retriever_pdf = vector2db.as_retriever()
    pdf_tool = create_retriever_tool(retriever_pdf, "recipes_pdf","Search delicious recipes for input ingredients. For questions about dinner simple meals, use this tool!")

    # Combine the tools
    tools = [wiki, pdf_tool, retriever_tool]

    # Initialize the language model
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, openai_api_key=openai_api_key)
    prompt = hub.pull("rlm/rag-prompt")
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    user_message = {"role": "user",
                    "content": f"Please suggest some meals that could be made using the following ingredients: {ingredients}."}
    inputs = {"input": [user_message]}
    response = agent_executor.invoke(inputs)

    # Extract and return the assistant's response
    return response['output']


def load_data(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    df['combined_text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    return df

def load_vectorstore(path):
    return faiss.read_index(path)

class FAISSRetriever:
    def __init__(self, index, texts):
        self.index = index
        self.texts = texts

    def retrieve(self, query_embedding, top_k=10):
        D, I = self.index.search(query_embedding, top_k)
        return [self.texts[i] for i in I[0]]

def initialize_agent(df):
    llm = OpenAI(api_key=openai_api_key, temperature=0)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    return agent

def query_agent(agent, query):
    response = agent.invoke({"input": query})
    return response

