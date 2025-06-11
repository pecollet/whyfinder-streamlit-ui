import streamlit as st
import os
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation.prompts import RagTemplate
from dotenv import load_dotenv
from streamlit_agraph import Node, Edge, agraph, Config
from typing import List, Tuple, Dict, Any

#run with : streamlit run whyfinder.py

load_dotenv()
driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))

embedder = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))
retriever = VectorRetriever(
    driver, 
    os.getenv("NEO4J_VECTOR_INDEX"), 
    embedder,
    neo4j_database=os.getenv("NEO4J_DATABASE")
)
llm = OpenAILLM(model_name=os.getenv("OPENAI_LLM_MODEL"), model_params={"temperature": 0})
prompt_template = RagTemplate(
    template="""
    Context:
    {context}
    You are an experienced Neo4j Support Engineer, expert on all technical topics related to Neo4j. 
    Your task is to analyze the symptoms provided by the user and provide a diagnosis based on the information available in the WhyFinder database.
    The database contains (:Observable) nodes representing symptoms, issues, and their possible causes, along with relevant Neo4j metrics, configurations, and log messages.
    The Observable nodes are connected with :MAY_CAUSE relationships to indicate potential causes of the symptoms.
    
    Examples:
    {examples}

    Symptoms: {query_text}

    Provide a detailed analysis and possible causes based on the symptoms.
    """,
    expected_inputs=['context', 'query_text', 'examples'],
    system_instructions='Answer the user question using the provided context.'
)

rag = GraphRAG(
    retriever=retriever, 
    llm=llm,
    prompt_template=prompt_template
)
ready_for_rca=False


def find_symptoms(symptom: str, k: int = 5):
    """
    Function to find symptoms in the WhyFinder database.
    """
    symptom_vector = embedder.embed_query(symptom)
    # print(symptom_vector)
    records, summary, keys = driver.execute_query("""
        CALL db.index.vector.queryNodes($vector_index, $k, $symptom_vector) YIELD node as observable, score
        LIMIT 5                                          
        RETURN observable.name AS title, score""", 
        database_=os.getenv("NEO4J_DATABASE"), 
        routing_="r",
        vector_index=os.getenv("NEO4J_VECTOR_INDEX"), 
        symptom_vector=symptom_vector,
        k=k
    )
    return records

def build_agraph_node(node, red_list: List = [], green_list: List = []) -> Node:
    """
    Helper function to build a streamlit_agraph Node object.
    """
    node_label = list(node.labels)[0] if node.labels else "Node"
    node_name = node.get("name", node_label)  # 'name' property, or fallback to the label
    return Node(
        id=node.element_id,
        label=node_name,
        shape="dot", 
        color="red" if node_name in red_list else "green" if node_name in green_list  else "lightblue"
    )

def transform_neo4j_paths_to_agraph(records, red_list: List = [], green_list: List = []) -> Tuple[List[Node], List[Edge]]:
    """
    Transforms Neo4j records containing a single 'p' path column into node and edge lists
    compatible with the streamlit-agraph component.

    This function correctly de-duplicates nodes and relationships that may appear in
    multiple paths returned by the Cypher query.

    Args:
        records: The result object from a `session.run()` call where the query
                 returns a single column named 'p' containing path data.
                 Example Query: "MATCH p=(a)-[r*]->(b) RETURN p"

    Returns:
        A tuple containing two lists:
        - A list of unique streamlit_agraph.Node objects.
        - A list of unique streamlit_agraph.Edge objects.
    """
    agraph_nodes = []
    agraph_edges = []
    print(f"red_list: {red_list}")
    
    # Use sets to keep track of processed element IDs to avoid duplicates
    seen_node_ids = set()
    seen_edge_ids = set()

    for record in records:
        path = record["p"]
        
        # Process all relationships in the path to ensure all nodes are included
        for rel in path.relationships:
            # 1. Process the start node of the relationship
            start_node = rel.start_node
            if start_node.element_id not in seen_node_ids:
                agraph_nodes.append(build_agraph_node(start_node, red_list, green_list))
                seen_node_ids.add(start_node.element_id)
            
            # 2. Process the end node of the relationship
            end_node = rel.end_node
            if end_node.element_id not in seen_node_ids:
                agraph_nodes.append(build_agraph_node(end_node, red_list, green_list))
                seen_node_ids.add(end_node.element_id)
            
            # 3. Process the relationship itself
            if rel.element_id not in seen_edge_ids:
                agraph_edges.append(Edge(
                    source=rel.start_node.element_id,
                    target=rel.end_node.element_id
                ))
                seen_edge_ids.add(rel.element_id)
                
    return agraph_nodes, agraph_edges

def find_root_causes():
    selected_symptoms = [symptom for symptom, selected in st.session_state["selected_symptoms"].items() if selected]
    print(f"Selected symptoms: {selected_symptoms}")
    records, summary, keys = driver.execute_query("""
        MATCH p=(symptom:Observable)<-[r:MAY_CAUSE]-+(cause:Observable) WHERE symptom.name IN $selected_symptoms RETURN p""", 
        database_=os.getenv("NEO4J_DATABASE"), 
        routing_="r",
        selected_symptoms=selected_symptoms
    )
    nodes, edges = transform_neo4j_paths_to_agraph(records, selected_symptoms)
    config = Config(width=800, height=400, directed=True, physics=True)
    agraph(nodes=nodes, edges=edges, config=config)



symptoms = st.text_area("Symptoms", key="symptoms", placeholder="Enter your symptoms here (1 per line)...", height=200, help="List the symptoms you are experiencing.")

#split by line breaks
symptoms_list = symptoms.splitlines()
# print(symptoms_list)
all_results = []
for i, symptom in enumerate(symptoms_list):
    if symptom.strip():
        results = find_symptoms(symptom)
        # response = rag.search(
        #     query_text=symptom, 
        #     retriever_config={"top_k": 5},
        #     # message_history=[],
        #     # examples="",
        #     return_context=True
        #     )
        # st.write(results)
        all_results.append(results)

if all(not inner_list for inner_list in all_results):
    st.write("No symptoms found. Please try again with different symptoms.")
else:
    ready_for_rca=True
    st.write("The following symptoms were matched in the WhyFinder database. Please select the ones you want to include in the analysis:") 

selected_symptoms = {}
for i, results in enumerate(all_results):
    for j, result in enumerate(results):
        if not result["title"] in selected_symptoms.keys():
            checked = True if j==0 else False # by default only select the top result for each symptom
            selected_symptoms[result["title"]] = st.checkbox(result["title"], checked, key="matched_symptoms_"+result["title"]+"_"+str(i)+"_"+str(j))

st.session_state["selected_symptoms"] = selected_symptoms
#test if any value in selected_symptoms is True
if any(selected_symptoms.values()):
    st.button("Find Root Cause", on_click=find_root_causes)