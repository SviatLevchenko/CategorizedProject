import psycopg2
import stanza
from sentence_transformers import SentenceTransformer
import warnings
import logging

# Suppressing FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Disabling Stanza logging
logging.getLogger('stanza').setLevel(logging.CRITICAL)

# Reducing the log level for transformers
logging.getLogger("transformers").setLevel(logging.ERROR)


# Function to connect to the PostgreSQL database
def connect_db():
    return psycopg2.connect(
        dbname="knowledge",
        user="postgres",
        password="1",
        host="localhost"
    )


# Fetch knowledge items and their categories from the database
def fetch_knowledge_from_db():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT k.item_name, kg.category_name 
        FROM knowledge_items k
        JOIN knowledge_groups kg ON k.group_id = kg.id;
    """)
    knowledge_items = cursor.fetchall()
    cursor.close()
    conn.close()

    return knowledge_items


# Analyze text and classify entities into categories using lemmatization
def analyze_text(text, nlp_pipeline, knowledge_items):
    doc = nlp_pipeline(text)
    categorized_knowledge = {
        "objects": [],  # Updated to English
        "actions": [],
        "descriptions": [],
        "prepositions": []
    }

    # Ignored words, typical adverbs, and punctuation
    ignored_words = {"собі", "мені", "йому", "їй", "тому", "і"}
    typical_adverbs = {"швидко", "тихо", "голосно", "добре"}
    punctuation = {",", ".", "!"}

    for sentence in doc.sentences:
        for word in sentence.words:
            entity = word.lemma.lower()  # Convert to lemma and lowercase
            pos_tag = word.upos

            # Skip ignored words and punctuation
            if entity in ignored_words or entity in punctuation:
                continue

            # Check if the word exists in the database
            for item, category in knowledge_items:
                if entity == item.lower():
                    # Mapping Ukrainian categories to English keys
                    if category == "объекты":
                        category = "objects"
                    elif category == "действия":
                        category = "actions"
                    elif category == "описания":
                        category = "descriptions"
                    elif category == "предлоги":
                        category = "prepositions"

                    categorized_knowledge[category].append(entity)
                    break
            else:
                # If the word is new, determine its category
                new_category = categorize_new_entity(entity, pos_tag, typical_adverbs)
                categorized_knowledge[new_category].append(entity)
                add_new_knowledge_item(entity, new_category)

    return categorized_knowledge


# Determine the category of a new entity based on its part of speech
def categorize_new_entity(entity, pos_tag, typical_adverbs):
    if entity in typical_adverbs:
        return "descriptions"
    if pos_tag in ["NOUN", "PROPN"]:
        return "objects"
    elif pos_tag == "VERB":
        return "actions"
    elif pos_tag in ["ADJ", "ADV"]:
        return "descriptions"
    elif pos_tag == "ADP":
        return "prepositions"
    else:
        return "objects"


# Add a new knowledge item to the database
def add_new_knowledge_item(entity, category):
    conn = connect_db()
    cursor = conn.cursor()

    # Map English category names back to Ukrainian for database consistency
    if category == "objects":
        category = "объекты"
    elif category == "actions":
        category = "действия"
    elif category == "descriptions":
        category = "описания"
    elif category == "prepositions":
        category = "предлоги"

    # Check if the category exists
    cursor.execute("SELECT id FROM knowledge_groups WHERE category_name = %s;", (category,))
    group_id = cursor.fetchone()

    if not group_id:
        # Insert the category if it doesn't exist
        cursor.execute("INSERT INTO knowledge_groups (category_name) VALUES (%s) RETURNING id;", (category,))
        group_id = cursor.fetchone()[0]

    # Insert the knowledge item
    cursor.execute(
        "INSERT INTO knowledge_items (item_name, group_id) VALUES (%s, %s) ON CONFLICT (item_name) DO NOTHING;",
        (entity, group_id))

    conn.commit()
    cursor.close()
    conn.close()


# Create a learning unit and save it to the database
def create_learning_unit(main_knowledge, text, group_name, related_knowledges):
    conn = connect_db()
    cursor = conn.cursor()

    # Insert the main knowledge and text into the learning unit
    cursor.execute("""
        INSERT INTO learning_units (main_knowledge, text, group_name)
        VALUES (
            (SELECT id FROM knowledge_items WHERE item_name = %s),
            %s,
            %s
        ) RETURNING id;
    """, (main_knowledge, text, group_name))
    unit_id = cursor.fetchone()[0]

    # Link the related knowledge items to the unit
    for knowledge in related_knowledges:
        cursor.execute("""
            INSERT INTO unit_knowledges (unit_id, knowledge_id)
            VALUES (%s, (SELECT id FROM knowledge_items WHERE item_name = %s));
        """, (unit_id, knowledge))

    conn.commit()
    cursor.close()
    conn.close()


# Check if a description is relevant to an object within a sentence
def is_description_relevant(obj, description, sentence):
    obj_word = next((word for word in sentence.words if word.lemma == obj), None)
    desc_word = next((word for word in sentence.words if word.lemma == description), None)

    if not obj_word or not desc_word:
        return False

    # Check if the description is the parent or child of the object in the dependency tree
    if desc_word.head == obj_word.id or obj_word.head == desc_word.id:
        return True

    # Check if the description is next to the object
    if abs(obj_word.id - desc_word.id) == 1:
        return True

    # Additional check for sequence (description before the object)
    if desc_word.id < obj_word.id and abs(obj_word.id - desc_word.id) <= 2:
        return True

    return False


# Generate a learning plan based on categorized knowledge
def generate_learning_plan(categorized_knowledge, embedding_model, original_text, doc):
    learning_plan = []

    for category, items in categorized_knowledge.items():
        if category == "actions" and items:
            for action in items:
                related_objects = categorized_knowledge.get("objects", [])
                descriptions = categorized_knowledge.get("descriptions", [])
                for sentence in doc.sentences:
                    object_phrases = []
                    for obj in related_objects:
                        relevant_descriptions = [desc for desc in descriptions if is_description_relevant(obj, desc, sentence)]
                        description_text = " ".join(relevant_descriptions) + " " if relevant_descriptions else ""
                        object_phrases.append(f"{description_text}{obj}")

                    text = f"{' '.join(object_phrases)} {action}"

                    create_learning_unit(action, text, category, related_objects)
                    learning_plan.append({
                        "mainKnowledge": action,
                        "knowledges": related_objects,
                        "text": text,
                        "group": category
                    })

    for unit in learning_plan:
        query_embedding = embedding_model.encode(unit['text'])
        print(f"Enhanced learning block: {unit['text']}")

    return learning_plan


def main():
    # Load the Ukrainian language model for Stanza
    stanza.download('uk')
    nlp_pipeline = stanza.Pipeline('uk')

    # Initialize the embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Fetch knowledge from the database
    knowledge_items = fetch_knowledge_from_db()

    while True:
        # Get input from the user
        text = input("Input text (or 'exit'): ")

        # Check if the user wants to exit
        if text.lower() == "exit":
            print("end.")
            break

        # Analyze the text using the NLP pipeline
        doc = nlp_pipeline(text)
        categorized_knowledge = analyze_text(text, nlp_pipeline, knowledge_items)

        # Display the categorized knowledge
        print("Categorized knowledge:")
        for category, items in categorized_knowledge.items():
            print(f"{category.capitalize()}: {', '.join(items)}")

        # Generate a learning plan based on the categorized knowledge
        learning_plan = generate_learning_plan(categorized_knowledge, embedding_model, text, doc)
        print("Learning plan generated:")
        for unit in learning_plan:
            print(unit)


if __name__ == "__main__":
    main()
