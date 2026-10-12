"""
Streamlit Application for FoodLoversTV Restaurant Recommendations using GraphRAG and Gemini LLM.
Features conversational AI, interactive Vis.js subgraph explorer, Folium/Streamlit maps, and database filtering.
"""

import os
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from graph_engine import FoodGraphEngine
from gemini_client import GeminiClient

# Page configuration
st.set_page_config(
    page_title="FoodLoversTV GraphRAG Concierge",
    page_icon="🥘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #FF4B4B, #FF8A65);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.2rem;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px;
    }
    .dish-badge {
        background-color: rgba(67, 160, 71, 0.15);
        color: #81C784;
        border: 1px solid #43A047;
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.8rem;
        margin-right: 4px;
        display: inline-block;
        margin-bottom: 4px;
    }
    .cuisine-badge {
        background-color: rgba(30, 136, 229, 0.15);
        color: #64B5F6;
        border: 1px solid #1E88E5;
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.8rem;
        margin-right: 4px;
        display: inline-block;
    }
    .veg-badge {
        background-color: rgba(76, 175, 80, 0.2);
        color: #81C784;
        border: 1px solid #4CAF50;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .nonveg-badge {
        background-color: rgba(239, 83, 80, 0.2);
        color: #E57373;
        border: 1px solid #E53935;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Knowledge Graph Engine (cached in session / memory)
@st.cache_resource(show_spinner="Building Food Knowledge Graph & Vector Indices...")
def get_graph_engine():
    # Resolve path to final_data.json
    candidates = ["final_data.json", "../final_data.json"]
    for p in candidates:
        if os.path.exists(p):
            return FoodGraphEngine(p)
    return FoodGraphEngine("final_data.json")

graph_engine = get_graph_engine()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = None


# ----------------------------------------------------------------------
# Sidebar Controls & Configuration
# ----------------------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/harikrishnad1997/YouTube-Scarpper-Map/main/Website/assets/img/logo.png", width=120) if os.path.exists("Website/assets/img/logo.png") else None
    st.title("🥘 FoodLoversTV KG")
    st.caption("AI Concierge powered by GraphRAG & Gemini")

    st.divider()

    # Knowledge Graph Metrics in Sidebar
    st.subheader("📊 Graph Metrics")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Restaurants", len(graph_engine.restaurants_data))
        st.metric("Dishes", len(graph_engine.dish_nodes))
    with col_sb2:
        st.metric("Cuisines", len(graph_engine.cuisine_nodes))
        st.metric("Graph Edges", graph_engine.graph.number_of_edges())

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_retrieval = None
        st.rerun()

# Default GraphRAG & Gemini Configuration
top_k = 5
hop_depth = 2

# Initialize Gemini Client (automatically loads key from secrets/env/gemini_api.txt)
gemini_client = GeminiClient(model_name="gemini-3.8-flash")


# ----------------------------------------------------------------------
# Header & Navigation Tabs
# ----------------------------------------------------------------------
st.markdown('<div class="main-title">🥘 FoodLoversTV GraphRAG Concierge</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Discover Bengaluru & Karnataka\'s best restaurants and iconic dishes, '
    'grounded in verified reviews by Kripal Amanna (FoodLoversTV).</div>',
    unsafe_allow_html=True
)

tab_chat, tab_explorer, tab_analytics = st.tabs([
    "💬 AI Food Concierge (GraphRAG)",
    "🗺️ Restaurant Explorer & Map",
    "🕸️ Knowledge Graph Analytics"
])


# ----------------------------------------------------------------------
# TAB 1: Chat Interface with GraphRAG & Subgraph Visualizer
# ----------------------------------------------------------------------
with tab_chat:
    # Quick suggestion prompt pills
    st.write("💡 **Quick Inquiries:**")
    pill_cols = st.columns(4)
    quick_queries = [
        "Best Donne Biryani in Bangalore",
        "Crispy Dosa & South Indian Breakfast in Jayanagar",
        "Authentic Mangalore Seafood & Fish Thali",
        "Pure Veg street food and snacks"
    ]

    selected_pill = None
    for i, q in enumerate(quick_queries):
        if pill_cols[i].button(q, key=f"pill_{i}", use_container_width=True):
            selected_pill = q

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If assistant message has graph metadata, render interactive viewer
            if message.get("subgraph") and message.get("top_restaurants"):
                with st.expander("🔍 View Knowledge Graph Retrieval & Subgraph Reasoning", expanded=False):
                    subgraph_data = message["subgraph"]
                    st.caption(
                        f"Retrieved **{len(subgraph_data.get('nodes', []))} nodes** and "
                        f"**{len(subgraph_data.get('edges', []))} edges** across "
                        f"**{len(message.get('top_restaurants', []))} candidate restaurants**."
                    )
                    
                    # Render interactive Vis.js HTML graph
                    vis_html = graph_engine.generate_vis_html(subgraph_data, height="460px")
                    components.html(vis_html, height=480, scrolling=False)

                    # Show recommended restaurant map
                    map_points = []
                    for r in message.get("top_restaurants", []):
                        if r.get("lat") and r.get("lon"):
                            map_points.append({"name": r.get("name"), "lat": r["lat"], "lon": r["lon"]})
                    if map_points:
                        st.markdown("**📍 Location Map:**")
                        df_map = pd.DataFrame(map_points)
                        st.map(df_map, latitude="lat", longitude="lon", zoom=11)

    # Chat Input Handler
    user_query = st.chat_input("Ask for restaurant recommendations, dishes, areas...") or selected_pill

    if user_query:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant generation
        with st.chat_message("assistant"):
            with st.spinner("Traversing Knowledge Graph & querying Gemini LLM..."):
                try:
                    # 1. GraphRAG Hybrid Retrieval
                    retrieval_res = graph_engine.retrieve_subgraph(
                        query=user_query,
                        top_k=top_k,
                        hop_depth=hop_depth
                    )
                    st.session_state.last_retrieval = retrieval_res

                    # 2. LLM Synthesis
                    answer = gemini_client.answer_query_with_graphrag(
                        query=user_query,
                        llm_context=retrieval_res["llm_context"],
                        chat_history=st.session_state.messages[:-1]
                    )

                    st.markdown(answer)

                    # 3. Interactive Subgraph Reasoning & Map
                    with st.expander("🔍 View Knowledge Graph Retrieval & Subgraph Reasoning", expanded=True):
                        subgraph_data = retrieval_res["subgraph"]
                        st.caption(
                            f"Retrieved **{len(subgraph_data.get('nodes', []))} nodes** and "
                            f"**{len(subgraph_data.get('edges', []))} edges** across "
                            f"**{len(retrieval_res['top_restaurants'])} candidate restaurants**."
                        )
                        
                        # Render Vis.js interactive network graph
                        vis_html = graph_engine.generate_vis_html(subgraph_data, height="460px")
                        components.html(vis_html, height=480, scrolling=False)

                        # Render Map
                        map_points = []
                        for r in retrieval_res["top_restaurants"]:
                            if r.get("lat") and r.get("lon"):
                                map_points.append({"name": r.get("name"), "lat": r["lat"], "lon": r["lon"]})
                        if map_points:
                            st.markdown("**📍 Recommended Locations:**")
                            df_map = pd.DataFrame(map_points)
                            st.map(df_map, latitude="lat", longitude="lon", zoom=11)

                    # Store in chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "subgraph": retrieval_res["subgraph"],
                        "top_restaurants": retrieval_res["top_restaurants"]
                    })

                except Exception as e:
                    error_msg = f"⚠️ **Error generating recommendation:** {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ----------------------------------------------------------------------
# TAB 2: Restaurant Explorer & Map Filter
# ----------------------------------------------------------------------
with tab_explorer:
    st.subheader("🔍 Explore the FoodLoversTV Database")

    # Filter Bar
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 1, 1])

    # Available options
    all_cuisines = sorted(list({c.split("::")[-1] for c in graph_engine.cuisine_nodes}))
    all_locations = sorted(list({l.split("::")[-1] for l in graph_engine.location_nodes}))

    with f_col1:
        sel_cuisines = st.multiselect("Filter by Cuisine", options=all_cuisines, default=[])
    with f_col2:
        sel_location = st.selectbox("Filter by Neighborhood / City", options=["All Locations"] + all_locations)
    with f_col3:
        sel_diet = st.selectbox("Dietary", options=["All", "Veg", "Non-veg"])
    with f_col4:
        sel_price = st.selectbox("Price Tier", options=["All", "$", "$$", "$$$"])

    # Search filter text
    search_term = st.text_input("Search restaurant name or dish specialty...", "").strip().lower()

    # Apply filters
    filtered_restaurants = []
    for r in graph_engine.restaurants_data.values():
        # Cuisine filter
        if sel_cuisines and not any(c in r["cuisines"] for c in sel_cuisines):
            continue
        # Location filter
        if sel_location != "All Locations" and sel_location not in r["locations"]:
            continue
        # Dietary filter
        if sel_diet != "All" and r["dietary"] != sel_diet:
            continue
        # Price filter
        if sel_price != "All" and r["price"] != sel_price:
            continue
        # Text search
        if search_term:
            in_name = search_term in r["name"].lower()
            in_dishes = any(search_term in d["name"].lower() for d in r["dishes"])
            in_summary = search_term in r["summary"].lower()
            if not (in_name or in_dishes or in_summary):
                continue

        filtered_restaurants.append(r)

    st.write(f"Showing **{len(filtered_restaurants)}** matching restaurants:")

    # Map of filtered restaurants
    map_data = []
    for r in filtered_restaurants:
        if r["lat"] and r["lon"]:
            map_data.append({"name": r["name"], "lat": r["lat"], "lon": r["lon"]})

    if map_data:
        st.map(pd.DataFrame(map_data), latitude="lat", longitude="lon", zoom=11)

    # Restaurant Cards
    for r in filtered_restaurants[:40]:  # Paginate top 40 for responsiveness
        diet_badge = '<span class="veg-badge">🌱 VEG</span>' if r["dietary"] == "Veg" else '<span class="nonveg-badge">🍗 NON-VEG</span>'
        
        with st.expander(f"**{r['name']}** — {', '.join(r['cuisines'][:3])} ({r['price']})", expanded=False):
            c_left, c_right = st.columns([3, 1])
            with c_left:
                st.markdown(f"### {r['name']} {diet_badge}", unsafe_allow_html=True)
                st.markdown(f"📍 **Address:** {r['address'] or 'N/A'}")
                if r['phone']:
                    st.markdown(f"📞 **Phone:** `{r['phone']}`")
                if r['summary']:
                    st.info(f"**Host Review:** {r['summary']}")
                
                # Signature dishes
                if r["dishes"]:
                    st.markdown("**Signature Dishes:**")
                    dish_htmls = []
                    for d in r["dishes"][:6]:
                        p_str = f" ({d['price']})" if d.get('price') else ""
                        dish_htmls.append(f'<span class="dish-badge">{d["name"]}{p_str}</span>')
                    st.markdown(" ".join(dish_htmls), unsafe_allow_html=True)

            with c_right:
                if r.get("video_link"):
                    st.markdown(f"🎬 [**Watch on YouTube**]({r['video_link']})")
                    # Try to extract video ID for thumbnail
                    if "watch?v=" in r["video_link"]:
                        v_id = r["video_link"].split("watch?v=")[-1].split("&")[0]
                        st.image(f"https://img.youtube.com/vi/{v_id}/hqdefault.jpg", use_container_width=True)


# ----------------------------------------------------------------------
# TAB 3: Knowledge Graph Network Analytics
# ----------------------------------------------------------------------
with tab_analytics:
    st.subheader("🕸️ Food Knowledge Graph Topology & Insights")

    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    with col_g1:
        st.metric("Total Nodes", graph_engine.graph.number_of_nodes())
    with col_g2:
        st.metric("Total Relations (Edges)", graph_engine.graph.number_of_edges())
    with col_g3:
        st.metric("Dish Entities", len(graph_engine.dish_nodes))
    with col_g4:
        st.metric("Cuisine Categories", len(graph_engine.cuisine_nodes))

    st.divider()

    # Top Cuisines Distribution
    st.markdown("#### 🍛 Top Cuisines by Restaurant Count")
    cuisine_counts = {}
    for c_id in graph_engine.cuisine_nodes:
        deg = graph_engine.graph.degree(c_id)
        c_label = graph_engine.graph.nodes[c_id]["label"]
        cuisine_counts[c_label] = deg
    
    top_cuisines_df = pd.DataFrame(
        sorted(cuisine_counts.items(), key=lambda x: x[1], reverse=True)[:15],
        columns=["Cuisine", "Number of Restaurants"]
    )
    st.bar_chart(top_cuisines_df.set_index("Cuisine"))

    # Most Central Restaurants in the Knowledge Graph
    st.markdown("#### 🏆 Most Connected Landmark Restaurants")
    st.caption("Restaurants with the richest graph connectivity (highest degree across signature dishes, multiple cuisines, and network similarity).")
    
    degrees = dict(graph_engine.graph.degree())
    rest_degrees = [
        (graph_engine.graph.nodes[n]["name"], d, graph_engine.graph.nodes[n]["dietary"], ", ".join(graph_engine.graph.nodes[n]["cuisines"][:3]))
        for n, d in degrees.items()
        if n.startswith("Restaurant::")
    ]
    top_rest_df = pd.DataFrame(
        sorted(rest_degrees, key=lambda x: x[1], reverse=True)[:12],
        columns=["Restaurant", "Graph Degree", "Dietary", "Top Cuisines"]
    )
    st.dataframe(top_rest_df, use_container_width=True)

    # Graph Schema Diagram
    st.markdown("#### 📐 Knowledge Graph Schema & Triples")
    st.markdown("""
    ```mermaid
    graph TD
        Restaurant["🏢 Restaurant (Name, Lat, Lon, Price, Review)"] -->|SERVES_CUISINE| Cuisine["🍜 Cuisine (e.g. Biryani, Mangalore)"]
        Restaurant -->|KNOWN_FOR_DISH| Dish["🍲 Dish (Name, Price, Summary)"]
        Restaurant -->|LOCATED_IN| Location["📍 Location (Neighborhood / City)"]
        Restaurant -->|HAS_DIET| Dietary["🥗 Dietary (Veg, Non-veg)"]
        Restaurant -.->|SIMILAR_TO (Co-occurrence)| Restaurant
    ```
    """)
