"""
Streamlit Application for FoodLoversTV Restaurant Recommendations using GraphRAG and Gemini LLM.
Features conversational AI, interactive Vis.js subgraph explorer, Folium/Streamlit maps, and database filtering.
"""

import os
import html
import urllib.parse
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import folium
from folium.plugins import MarkerCluster
from graph_engine import FoodGraphEngine
from gemini_client import GeminiClient
from opik_tracer import (
    track,
    configure_opik,
    get_opik_status,
    get_trace_url,
    get_current_trace_id,
    update_current_trace,
    is_opik_active,
)

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


def get_restaurant_maps_link(restaurant: dict) -> str:
    """Retrieve or construct a valid Google Maps link for a restaurant."""
    if restaurant.get("maps_link"):
        return restaurant["maps_link"]
    lat, lon = restaurant.get("lat"), restaurant.get("lon")
    if lat is not None and lon is not None:
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    name = restaurant.get("name", "Restaurant")
    return f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(name + ' Bengaluru')}"


def render_folium_map(restaurants: list, height: int = 480, default_zoom: int = 12):
    """
    Renders an interactive Folium map with rich markers, custom pins,
    major cuisines badges, Google Maps links, dishes, and clustering.
    """
    # Safely filter restaurants with valid numeric coordinates
    valid_entries = []
    for r in restaurants:
        try:
            lat = float(r.get("lat"))
            lon = float(r.get("lon"))
            valid_entries.append((r, lat, lon))
        except (ValueError, TypeError):
            continue

    if not valid_entries:
        st.info("📍 No geographical coordinates available to display on the map.")
        return

    # Calculate geographic center
    avg_lat = sum(coords[1] for coords in valid_entries) / len(valid_entries)
    avg_lon = sum(coords[2] for coords in valid_entries) / len(valid_entries)

    zoom_level = 15 if len(valid_entries) == 1 else default_zoom
    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=zoom_level,
        tiles="OpenStreetMap"
    )

    # Auto-fit bounds when multiple restaurants are present
    if len(valid_entries) > 1:
        all_lats = [coords[1] for coords in valid_entries]
        all_lons = [coords[2] for coords in valid_entries]
        min_lat, max_lat = min(all_lats), max(all_lats)
        min_lon, max_lon = min(all_lons), max(all_lons)
        if min_lat != max_lat and min_lon != max_lon:
            m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(25, 25))

    # Cluster markers if large dataset, otherwise show individual pins directly
    target_layer = MarkerCluster().add_to(m) if len(valid_entries) > 20 else m

    for r, lat, lon in valid_entries:
        name = r.get("name") or "Restaurant"
        safe_name = html.escape(name)
        cuisines = r.get("cuisines") or []
        major_cuisines = cuisines[:4]
        major_cuisines_str = ", ".join(major_cuisines) if major_cuisines else "Indian"
        safe_cuisines = html.escape(major_cuisines_str)
        dietary = r.get("dietary", "Veg")
        is_veg = "non" not in str(dietary).lower()
        price = html.escape(str(r.get("price") or "$"))
        address = r.get("address") or ""
        maps_link = get_restaurant_maps_link(r)
        video_link = r.get("video_link") or ""
        dishes = r.get("dishes") or []

        # Major Cuisines Badges HTML
        cuisine_badges_html = "".join([
            f'<span style="background-color: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; border-radius: 12px; padding: 2px 7px; font-size: 11px; margin-right: 4px; display: inline-block; margin-bottom: 3px; font-weight: 500;">{html.escape(c)}</span>'
            for c in major_cuisines
        ])

        # Dietary Badge
        diet_badge_html = (
            '<span style="background-color: #dcfce7; color: #15803d; border: 1px solid #86efac; border-radius: 4px; padding: 2px 6px; font-size: 10px; font-weight: 700;">🌱 PURE VEG</span>'
            if is_veg else
            '<span style="background-color: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; border-radius: 4px; padding: 2px 6px; font-size: 10px; font-weight: 700;">🍗 NON-VEG</span>'
        )

        # Signature Dishes List
        dishes_html = ""
        if dishes:
            top_dishes = dishes[:3]
            dish_items = "".join([
                f'<li style="margin-bottom: 2px;">{html.escape(d.get("name", ""))}' +
                (f' <span style="color: #6b7280; font-size: 11px;">({html.escape(str(d.get("price")))})</span>' if d.get("price") else "") +
                '</li>'
                for d in top_dishes if d.get("name")
            ])
            if dish_items:
                dishes_html = f'''
                <div style="margin-top: 6px; font-size: 12px;">
                    <strong style="color: #374151;">🍲 Signature Dishes:</strong>
                    <ul style="margin: 2px 0 4px 16px; padding: 0; color: #4b5563;">
                        {dish_items}
                    </ul>
                </div>
                '''

        # Address snippet
        address_html = ""
        if address:
            short_addr = address[:90] + ("..." if len(address) > 90 else "")
            address_html = f'<div style="font-size: 11px; color: #6b7280; margin-top: 4px;">📍 {html.escape(short_addr)}</div>'

        # Action Buttons
        yt_btn_html = (
            f'<a href="{html.escape(video_link)}" target="_blank" rel="noopener noreferrer" style="background-color: #dc2626; color: white !important; padding: 5px 10px; border-radius: 4px; text-decoration: none; font-size: 11px; font-weight: 600; display: inline-flex; align-items: center; gap: 4px;">🎬 Watch Video ↗</a>'
            if video_link else ""
        )

        maps_btn_html = (
            f'<a href="{html.escape(maps_link)}" target="_blank" rel="noopener noreferrer" style="background-color: #1a73e8; color: white !important; padding: 5px 10px; border-radius: 4px; text-decoration: none; font-size: 11px; font-weight: 600; display: inline-flex; align-items: center; gap: 4px;">🗺️ Open in Google Maps ↗</a>'
        )

        # Popup Card Layout
        popup_html = f'''
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13px; line-height: 1.4; color: #1f2937; min-width: 230px; max-width: 290px;">
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 4px;">
                <h4 style="margin: 0; font-size: 15px; font-weight: 700; color: #111827;">{safe_name}</h4>
                {diet_badge_html}
            </div>
            <div style="font-size: 11px; color: #6b7280; margin-bottom: 6px;">Price Tier: <b>{price}</b></div>
            <div style="margin-bottom: 4px;">
                <div style="font-size: 10px; text-transform: uppercase; font-weight: 700; color: #9ca3af; margin-bottom: 2px;">Major Cuisines</div>
                <div>{cuisine_badges_html}</div>
            </div>
            {dishes_html}
            {address_html}
            <div style="margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap;">
                {maps_btn_html}
                {yt_btn_html}
            </div>
        </div>
        '''

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"📍 {safe_name} — {safe_cuisines} ({dietary})",
            icon=folium.Icon(color="green" if is_veg else "red", icon="cutlery", prefix="fa")
        ).add_to(target_layer)

    components.html(m._repr_html_(), height=height, scrolling=False)
    st.caption("🟢 Pure Veg Marker | 🔴 Non-veg Marker • Click any marker to view restaurant name, major cuisines, signature dishes, and Google Maps link.")


# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = None


# ----------------------------------------------------------------------
# Sidebar Controls & Configuration
# ----------------------------------------------------------------------
# with st.sidebar:
#     st.image("https://raw.githubusercontent.com/harikrishnad1997/YouTube-Scarpper-Map/main/Website/assets/img/logo.png", width=120) if os.path.exists("Website/assets/img/logo.png") else None
#     st.title("🥘 FoodLoversTV KG")
#     st.caption("AI Concierge powered by GraphRAG & Gemini")

#     st.divider()

#     # Knowledge Graph Metrics in Sidebar
#     st.subheader("📊 Graph Metrics")
#     col_sb1, col_sb2 = st.columns(2)
#     with col_sb1:
#         st.metric("Restaurants", len(graph_engine.restaurants_data))
#         st.metric("Dishes", len(graph_engine.dish_nodes))
#     with col_sb2:
#         st.metric("Cuisines", len(graph_engine.cuisine_nodes))
#         st.metric("Graph Edges", graph_engine.graph.number_of_edges())

#     st.divider()
#     st.subheader("🔭 Opik Observability")
#     opik_info = get_opik_status()
#     if opik_info["active"]:
#         st.success(f"● Tracing Active (`{opik_info['project_name']}`)")
#     elif opik_info["installed"]:
#         st.info("● Opik Ready (Configure API Key or Local)")
#     else:
#         st.caption("● Opik Pass-through (`pip install opik`)")

#     # with st.expander("⚙️ Opik Settings", expanded=False):
#     #     opik_enabled = st.checkbox("Enable Tracing", value=opik_info["enabled"])
#     #     opik_project = st.text_input("Project Name", value=opik_info["project_name"])
#     #     opik_api_key_in = st.text_input(
#     #         "Opik API Key",
#     #         type="password",
#     #         value=os.environ.get("OPIK_API_KEY", ""),
#     #         placeholder="Paste Comet / Opik API key..."
#     #     )
#     #     opik_local = st.checkbox("Use Local Opik (localhost:5173)", value=opik_info["use_local"])
#     #     if st.button("Save Opik Settings", use_container_width=True):
#     #         configure_opik(
#     #             api_key=opik_api_key_in if opik_api_key_in else None,
#     #             project_name=opik_project,
#     #             use_local=opik_local,
#     #             enabled=opik_enabled,
#     #         )
#     #         st.rerun()

#     st.divider()
#     if st.button("🗑️ Clear Chat History", use_container_width=True):
#         st.session_state.messages = []
#         st.session_state.last_retrieval = None
#         st.rerun()

# Default GraphRAG & Gemini Configuration
top_k = 5
hop_depth = 2

# Initialize Gemini Client (automatically loads key from secrets/env/gemini_api.txt)
gemini_client = GeminiClient(model_name="gemini-3.8-flash")


@track(name="foodloverstv_concierge_query", tags=["chat", "graphrag", "concierge"])
def process_concierge_query(query: str, history: list, top_k_val: int = 5, hop_val: int = 2):
    """Execute end-to-end GraphRAG retrieval + Gemini synthesis with Opik tracing."""
    update_current_trace(
        name=f"Query: {query[:45]}...",
        tags=["foodloverstv", "graphrag"],
        metadata={"query": query, "top_k": top_k_val, "hop_depth": hop_val}
    )
    # 1. GraphRAG Hybrid Retrieval
    retrieval = graph_engine.retrieve_subgraph(
        query=query,
        top_k=top_k_val,
        hop_depth=hop_val
    )
    # 2. LLM Synthesis
    ans = gemini_client.answer_query_with_graphrag(
        query=query,
        llm_context=retrieval["llm_context"],
        chat_history=history
    )
    trace_id = get_current_trace_id()
    trace_url = get_trace_url(trace_id)
    return retrieval, ans, trace_id, trace_url


# ----------------------------------------------------------------------
# Header & Navigation Tabs
# ----------------------------------------------------------------------
st.markdown('<div class="main-title">🥘 FoodLoversTV Concierge</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Discover Bengaluru & Karnataka\'s best restaurants and iconic dishes, '
    'grounded in verified reviews by Kripal Amanna (FoodLoversTV).</div>',
    unsafe_allow_html=True
)

tab_chat, tab_explorer, tab_analytics = st.tabs([
    "💬 AI Food Concierge",
    "🗺️ Restaurant Explorer & Map",
    "🕸️ Knowledge Graph Analytics"
])


# ----------------------------------------------------------------------
# TAB 1: Chat Interface with GraphRAG & Subgraph Visualizer
# ----------------------------------------------------------------------
with tab_chat:
    # Quick suggestion prompt pills
    # st.write("💡 **Quick Inquiries:**")
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
                    if message.get("trace_url"):
                        st.markdown(f"🔭 **Opik Trace:** [Open in Opik Dashboard]({message['trace_url']}) (`{message.get('trace_id', '')}`)")
                    elif message.get("trace_id"):
                        st.caption(f"🔭 **Opik Trace ID:** `{message['trace_id']}`")
                    subgraph_data = message["subgraph"]
                    st.caption(
                        f"Retrieved **{len(subgraph_data.get('nodes', []))} nodes** and "
                        f"**{len(subgraph_data.get('edges', []))} edges** across "
                        f"**{len(message.get('top_restaurants', []))} candidate restaurants**."
                    )
                    
                    # Render interactive Vis.js HTML graph
                    vis_html = graph_engine.generate_vis_html(subgraph_data, height="460px")
                    components.html(vis_html, height=480, scrolling=False)

                    # Show recommended restaurant map with rich markers & details
                    top_rests = message.get("top_restaurants", [])
                    if any(r.get("lat") and r.get("lon") for r in top_rests):
                        st.markdown("**📍 Location Map:**")
                        render_folium_map(top_rests, height=420)

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
                    # 1. Execute with Opik Tracing (Retrieval + Synthesis)
                    retrieval_res, answer, trace_id, trace_url = process_concierge_query(
                        query=user_query,
                        history=st.session_state.messages[:-1],
                        top_k_val=top_k,
                        hop_val=hop_depth,
                    )
                    st.session_state.last_retrieval = retrieval_res

                    st.markdown(answer)

                    # 2. Interactive Subgraph Reasoning & Map
                    with st.expander("🔍 View Knowledge Graph Retrieval & Subgraph Reasoning", expanded=True):
                        if trace_url:
                            st.markdown(f"🔭 **Opik Trace:** [Open Trace in Opik Dashboard]({trace_url}) (`{trace_id}`)")
                        elif trace_id:
                            st.caption(f"🔭 **Opik Trace ID:** `{trace_id}`")

                        subgraph_data = retrieval_res["subgraph"]
                        st.caption(
                            f"Retrieved **{len(subgraph_data.get('nodes', []))} nodes** and "
                            f"**{len(subgraph_data.get('edges', []))} edges** across "
                            f"**{len(retrieval_res['top_restaurants'])} candidate restaurants**."
                        )
                        
                        # Render Vis.js interactive network graph
                        vis_html = graph_engine.generate_vis_html(subgraph_data, height="460px")
                        components.html(vis_html, height=480, scrolling=False)

                        # Render interactive map with rich markers & details
                        top_rests = retrieval_res.get("top_restaurants", [])
                        if any(r.get("lat") and r.get("lon") for r in top_rests):
                            st.markdown("**📍 Recommended Locations Map:**")
                            render_folium_map(top_rests, height=420)

                    # Store in chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "subgraph": retrieval_res["subgraph"],
                        "top_restaurants": retrieval_res["top_restaurants"],
                        "trace_id": trace_id,
                        "trace_url": trace_url,
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

    # Interactive Map of filtered restaurants
    if filtered_restaurants:
        render_folium_map(filtered_restaurants, height=520)

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
                maps_link = get_restaurant_maps_link(r)
                if maps_link:
                    st.markdown(f"📍 [**Open in Google Maps**]({maps_link})")
                if r.get("video_link"):
                    st.markdown(f"🎬 [**Watch on YouTube**]({r['video_link']})")
                # Try to extract video ID for thumbnail
                if r.get("video_link") and "watch?v=" in r["video_link"]:
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
        Restaurant -.->|"SIMILAR_TO (Co-occurrence)"| Restaurant
    ```
    """)
