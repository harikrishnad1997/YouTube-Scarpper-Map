"""
Knowledge Graph Construction and GraphRAG Engine for FoodLoversTV Restaurant Database.
Builds a heterogeneous knowledge graph over restaurants, cuisines, dishes, locations, and dietary types.
Provides multi-hop subgraph extraction, hybrid retrieval, and interactive Vis.js graph rendering.
"""

import os
import re
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from opik_tracer import track, update_current_span

logger = logging.getLogger("graph_engine")

# Canonical location aliases to normalize messy addresses
LOCATION_MAPPINGS = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "mysore": "Mysuru",
    "mysuru": "Mysuru",
    "jayanagar": "Jayanagar",
    "shivajinagar": "Shivajinagar",
    "shivaji nagar": "Shivajinagar",
    "indiranagar": "Indiranagar",
    "indira nagar": "Indiranagar",
    "koramangala": "Koramangala",
    "malleshwaram": "Malleshwaram",
    "malleswaram": "Malleshwaram",
    "basavanagudi": "Basavanagudi",
    "frazer town": "Frazer Town",
    "whitefield": "Whitefield",
    "rajajinagar": "Rajajinagar",
    "hsr layout": "HSR Layout",
    "hsr": "HSR Layout",
    "commercial street": "Commercial Street",
    "chickpet": "Chickpet",
    "brigade road": "Brigade Road",
    "church street": "Church Street",
    "mg road": "MG Road",
    "vijayanagar": "Vijayanagar",
    "vv puram": "VV Puram",
    "v v puram": "VV Puram",
    "ulsoor": "Ulsoor",
    "halasuru": "Halasuru",
    "btm layout": "BTM Layout",
    "jp nagar": "JP Nagar",
    "j.p. nagar": "JP Nagar",
    "electronic city": "Electronic City",
    "sadashivanagar": "Sadashivanagar",
    "marathahalli": "Marathahalli",
    "yelahanka": "Yelahanka",
    "kammanahalli": "Kammanahalli",
    "seshadripuram": "Seshadripuram",
    "hassan": "Hassan",
    "mangalore": "Mangalore",
    "mangaluru": "Mangalore",
    "udupi": "Udupi",
    "hubli": "Hubli",
    "dharwad": "Dharwad",
    "coorg": "Coorg",
    "madikeri": "Madikeri",
}


def clean_text(text: Any) -> str:
    """Safely convert any value to clean stripped string."""
    if text is None:
        return ""
    if isinstance(text, list):
        return " ".join([clean_text(x) for x in text if x])
    return str(text).strip()


def extract_locations_from_address(address_raw: Any) -> List[str]:
    """Extract recognized neighborhoods and cities from address string/list."""
    if not address_raw:
        return ["Bengaluru"]
    if isinstance(address_raw, list):
        addr_str = ", ".join([str(x) for x in address_raw if x])
    else:
        addr_str = str(address_raw)

    found_locations: Set[str] = set()
    addr_lower = addr_str.lower()

    # Match predefined neighborhood and city keywords
    for key, canon in LOCATION_MAPPINGS.items():
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, addr_lower):
            found_locations.add(canon)

    # If no specific neighborhood found, try to parse by commas
    if not found_locations:
        tokens = [t.strip() for t in addr_str.split(",") if t.strip()]
        for token in tokens[-3:]:
            cleaned = re.sub(r"\b\d{6}\b", "", token).strip()
            if cleaned and len(cleaned) > 3 and not cleaned.startswith("+"):
                found_locations.add(cleaned.title())

    return list(found_locations) if found_locations else ["Bengaluru"]


class FoodGraphEngine:
    """
    Constructs and queries an in-memory Food Knowledge Graph using NetworkX.
    Supports hybrid graph traversal + TF-IDF text search for GraphRAG.
    """

    def __init__(self, data_path: str = "final_data.json"):
        self.data_path = data_path
        self.graph = nx.Graph()
        self.restaurants_data: Dict[str, Dict[str, Any]] = {}
        self.restaurant_nodes: List[str] = []
        self.cuisine_nodes: Set[str] = set()
        self.dish_nodes: Set[str] = set()
        self.location_nodes: Set[str] = set()
        self.dietary_nodes: Set[str] = set()

        # Search index
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.corpus_ids: List[str] = []
        self.corpus_docs: List[str] = []

        self._load_and_build_graph()

    def _normalize_dishes(self, best_dishes_raw: Any) -> List[Dict[str, Any]]:
        """Normalize various best_dishes data formats (list of dicts, dict of dicts, strings)."""
        dishes = []
        if not best_dishes_raw:
            return dishes

        if isinstance(best_dishes_raw, list):
            for item in best_dishes_raw:
                if isinstance(item, dict):
                    name = clean_text(item.get("name") or item.get("dish") or item.get("item"))
                    summary = clean_text(item.get("summary") or item.get("desc") or item.get("description"))
                    price = item.get("price")
                    if name:
                        dishes.append({
                            "name": name,
                            "summary": summary,
                            "price": str(price).strip() if price else None
                        })
                elif isinstance(item, str) and item.strip():
                    dishes.append({"name": item.strip(), "summary": "", "price": None})

        elif isinstance(best_dishes_raw, dict):
            for key, val in best_dishes_raw.items():
                name = clean_text(key)
                if isinstance(val, dict):
                    summary = clean_text(val.get("summary") or val.get("description"))
                    price = val.get("price")
                    dishes.append({
                        "name": name,
                        "summary": summary,
                        "price": str(price).strip() if price else None
                    })
                elif isinstance(val, str):
                    dishes.append({"name": name, "summary": val.strip(), "price": None})
                else:
                    dishes.append({"name": name, "summary": "", "price": None})

        return dishes

    def _normalize_cuisines(self, cuisine_raw: Any) -> List[str]:
        """Normalize cuisines into a clean list of strings."""
        if not cuisine_raw:
            return ["Indian"]
        cuisines = []
        if isinstance(cuisine_raw, list):
            for c in cuisine_raw:
                if isinstance(c, str):
                    cuisines.extend([x.strip() for x in c.split(",") if x.strip()])
        elif isinstance(cuisine_raw, str):
            cuisines.extend([x.strip() for x in cuisine_raw.split(",") if x.strip()])
        return list(set(cuisines)) if cuisines else ["Indian"]

    def _normalize_coords(self, lat_raw: Any, long_raw: Any) -> Tuple[Optional[float], Optional[float]]:
        """Safely parse coordinates to floats."""
        try:
            lat = float(str(lat_raw).strip()) if lat_raw is not None else None
            lon = float(str(long_raw).strip()) if long_raw is not None else None
            # Validate reasonable coordinate bounds for India
            if lat and lon and 8.0 <= lat <= 37.0 and 68.0 <= lon <= 97.0:
                return lat, lon
        except (ValueError, TypeError):
            pass
        return None, None

    def _load_and_build_graph(self):
        """Construct knowledge graph nodes and edges from final_data.json."""
        logger.info(f"Loading data from {self.data_path}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Database file not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        for rest_name, entries in raw_data.items():
            if not rest_name or not entries:
                continue

            clean_rest_name = rest_name.strip()
            primary_entry = entries[0]
            
            # Extract attributes
            cuisines = self._normalize_cuisines(primary_entry.get("cuisine"))
            v_n_ng = primary_entry.get("v_n_ng", "Veg")
            dietary = "Veg" if "veg" in str(v_n_ng).lower() and "non" not in str(v_n_ng).lower() else "Non-veg"
            price = clean_text(primary_entry.get("price")) or "$"
            address = clean_text(primary_entry.get("address"))
            phone = clean_text(primary_entry.get("ph_no"))
            video_link = clean_text(primary_entry.get("link"))
            video_title = clean_text(primary_entry.get("title"))
            summary = clean_text(primary_entry.get("summary"))
            lat, lon = self._normalize_coords(primary_entry.get("lat"), primary_entry.get("long"))
            locations = extract_locations_from_address(address)

            # Extract all dishes across entries
            all_dishes = []
            seen_dish_names = set()
            for e in entries:
                dishes = self._normalize_dishes(e.get("best_dishes"))
                for d in dishes:
                    d_name = d["name"].lower()
                    if d_name not in seen_dish_names:
                        seen_dish_names.add(d_name)
                        all_dishes.append(d)

            restaurant_info = {
                "id": f"rest:{clean_rest_name}",
                "name": clean_rest_name,
                "type": "Restaurant",
                "cuisines": cuisines,
                "dietary": dietary,
                "price": price,
                "address": address,
                "phone": phone,
                "video_link": video_link,
                "video_title": video_title,
                "summary": summary,
                "lat": lat,
                "lon": lon,
                "locations": locations,
                "dishes": all_dishes,
            }
            self.restaurants_data[clean_rest_name] = restaurant_info

            # 1. Add Restaurant Node
            rest_node_id = f"Restaurant::{clean_rest_name}"
            self.graph.add_node(
                rest_node_id,
                label=clean_rest_name,
                node_type="Restaurant",
                **restaurant_info
            )
            self.restaurant_nodes.append(rest_node_id)

            # 2. Add Dietary Node & Edge
            diet_node_id = f"Dietary::{dietary}"
            self.graph.add_node(diet_node_id, label=dietary, node_type="Dietary")
            self.dietary_nodes.add(diet_node_id)
            self.graph.add_edge(rest_node_id, diet_node_id, relation="HAS_DIET")

            # 3. Add Cuisine Nodes & Edges
            for c in cuisines:
                c_clean = c.strip()
                if not c_clean:
                    continue
                c_node_id = f"Cuisine::{c_clean}"
                self.graph.add_node(c_node_id, label=c_clean, node_type="Cuisine")
                self.cuisine_nodes.add(c_node_id)
                self.graph.add_edge(rest_node_id, c_node_id, relation="SERVES_CUISINE")

            # 4. Add Location Nodes & Edges
            for loc in locations:
                loc_node_id = f"Location::{loc}"
                self.graph.add_node(loc_node_id, label=loc, node_type="Location")
                self.location_nodes.add(loc_node_id)
                self.graph.add_edge(rest_node_id, loc_node_id, relation="LOCATED_IN")

            # 5. Add Dish Nodes & Edges
            for dish in all_dishes:
                dish_name = dish["name"]
                dish_node_id = f"Dish::{dish_name.title()}"
                self.graph.add_node(
                    dish_node_id,
                    label=dish_name.title(),
                    node_type="Dish",
                    summary=dish.get("summary", ""),
                    price=dish.get("price"),
                )
                self.dish_nodes.add(dish_node_id)
                self.graph.add_edge(
                    rest_node_id,
                    dish_node_id,
                    relation="KNOWN_FOR_DISH",
                    dish_summary=dish.get("summary", ""),
                    price=dish.get("price")
                )

        # 6. Build Restaurant-to-Restaurant Similarity Edges
        self._build_similarity_edges()

        # 7. Build TF-IDF Semantic Search Index
        self._build_text_search_index()

        logger.info(
            f"Knowledge Graph ready: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges across {len(self.restaurants_data)} restaurants."
        )

    def _build_similarity_edges(self):
        """Connect restaurants that share multiple cuisines or signature dishes."""
        restaurant_list = list(self.restaurants_data.values())
        n = len(restaurant_list)
        for i in range(n):
            r1 = restaurant_list[i]
            c1 = set(r1["cuisines"])
            d1 = {d["name"].lower() for d in r1["dishes"]}
            l1 = set(r1["locations"])
            n1 = f"Restaurant::{r1['name']}"

            for j in range(i + 1, min(i + 35, n)):
                r2 = restaurant_list[j]
                n2 = f"Restaurant::{r2['name']}"
                common_c = c1.intersection(set(r2["cuisines"]))
                common_d = d1.intersection({d["name"].lower() for d in r2["dishes"]})
                common_l = l1.intersection(set(r2["locations"]))

                sim_weight = len(common_c) * 1.5 + len(common_d) * 2.0 + (1.0 if common_l else 0.0)
                if sim_weight >= 3.0:
                    self.graph.add_edge(
                        n1, n2,
                        relation="SIMILAR_TO",
                        weight=sim_weight,
                        common_cuisines=list(common_c),
                        common_dishes=list(common_d),
                    )

    def _build_text_search_index(self):
        """Index restaurant textual summaries and dishes with TF-IDF for semantic hybrid matching."""
        self.corpus_ids = []
        self.corpus_docs = []

        for name, r in self.restaurants_data.items():
            doc_parts = [
                r["name"],
                " ".join(r["cuisines"]),
                r["dietary"],
                " ".join(r["locations"]),
                r["address"],
                r["summary"],
            ]
            for d in r["dishes"]:
                doc_parts.append(d["name"])
                if d.get("summary"):
                    doc_parts.append(d["summary"])

            doc_text = " ".join(doc_parts).lower()
            self.corpus_ids.append(f"Restaurant::{name}")
            self.corpus_docs.append(doc_text)

        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus_docs)

    @track(name="graphrag_search_text_scores", tags=["graphrag", "tfidf"])
    def search_text_scores(self, query: str, top_k: int = 20) -> Dict[str, float]:
        """Compute TF-IDF cosine similarity scores for query across restaurants."""
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return {}
        query_vec = self.tfidf_vectorizer.transform([query.lower()])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        results = {}
        ranked_indices = scores.argsort()[::-1][:top_k]
        for idx in ranked_indices:
            score = float(scores[idx])
            if score > 0.01:
                results[self.corpus_ids[idx]] = score
        return results

    @track(name="graphrag_extract_entities", tags=["graphrag", "ner"])
    def extract_query_entities(self, query: str) -> Dict[str, List[str]]:
        """Identify matching nodes directly mentioned in the user query."""
        query_lower = query.lower()
        matched = {
            "cuisines": [],
            "dishes": [],
            "locations": [],
            "dietary": [],
            "restaurants": [],
        }

        # Match dietary
        if "non-veg" in query_lower or "non veg" in query_lower or "meat" in query_lower or "chicken" in query_lower or "mutton" in query_lower or "fish" in query_lower or "seafood" in query_lower:
            matched["dietary"].append("Dietary::Non-veg")
        elif "pure veg" in query_lower or "vegetarian" in query_lower or "veg" in query_lower:
            matched["dietary"].append("Dietary::Veg")

        # Match locations
        for loc_id in self.location_nodes:
            loc_label = self.graph.nodes[loc_id]["label"].lower()
            if loc_label in query_lower or any(re.search(r"\b" + re.escape(k) + r"\b", query_lower) for k, v in LOCATION_MAPPINGS.items() if v.lower() == loc_label):
                matched["locations"].append(loc_id)

        # Match cuisines
        for c_id in self.cuisine_nodes:
            c_label = self.graph.nodes[c_id]["label"].lower()
            if len(c_label) >= 3 and re.search(r"\b" + re.escape(c_label) + r"\b", query_lower):
                matched["cuisines"].append(c_id)

        # Match dishes
        for d_id in self.dish_nodes:
            d_label = self.graph.nodes[d_id]["label"].lower()
            if len(d_label) >= 4 and d_label in query_lower:
                matched["dishes"].append(d_id)

        # Match restaurants directly
        for r_id in self.restaurant_nodes:
            r_label = self.graph.nodes[r_id]["label"].lower()
            if len(r_label) >= 4 and r_label in query_lower:
                matched["restaurants"].append(r_id)

        return matched

    @track(name="graphrag_retrieve_subgraph", tags=["graphrag", "retrieval"])
    def retrieve_subgraph(
        self,
        query: str,
        top_k: int = 5,
        hop_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Execute GraphRAG hybrid retrieval:
        1. Extract query entities.
        2. Perform multi-hop graph traversal to discover connected candidate restaurants.
        3. Score candidates with a combination of graph connectivity + TF-IDF semantic relevance.
        4. Extract the induced subgraph containing top restaurants and related entities.
        5. Format structured textual context for the LLM.
        """
        query_entities = self.extract_query_entities(query)
        text_scores = self.search_text_scores(query, top_k=25)

        candidate_scores: Dict[str, float] = {}

        # 1. Score restaurants from direct entity matches (1-hop & 2-hop)
        for r_id in query_entities["restaurants"]:
            candidate_scores[r_id] = candidate_scores.get(r_id, 0.0) + 10.0

        for d_id in query_entities["dishes"]:
            for neighbor in self.graph.neighbors(d_id):
                if self.graph.nodes[neighbor].get("node_type") == "Restaurant":
                    candidate_scores[neighbor] = candidate_scores.get(neighbor, 0.0) + 6.0

        for c_id in query_entities["cuisines"]:
            for neighbor in self.graph.neighbors(c_id):
                if self.graph.nodes[neighbor].get("node_type") == "Restaurant":
                    candidate_scores[neighbor] = candidate_scores.get(neighbor, 0.0) + 4.0

        for loc_id in query_entities["locations"]:
            for neighbor in self.graph.neighbors(loc_id):
                if self.graph.nodes[neighbor].get("node_type") == "Restaurant":
                    candidate_scores[neighbor] = candidate_scores.get(neighbor, 0.0) + 3.5

        # 2. Add TF-IDF semantic scores
        for r_id, t_score in text_scores.items():
            candidate_scores[r_id] = candidate_scores.get(r_id, 0.0) + (t_score * 12.0)

        # 3. Dietary filtering / boosting
        if query_entities["dietary"]:
            target_diet = query_entities["dietary"][0].split("::")[-1]
            for r_id in list(candidate_scores.keys()):
                r_diet = self.graph.nodes[r_id].get("dietary")
                if r_diet == target_diet:
                    candidate_scores[r_id] += 2.0
                elif target_diet == "Veg" and r_diet == "Non-veg":
                    candidate_scores[r_id] -= 5.0

        # Fallback if no candidate scored
        if not candidate_scores:
            for r_id, t_score in text_scores.items():
                candidate_scores[r_id] = t_score * 10.0

        if not candidate_scores:
            degrees = dict(self.graph.degree())
            sorted_by_degree = sorted(
                [(n, d) for n, d in degrees.items() if n.startswith("Restaurant::")],
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            for r_id, d in sorted_by_degree:
                candidate_scores[r_id] = float(d)

        # Rank candidates
        ranked_restaurants = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_restaurant_nodes = [r_id for r_id, _ in ranked_restaurants]

        # Build induced Subgraph
        subgraph_nodes: Set[str] = set(top_restaurant_nodes)
        
        for r_id in top_restaurant_nodes:
            for neighbor in self.graph.neighbors(r_id):
                node_type = self.graph.nodes[neighbor].get("node_type")
                if node_type in ("Cuisine", "Location", "Dietary"):
                    subgraph_nodes.add(neighbor)
                elif node_type == "Dish":
                    d_label = self.graph.nodes[neighbor].get("label", "").lower()
                    if d_label in query.lower() or len([n for n in subgraph_nodes if n.startswith("Dish::")]) < 25:
                        subgraph_nodes.add(neighbor)
                elif node_type == "Restaurant" and hop_depth >= 2:
                    edge_data = self.graph.get_edge_data(r_id, neighbor, {})
                    if edge_data.get("relation") == "SIMILAR_TO" and neighbor in [r for r, _ in ranked_restaurants[:top_k]]:
                        subgraph_nodes.add(neighbor)

        subgraph = self.graph.subgraph(subgraph_nodes)
        subgraph_dict = self._serialize_subgraph(subgraph, top_restaurant_nodes)
        llm_context = self._build_llm_context(top_restaurant_nodes, subgraph)

        update_current_span(
            metadata={
                "top_k": top_k,
                "hop_depth": hop_depth,
                "subgraph_nodes_count": len(subgraph_dict.get("nodes", [])),
                "subgraph_edges_count": len(subgraph_dict.get("edges", [])),
                "candidate_count": len(ranked_restaurants),
                "matched_cuisines": query_entities.get("cuisines", []),
                "matched_dishes": query_entities.get("dishes", []),
                "matched_locations": query_entities.get("locations", []),
                "matched_dietary": query_entities.get("dietary", []),
            }
        )

        return {
            "query": query,
            "top_restaurants": [self.graph.nodes[r_id] for r_id in top_restaurant_nodes],
            "subgraph": subgraph_dict,
            "llm_context": llm_context,
            "matched_entities": query_entities,
            "candidate_scores": {self.graph.nodes[r_id]["label"]: round(score, 2) for r_id, score in ranked_restaurants}
        }

    def _serialize_subgraph(self, subgraph: nx.Graph, focal_nodes: List[str]) -> Dict[str, Any]:
        """Convert NetworkX subgraph to JSON format for Vis.js interactive network."""
        nodes = []
        edges = []

        type_colors = {
            "Restaurant": "#FF4B4B",
            "Cuisine": "#1E88E5",
            "Dish": "#43A047",
            "Location": "#8E24AA",
            "Dietary": "#FB8C00",
        }

        type_shapes = {
            "Restaurant": "box",
            "Cuisine": "ellipse",
            "Dish": "dot",
            "Location": "diamond",
            "Dietary": "star",
        }

        for node_id in subgraph.nodes():
            attrs = subgraph.nodes[node_id]
            node_type = attrs.get("node_type", "Entity")
            label = attrs.get("label", node_id.split("::")[-1])
            is_focal = node_id in focal_nodes

            tooltip_parts = [f"<b>{label}</b> ({node_type})"]
            if node_type == "Restaurant":
                tooltip_parts.append(f"<b>Cuisines:</b> {', '.join(attrs.get('cuisines', []))}")
                tooltip_parts.append(f"<b>Dietary:</b> {attrs.get('dietary')}")
                tooltip_parts.append(f"<b>Price:</b> {attrs.get('price')}")
                if attrs.get("address"):
                    tooltip_parts.append(f"<b>Address:</b> {attrs.get('address')}")
                if attrs.get("video_link"):
                    tooltip_parts.append(f"<a href='{attrs.get('video_link')}' target='_blank'>Watch YouTube Review</a>")
            elif node_type == "Dish":
                if attrs.get("price"):
                    tooltip_parts.append(f"<b>Price:</b> {attrs.get('price')}")
                if attrs.get("summary"):
                    tooltip_parts.append(f"<b>Details:</b> {attrs.get('summary')}")

            nodes.append({
                "id": node_id,
                "label": label,
                "group": node_type,
                "color": "#D32F2F" if (is_focal and node_type == "Restaurant") else type_colors.get(node_type, "#757575"),
                "shape": type_shapes.get(node_type, "dot"),
                "size": 26 if (is_focal and node_type == "Restaurant") else (20 if node_type in ("Restaurant", "Location") else 14),
                "title": "<br>".join(tooltip_parts),
                "font": {"color": "#FFFFFF" if node_type in ("Restaurant", "Location", "Cuisine") else "#212121", "size": 13, "face": "sans-serif"},
            })

        for u, v, data in subgraph.edges(data=True):
            relation = data.get("relation", "CONNECTED_TO")
            edge_color = "#BDBDBD"
            dashes = False
            if relation == "SIMILAR_TO":
                edge_color = "#FFA000"
                dashes = True

            edges.append({
                "from": u,
                "to": v,
                "label": relation,
                "color": {"color": edge_color, "highlight": "#FF4B4B"},
                "dashes": dashes,
                "font": {"size": 9, "align": "middle", "color": "#757575"}
            })

        return {"nodes": nodes, "edges": edges}

    def _build_llm_context(self, top_restaurant_nodes: List[str], subgraph: nx.Graph) -> str:
        """Synthesize facts, dishes, prices, and video URLs into a clean prompt context."""
        context_lines = [
            "### RETRIEVED KNOWLEDGE GRAPH CONTEXT (FoodLoversTV Verified Reviews):\n"
        ]

        for idx, r_id in enumerate(top_restaurant_nodes, 1):
            r_data = self.graph.nodes[r_id]
            name = r_data.get("name", "Unknown")
            cuisines = ", ".join(r_data.get("cuisines", []))
            dietary = r_data.get("dietary", "")
            price = r_data.get("price", "$$")
            address = r_data.get("address", "Bangalore")
            locations = ", ".join(r_data.get("locations", []))
            phone = r_data.get("phone", "N/A")
            video_link = r_data.get("video_link", "")
            summary = r_data.get("summary", "")

            dishes_info = []
            for neighbor in subgraph.neighbors(r_id):
                if neighbor.startswith("Dish::"):
                    d_data = subgraph.nodes[neighbor]
                    d_name = d_data.get("label", "")
                    d_price = d_data.get("price")
                    d_sum = d_data.get("summary")
                    dish_str = f"• {d_name}"
                    if d_price:
                        dish_str += f" ({d_price})"
                    if d_sum:
                        dish_str += f": {d_sum}"
                    dishes_info.append(dish_str)

            context_lines.append(f"--- Restaurant {idx}: {name} ---")
            context_lines.append(f"- Cuisines: {cuisines}")
            context_lines.append(f"- Dietary: {dietary} | Price Tier: {price}")
            context_lines.append(f"- Area/City: {locations}")
            context_lines.append(f"- Full Address: {address}")
            if phone and phone != "N/A":
                context_lines.append(f"- Phone: {phone}")
            if video_link:
                context_lines.append(f"- YouTube Review: {video_link}")
            if dishes_info:
                context_lines.append("- Highlight Dishes / Specialties:")
                context_lines.extend(dishes_info[:5])
            if summary:
                context_lines.append(f"- FoodLoversTV Host Review & Notes: {summary}")
            context_lines.append("")

        return "\n".join(context_lines)

    def generate_vis_html(self, subgraph_dict: Dict[str, Any], height: str = "520px") -> str:
        """Generate standalone HTML/JS rendering the Vis.js interactive network graph."""
        nodes_json = json.dumps(subgraph_dict.get("nodes", []))
        edges_json = json.dumps(subgraph_dict.get("edges", []))

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body, html {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0E1117;
            color: #FAFAFA;
        }}
        #mynetwork {{
            width: 100%;
            height: {height};
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            background: #161B22;
        }}
        .legend {{
            position: absolute;
            top: 10px;
            right: 12px;
            background: rgba(22, 27, 34, 0.88);
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            font-size: 11px;
            z-index: 999;
            backdrop-filter: blur(4px);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 4px;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
            margin-right: 6px;
        }}
    </style>
</head>
<body>
    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background:#FF4B4B;"></div><span>Restaurant</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#1E88E5;"></div><span>Cuisine</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#43A047;"></div><span>Dish / Specialty</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#8E24AA;"></div><span>Location / Area</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#FB8C00;"></div><span>Dietary</span></div>
    </div>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        const nodes = new vis.DataSet({nodes_json});
        const edges = new vis.DataSet({edges_json});

        const container = document.getElementById('mynetwork');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            nodes: {{
                shape: 'dot',
                shadow: true,
                borderWidth: 1.5,
                borderWidthSelected: 3,
                font: {{ color: '#FFFFFF', size: 12 }}
            }},
            edges: {{
                width: 1.2,
                selectionWidth: 2.5,
                smooth: {{ type: 'continuous' }},
                arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }}
            }},
            physics: {{
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 95,
                    springConstant: 0.08
                }},
                maxVelocity: 35,
                stabilization: {{ iterations: 120 }}
            }},
            interaction: {{
                hover: true,
                navigationButtons: true,
                keyboard: false,
                tooltipDelay: 150
            }}
        }};
        const network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""
        return html
