# YouTube Scrapper for FoodLoversTV

## Description

This project aims to create an interactive map and bot to navigate among reviews by FoodLoversTV. The scraper extracts video details from FoodLoversTV's YouTube channel and organizes them in a way that enhances the user experience, allowing users to easily access and explore reviews based on various criteria.

## Features

- **GraphRAG AI Food Concierge**: Streamlit web application that uses Knowledge Graph retrieval + Google Gemini LLM to answer user queries with verified restaurant facts, dishes, prices, locations, and YouTube review links.
- **Interactive Subgraph Visualizer**: Inline Vis.js network visualizer displaying the retrieved subgraph entities (Restaurants, Cuisines, Dishes, Locations) and reasoning path for each query.
- **Interactive Restaurant Explorer & Map**: Filter 470+ restaurants by cuisine, neighborhood/city, veg/non-veg, and price tier, with interactive map pins and video links.
- **YouTube Scraper & Pipeline**: Extracts video details, titles, descriptions, coordinates, and Gemini-enriched restaurant metadata.
- **Knowledge Graph Network Analytics**: Topology statistics, degree centrality, and top cuisine distribution.

## Running the GraphRAG Streamlit App

1. Ensure your Gemini API key is in `gemini_api.txt` or set as an environment variable:
   ```bash
   export GEMINI_API_KEY="your-api-key"
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
   *(Alternatively, run `streamlit run app/app.py`)*

3. Explore:
   - Ask natural language questions like *"What is the best Donne Biryani in Bangalore?"* or *"Where can I get authentic Mangalorean seafood?"*
   - Inspect the extracted knowledge graph and interactive map for each recommendation.
   - Use the **Restaurant Explorer** tab to filter by cuisine and neighborhood.

## Live Demo
Check out the interactive map at: [FoodLoversTV YouTube Scrapper Map](https://harikrishnad1997.github.io/YouTube-Scarpper-Map/)

## Contributing

We welcome contributions! Please feel free to submit issues, fork the repository, and open pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

