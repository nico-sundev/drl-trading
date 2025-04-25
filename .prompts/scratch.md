# Task: Multi-Timeframe GNN with Vector DB Integration for DRL Trading Agent

## Overview

You need to develop a module that integrates a Graph Neural Network (GNN) for encoding multi-timeframe market structure, stores the generated embeddings in a Vector Database, and allows retrieval of similar market structures and their future context. The module will be used within a Deep Reinforcement Learning (DRL) agent's environment, where the agent will leverage both current market embeddings and historical patterns retrieved from the database.

## Objectives

- **Graph Construction**: Build graphs from OHLCV data across multiple timeframes.
- **GNN Encoding**: Use a pretrained GNN to produce embeddings from market graphs.
- **Vector Database**: Store embeddings along with future context metrics (e.g., ATR, Rate of Change, etc.).
- **Retrieval**: Retrieve similar past embeddings based on a query, and pass associated future context to the DRL agent.

---

## 1. Graph Construction from Multi-Timeframe OHLCV Data

### **Input Data**
- The system should accept synchronized OHLCV dataframes for multiple timeframes, such as:
    - **5m**, **30m**, **1h**, **4h**
- Each timeframe's data includes:
    - Timestamp, OHLCV (Open, High, Low, Close, Volume)
    - Optional indicators (e.g., ATR, RVI, etc.)
- **Output**: A graph representation where each candle is a node, and temporal/cross-timeframe relations form the edges.

### **Graph Building Logic**
- **Nodes**: Each candle is a node, and each node has the following features:
    - OHLCV values
    - Timeframe ID (e.g., "5m", "30m", etc.)
    - Normalized timestamp (decay factor)
    - Optional indicators (e.g., ATR, RVI, etc.)
- **Edges**:
    - Temporal edges: connect consecutive candles within the same timeframe.
    - Cross-timeframe edges: link candles across different timeframes.
    - Edge weights can be optional, like volume-based or time-distance-based.

---

## 2. GNN Encoder for Market Graph

### **GNN Model**
- Use a Graph Neural Network (e.g., GraphSAGE, GAT, or GIN) to encode the market structure into a dense, fixed-size embedding vector.
- Input: Graph constructed from the multi-timeframe data.
- Output: A single embedding tensor of shape `(embedding_dim,)`.

**Steps**:
1. Define the GNN architecture using `torch-geometric` or `DGL`.
2. Pass the market graph through the GNN to obtain the embedding.
3. Optionally apply a projection head or normalization layer (e.g., `LayerNorm`).

---

## 3. Vector Database for Pattern Memory and Future Context

### **Embedding Storage**
- After encoding the market graph, store the resulting embeddings along with future context metrics (e.g., ATR, RoC, RVI, etc.) into a Vector Database such as **FAISS** or **Pinecone**.
- Each stored entry should include:
    - The **embedding**: output from the GNN.
    - **Metadata**: timestamp, asset symbol, timeframe.
    - **Future context**: ATR, Rate of Change, RVI, or any other metric that describes future movement.

**Vector Database Schema**:

| Embedding | Context (ATR, RoC, RVI) | Metadata (Timestamp, Symbol, Timeframe) |
|-----------|-------------------------|----------------------------------------|
| `[0.1, 0.3, ...]` | `{ 'ATR': 0.02, 'RoC': 0.05, 'RVI': 70.2 }` | `{"timestamp": "2025-04-24 10:00", "symbol": "BTCUSD", "timeframe": "5m"}` |

### **Embedding Retrieval**
- During each environment step, query the vector database:
    - Use the current embedding (obtained from the GNN) to find the K most similar past market structures.
    - Retrieve not only the top-K embeddings but also their corresponding **future context** metrics.

---

## 4. API Methods for GNN and Vector DB Integration

### **Methods Overview**
The `MultiTimeframeGNNMemoryEncoder` class should have the following methods:

1. **encode_market_state**: Encodes the current multi-timeframe market state into an embedding.
2. **store_embedding_with_context**: Stores the generated embedding and future context into the Vector Database.
3. **query_similar_patterns**: Retrieves the top-K similar embeddings and their associated future context from the Vector Database.

### **Class Definition**

```python
import torch
import torch_geometric
import faiss  # or pinecone

class MultiTimeframeGNNMemoryEncoder:
    def __init__(self, gnn_model, vector_db):
        self.gnn_model = gnn_model  # Pretrained GNN model (e.g., GAT, GraphSAGE)
        self.vector_db = vector_db  # Vector database instance (e.g., FAISS or Pinecone)

    def encode_market_state(self, multi_tf_ohlcv: dict) -> torch.Tensor:
        """
        Encodes the market state (multi-timeframe OHLCV data) into a single embedding tensor.
        """
        # Step 1: Construct the market graph from OHLCV data
        graph = build_graph_from_ohlcv(multi_tf_ohlcv)

        # Step 2: Pass the graph through the GNN to get the embedding
        embedding = self.gnn_model(graph)

        return embedding

    def store_embedding_with_context(self, embedding: torch.Tensor, context: dict, metadata: dict):
        """
        Store the GNN embedding along with future context and metadata in the Vector DB.
        """
        vector = embedding.cpu().numpy().astype('float32')
        metadata["context"] = context
        self.vector_db.add(vector, metadata)  # Vector DB specific function

    def query_similar_patterns(self, embedding: torch.Tensor, k: int = 5) -> list:
        """
        Query the vector database for the top-K most similar embeddings and their associated context.
        """
        vector = embedding.cpu().numpy().astype('float32')
        results = self.vector_db.query(vector, k)  # Vector DB specific function

        return results
```
### **Helper Function: Graph Construction**
```python
import networkx as nx

def build_graph_from_ohlcv(multi_tf_ohlcv: dict) -> nx.Graph:
    """
    Build the graph representation from OHLCV data across multiple timeframes.
    """
    G = nx.Graph()

    for timeframe, ohlcv_data in multi_tf_ohlcv.items():
        # Convert OHLCV to nodes (one node per candle)
        for idx, row in ohlcv_data.iterrows():
            node_id = f"{timeframe}_{idx}"
            G.add_node(node_id, features=row.values.tolist())

        # Add temporal edges (intra-timeframe)
        for idx in range(1, len(ohlcv_data)):
            G.add_edge(f"{timeframe}_{idx-1}", f"{timeframe}_{idx}")

        # Cross-timeframe edges can be added here if needed

    return G
```

## 5. Usage in DRL Agent
### Environment Integration
Your DRL agent will interact with the MultiTimeframeGNNMemoryEncoder module during each environment step:
1. Collect the current market window of OHLCV data across multiple timeframes.
2. Pass this data to encode_market_state() to obtain the current market embedding.
3. Use query_similar_patterns() to retrieve similar historical patterns and their future context.
4. Combine the current embedding and retrieved context as input to the DRL agent.

Example of environment interaction:

```python
class TradingEnv(gym.Env):
    def __init__(self, multi_timeframe_gnn_encoder):
        self.gnn_encoder = multi_timeframe_gnn_encoder
        # Initialize your trading environment specifics here

    def step(self, action):
        # Collect OHLCV data for current market window
        multi_tf_ohlcv = get_ohlcv_data()

        # Get GNN embedding for the current market state
        current_embedding = self.gnn_encoder.encode_market_state(multi_tf_ohlcv)

        # Query similar patterns from the vector DB
        similar_patterns = self.gnn_encoder.query_similar_patterns(current_embedding, k=5)

        # Combine embedding and retrieved context to form the observation
        observation = combine_embedding_with_context(current_embedding, similar_patterns)

        # The agent proceeds with its action and we return the observation
        return observation, reward, done, info

```

## 6. Testing and Debugging
1. Test graph construction: Ensure that the graph is correctly built from OHLCV data.
2. Test GNN encoding: Verify that the GNN produces meaningful embeddings.
3. Test vector database integration: Confirm that embeddings are stored and retrieved properly.
4. Test DRL agent: Ensure that the agent can make decisions using the augmented observations.

# Deliverables
- Python module: multi_timeframe_gnn_memory_encoder.py
- GNN-based encoder and helper functions
- Integration with FAISS for vector storage
- Integration with gym environment
