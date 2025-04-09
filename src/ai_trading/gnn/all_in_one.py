import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv
import numpy as np
import pandas as pd


def preprocess_market_data(df, pattern_length=7, prediction_length=7, similarity_threshold=0.8):
    patterns = []
    targets = []
    indices = []

    for i in range(len(df) - pattern_length - prediction_length):
        # Extract past 7-candle pattern
        pattern = df.iloc[i:i+pattern_length][["Open", "High", "Low", "Close"]].values.flatten()
        patterns.append(pattern)

        # Calculate future percentage price change
        future_close = df.iloc[i + pattern_length + prediction_length - 1]["Close"]
        current_close = df.iloc[i + pattern_length - 1]["Close"]
        price_change = (future_close - current_close) / current_close  # % change

        targets.append(price_change)
        indices.append(i)

    # Convert to PyTorch tensors
    x = torch.tensor(np.array(patterns), dtype=torch.float)
    y = torch.tensor(np.array(targets), dtype=torch.float).unsqueeze(1)  # Make it a column vector

    # Create edges based on cosine similarity
    edge_index = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            similarity = cosine_similarity(x[i], x[j], dim=0)
            if similarity > similarity_threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    print(f"Extracted {len(patterns)} patterns.")
    print(f"Edge connections created: {len(edge_index[0]) // 2}")

    return Data(x=x, y=y, edge_index=edge_index), indices

# Apply function
market_graph, pattern_indices = preprocess_market_data(df)
print("Graph Data:", market_graph)


class MarketGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MarketGNN, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1)  # Predict future price change

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.fc(x)  # Predict price movement
        return x

# Initialize model
input_dim = market_graph.x.shape[1]  # 28 (flattened OHLC of 7 candles)
hidden_dim = 16
output_dim = 8  # Compressed market pattern representation
gnn_model = MarketGNN(input_dim, hidden_dim, output_dim)

# Define optimizer and loss function
optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training function
def train_gnn(model, data, epochs=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = loss_fn(output, data.y)  # Now predicting future price movement
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

train_gnn(gnn_model, market_graph)

def query_gnn(model, current_state, historical_data):
    model.eval()
    with torch.no_grad():
        state_vector = torch.tensor(current_state.values.flatten(), dtype=torch.float)

        # Get embeddings for all patterns
        graph_embeddings = model(historical_data.x, historical_data.edge_index)

        # Compute similarity
        similarities = cosine_similarity(state_vector, graph_embeddings, dim=1)
        closest_pattern_idx = torch.argmax(similarities).item()
        
        predicted_movement = graph_embeddings[closest_pattern_idx].item()
        
        print(f"Closest past pattern index: {closest_pattern_idx}")
        print(f"Predicted price movement in next {7} candles: {predicted_movement:.4%}")
        
        return predicted_movement

# Get the most recent 7-candle pattern
current_market_state = df.iloc[-7:][["Open", "High", "Low", "Close"]]

# Query the GNN for prediction
gnn_prediction = query_gnn(gnn_model, current_market_state, market_graph)
