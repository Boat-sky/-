import torch
from Params import args
from Model import Model, GTLayer
from DataHandler import DataHandler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Now using {device}.")

# Load data 
handler = DataHandler()
handler.LoadData()

# Load trained model
checkpoint = torch.load("/public/home/zhemodedc/tianchi/GFormer/GFormer-main/Models/tem.mod", map_location=device)
gt_layer = GTLayer().to(device)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Prepare input graph structure
sub = handler.torchBiAdj
cmp = handler.allOneAdj
encoderAdj = handler.torchBiAdj


# Predict for specific user and item
user_id = 5
item_id = 42

with torch.no_grad():
    user_embeds, item_embeds, _, _ = model(
        handler = handler,
        is_test = True,
        sub = sub,
        cmp = cmp,
        encoderAdj = encoderAdj
    )

# Compute the score
user_embed = user_embeds[user_id]
item_embed = item_embeds[item_id]
score = torch.dot(user_embed, item_embed).item()
print(f"Predicted interaction score for user {user_id} and item {item_id}: {score}")

# Step 5: Recommend top-k items for a user
scores = torch.matmul(user_embeds[user_id], item_embeds.T)
top_k = torch.topk(scores, k=10)
recommended_items = top_k.indices.cpu().numpy()
print(f"Top recommended items for user {user_id}: {recommended_items}")    

print("All OK!")

