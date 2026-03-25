# def fusion_score(spatial, frequency, biological):
#     final_score = ((0.4 * spatial)+(0.3 * frequency) +(0.3 * biological)
#     )
#     return final_score

# def fusion_score(spatial, frequency, biological):
#     import math
    
#     spatial = float(spatial)
#     frequency = float(frequency)
#     biological = float(biological)

#     # Normalize frequency (sigmoid scaling)
#     frequency_norm = 1 / (1 + math.exp(-frequency / 50))

#     # Optional safety clamp (keeps values between 0 and 1)
#     spatial = max(0, min(spatial, 1))
#     biological = max(0, min(biological, 1))

#     final_score = (
#         (0.4 * spatial) +
#         (0.3 * frequency_norm) +
#         (0.3 * biological)
#     )

#     return float(final_score)

def fusion_score(spatial, frequency, biological):
     import math

     spatial = float(spatial)
     frequency = float(frequency)
     biological = float(biological)

     # Normalize frequency
     frequency_norm = 1 / (1 + math.exp(-frequency / 50))

     # Clamp
     spatial = max(0, min(spatial, 1))
     biological = max(0, min(biological, 1))

     # Adjusted weights
     final_score = (
         (0.3 * spatial) +
         (0.4 * frequency_norm) +
         (0.3 * biological)
     )
     return round(final_score, 3)

 # Convert all to float

# def fusion_score(spatial, frequency, biological):
#     import math
#     spatial = float(spatial)
#     frequency = float(frequency)
#     biological = float(biological)

#     # Normalize frequency properly
#     frequency_norm = 1 / (1 + math.exp(- (frequency - 70) / 15))  
#     # shift & scale to better separate typical values
#     # tweak 70 and 15 based on your dataset

#     # Clamp spatial & biological
#     spatial = max(0, min(spatial, 1))
#     biological = max(0, min(biological, 1))

#     # Adjusted weights
#     final_score = (0.7 * spatial) + (0.2 * frequency_norm) + (0.1 * biological)

#     # Optional: scale to 0–1 probability
#     final_score = 1 / (1 + math.exp(- (final_score - 0.5) * 6))  

#     return round(final_score, 3)