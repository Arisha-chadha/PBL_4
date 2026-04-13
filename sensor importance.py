base_score = evaluate_features(feature_cols)
print("All features R2:", base_score)

results = []

for feature in feature_cols:

    new_features = [f for f in feature_cols if f != feature]

    score = evaluate_features(new_features)

    drop = base_score - score

    results.append((feature, score, drop))

    print(f"Removed {feature} → R2: {score:.4f} | Drop: {drop:.4f}")
